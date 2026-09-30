use crate::config::LLMConfig;
use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{
        Bool, Device, FloatDType, Int,
        activation::{silu, softmax},
        s,
    },
};

#[derive(Module, Debug)]
pub struct Qwen3RMSNorm {
    weight: Param<Tensor<1>>,
    variance_epsilon: f64,
}

impl Qwen3RMSNorm {
    pub fn forward(&self, mut hidden_states: Tensor<3>) -> Tensor<3> {
        // hidden_states: (B, T, C)
        let input_dtype = hidden_states.dtype();
        hidden_states = hidden_states.cast(FloatDType::F32);
        let variance = hidden_states.clone().powf_scalar(2.0).mean_dim(2);
        hidden_states = hidden_states * (variance + self.variance_epsilon).sqrt().recip();
        let weight_broadcast = self.weight.val().reshape([1, 1, hidden_states.dims()[2]]); // (1, 1, hidden_size)
        // (1, 1, hidden_size) * (B, T, hidden_size)
        weight_broadcast * hidden_states.cast(input_dtype)
    }

    pub fn forward_qk(&self, mut hidden_states: Tensor<4>) -> Tensor<4> {
        // hidden_states: (B, T, -1, head_dim)
        let input_dtype = hidden_states.dtype();
        hidden_states = hidden_states.cast(FloatDType::F32);
        let variance = hidden_states.clone().powf_scalar(2.0).mean_dim(3);
        hidden_states = hidden_states * (variance + self.variance_epsilon).sqrt().recip();
        let weight_broadcast = self
            .weight
            .val()
            .reshape([1, 1, 1, hidden_states.dims()[3]]); // (1, 1, 1, hidden_size)
        // (1, 1, hidden_size) * (B, T, hidden_size)
        weight_broadcast * hidden_states.cast(input_dtype)
    }
}

#[derive(Config, Debug)]
pub struct Qwen3RMSNormConfig {
    hidden_size: usize,
    eps: f64,
}

impl Qwen3RMSNormConfig {
    pub fn init(&self, device: &Device) -> Qwen3RMSNorm {
        Qwen3RMSNorm {
            weight: Param::from_tensor(Tensor::ones([self.hidden_size], device)),
            variance_epsilon: self.eps,
        }
    }
}

// act_fn = silu
#[derive(Module, Debug)]
pub struct Qwen3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3MLP {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        // x: (B, T, hidden_size)
        let gate = silu(self.gate_proj.forward(x.clone())); // (B, T, intermediate_size)
        let up = self.up_proj.forward(x); // (B, T, intermediate_size)

        self.down_proj.forward(gate * up) // (B, T, hidden_size)
    }
}

#[derive(Config, Debug)]
pub struct Qwen3MLPConfig {}

impl Qwen3MLPConfig {
    pub fn init(&self, config: &LLMConfig, device: &Device) -> Qwen3MLP {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        Qwen3MLP {
            gate_proj: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(intermediate_size, hidden_size)
                .with_bias(false)
                .init(device),
        }
    }
}

// rope_type = default
#[derive(Module, Debug)]
pub struct Qwen3RotaryEmbedding {
    inv_freq: Tensor<1>, // [64] as head_dim is 128
}

impl Qwen3RotaryEmbedding {
    pub fn forward(&self, x: Tensor<3>, position_ids: Tensor<2, Int>) -> (Tensor<3>, Tensor<3>) {
        // x: (B, T, C), position_ids: (B, T)
        let inv = self
            .inv_freq
            .clone()
            .reshape([1, 1, self.inv_freq.dims()[0]])
            .cast(FloatDType::F32); // (1, 1, 64)

        let pos = position_ids.unsqueeze_dim(2).cast(FloatDType::F32); // (B, T, 1)

        // (B, T, 1) * (1, 1, 64) = (B, T, 64)
        let freqs = pos * inv;
        let emb = Tensor::cat(vec![freqs.clone(), freqs], 2); // (B, T, 128)

        let cos = emb.clone().cos().cast(x.dtype());
        let sin = emb.sin().cast(x.dtype());
        (cos, sin)
    }
}

pub fn rotate_half(x: Tensor<4>) -> Tensor<4> {
    // x: (B, T, num_heads, head_dim)
    let dim = x.dims()[3];
    let slice = dim / 2;
    let x1 = x.clone().slice(s![.., .., .., 0..slice]);
    let x2 = x.slice(s![.., .., .., slice..dim]);
    Tensor::cat(vec![-x2, x1], 3)
}

pub fn apply_rotary_pos_emb(
    q: Tensor<4>,
    k: Tensor<4>,
    cos: Tensor<3>,
    sin: Tensor<3>,
) -> (Tensor<4>, Tensor<4>) {
    let cos = cos.unsqueeze_dim(1); // (B, 1, T, 128)
    let sin = sin.unsqueeze_dim(1); // (B, 1, T, 128)
    let q_embed = (q.clone() * cos.clone()) + (rotate_half(q) * sin.clone());
    let k_embed = (k.clone() * cos) + (rotate_half(k) * sin);
    (q_embed, k_embed)
}

#[derive(Config, Debug)]
pub struct Qwen3RotaryEmbeddingConfig {}

impl Qwen3RotaryEmbeddingConfig {
    pub fn init(&self, config: &LLMConfig, device: &Device) -> Qwen3RotaryEmbedding {
        let base = Tensor::<1>::from_floats([config.rope_parameters.rope_theta as f32], device);
        let dim = config.head_dim as i64;
        let aranged = Tensor::arange_step(0..dim, 2, device).cast(FloatDType::F32);
        let inv_freq = 1.0 / (base.powf(aranged / dim as f64));
        Qwen3RotaryEmbedding { inv_freq }
    }
}

// layer_type = full_attention for every layer
// sliding_window = None
#[derive(Module, Debug)]
pub struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Qwen3RMSNorm,
    k_norm: Qwen3RMSNorm,
}

impl Qwen3Attention {
    pub fn forward(
        &self,
        hidden_states: Tensor<3>,
        position_embeddings: (Tensor<3>, Tensor<3>),
        attention_mask: Tensor<4, Bool>,
    ) -> Tensor<3> {
        // hidden_states: (B, T, hidden_size)
        // position_embeddings: (B, T, 128), (B, T, 128)
        // attention_mask: (B, 1, T, T)
        let [b, t, _c] = hidden_states.dims();

        // C = hidden_size = 1024
        let head_dim = 128 as usize;
        let num_attention_heads = 16 as usize;
        let num_key_value_heads = 8 as usize;

        // q_proj will output (B, T, num_attention_heads * head_dim) which is (B, T, 2048)

        let query_states = self.q_proj.forward(hidden_states.clone()); // (B, T, 2048)
        let query_states = query_states.reshape([b, t, num_attention_heads, head_dim]); // (B, T, 16, 128)
        let query_states = self.q_norm.forward_qk(query_states).swap_dims(1, 2); // (B, 16, T, 128)

        // k_proj and v_proj will output (B, T, num_key_values_heads * head_dim)
        let key_states = self.k_proj.forward(hidden_states.clone()); // (B, T, 1024)
        let key_states = key_states.reshape([b, t, num_key_value_heads, head_dim]); // (B, T, 8, 128)
        let key_states = self.k_norm.forward_qk(key_states).swap_dims(1, 2); // (B, 8, T, 128)

        let value_states = self.v_proj.forward(hidden_states.clone()); // (B, T, 1024)
        let value_states = value_states.reshape([b, t, num_key_value_heads, head_dim]); // (B, T, 8, 128)
        let value_states = value_states.swap_dims(1, 2); // (B, 8, T, 128)

        let (cos, sin) = position_embeddings;
        let (query_states, key_states) = apply_rotary_pos_emb(query_states, key_states, cos, sin);

        // Grouped Query Attention
        let num_key_value_groups = num_attention_heads / num_key_value_heads;
        let key_states = repeat_kv(key_states, num_key_value_groups); // (B, 16, T, 128)
        let value_states = repeat_kv(value_states, num_key_value_groups); // (B, 16, T, 128)

        // (B, 16, T, 128) @ (B, 16, 128, T) => (B, 16, T, T)
        let attn_weights = query_states.matmul(key_states.transpose()) / (head_dim as f64).sqrt();
        let attn_weights = attn_weights.cast(FloatDType::F32); // cast up to f32

        let attention = attn_weights.mask_fill(attention_mask.bool_not(), -f32::INFINITY);
        let attention = softmax(attention, 3); // (B, 16, T, T) now with last dim normalized to sum to 1
        let attention = attention.cast(hidden_states.dtype()); // (cast back down)

        // (B, 16, T, T) @ (B, 16, T, 128) => (B, 16, T, 128)
        let attn_output = attention.matmul(value_states); // (B, 16, T, 128)
        let attn_output = attn_output.swap_dims(1, 2); // (B, T, 16, 128);
        let attn_output = attn_output.reshape([b, t, num_attention_heads * head_dim]); // (B, T, 2048)
        self.o_proj.forward(attn_output) // (B, T, hidden_size)
    }
}

// n_rep is 2
fn repeat_kv(hidden_states: Tensor<4>, n_rep: usize) -> Tensor<4> {
    // hidden_states: (B, 8, T, 128)
    let [b, num_key_value_heads, t, head_dim] = hidden_states.dims();
    if n_rep == 1 {
        return hidden_states;
    }
    let hidden_states: Tensor<5> = hidden_states.unsqueeze_dim(2); // (B, 8, 1, T, 128)
    let hidden_states = hidden_states.expand([b, num_key_value_heads, n_rep, t, head_dim]); // (B, 8, n_rep, T, 128)
    hidden_states.reshape([b, num_key_value_heads * n_rep, t, head_dim]) // (B, 8*n_rep, T, 128)
}

#[derive(Config, Debug)]
pub struct Qwen3AttentionConfig {}

impl Qwen3AttentionConfig {
    pub fn init(&self, config: &LLMConfig, device: &Device) -> Qwen3Attention {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let attention_bias = config.attention_bias;
        let rms_norm_eps = config.rms_norm_eps;

        Qwen3Attention {
            q_proj: LinearConfig::new(hidden_size, num_attention_heads * head_dim)
                .with_bias(attention_bias)
                .init(device),
            k_proj: LinearConfig::new(hidden_size, num_key_value_heads * head_dim)
                .with_bias(attention_bias)
                .init(device),
            v_proj: LinearConfig::new(hidden_size, num_key_value_heads * head_dim)
                .with_bias(attention_bias)
                .init(device),
            o_proj: LinearConfig::new(num_attention_heads * head_dim, hidden_size)
                .with_bias(attention_bias)
                .init(device),
            q_norm: Qwen3RMSNormConfig::new(head_dim, rms_norm_eps).init(device),
            k_norm: Qwen3RMSNormConfig::new(head_dim, rms_norm_eps).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Qwen3DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    input_layernorm: Qwen3RMSNorm,
    post_attention_layernorm: Qwen3RMSNorm,
}

impl Qwen3DecoderLayer {
    pub fn forward(
        &self,
        hidden_states: Tensor<3>,
        attention_mask: Tensor<4, Bool>,
        position_embeddings: (Tensor<3>, Tensor<3>),
    ) -> Tensor<3> {
        let residual = hidden_states.clone();

        let mut hidden_states = self.input_layernorm.forward(hidden_states);
        hidden_states = self
            .self_attn
            .forward(hidden_states, position_embeddings, attention_mask);
        hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let mut hidden_states = self.post_attention_layernorm.forward(hidden_states);
        hidden_states = self.mlp.forward(hidden_states);
        residual + hidden_states
    }
}

#[derive(Config, Debug)]
pub struct Qwen3DecoderLayerConfig {}

impl Qwen3DecoderLayerConfig {
    pub fn init(&self, config: &LLMConfig, device: &Device) -> Qwen3DecoderLayer {
        let hidden_size = config.hidden_size;
        let rms_norm_eps = config.rms_norm_eps;
        Qwen3DecoderLayer {
            self_attn: Qwen3AttentionConfig::new().init(config, device),
            mlp: Qwen3MLPConfig::new().init(config, device),
            input_layernorm: Qwen3RMSNormConfig::new(hidden_size, rms_norm_eps).init(device),
            post_attention_layernorm: Qwen3RMSNormConfig::new(hidden_size, rms_norm_eps)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Qwen3Model {
    // Shape: (151676, 1024)
    pub embed_tokens: Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: Qwen3RMSNorm,
    rotary_emb: Qwen3RotaryEmbedding,
}

impl Qwen3Model {
    pub fn forward(
        &self,
        position_ids: Tensor<2, Int>,
        input_embeds: Tensor<3>,
        attention_mask: Tensor<4, Bool>,
    ) -> Tensor<3> {
        let mut hidden_states = input_embeds;
        let position_embeddings = self.rotary_emb.forward(hidden_states.clone(), position_ids);

        for decoder_layer in &self.layers {
            hidden_states = decoder_layer.forward(
                hidden_states,
                attention_mask.clone(),
                position_embeddings.clone(),
            );
        }

        self.norm.forward(hidden_states)
    }
}

#[derive(Config, Debug)]
pub struct Qwen3ModelConfig {}

impl Qwen3ModelConfig {
    pub fn init(&self, config: &LLMConfig, device: &Device) -> Qwen3Model {
        let mut layers = vec![];
        for _ in 0..config.num_hidden_layers {
            layers.push(Qwen3DecoderLayerConfig::new().init(config, device));
        }
        Qwen3Model {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers,
            norm: Qwen3RMSNormConfig::new(config.hidden_size, config.rms_norm_eps).init(device),
            rotary_emb: Qwen3RotaryEmbeddingConfig::new().init(config, device),
        }
    }
}
