// Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py
use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{
        GroupNorm, GroupNormConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{
        Device,
        activation::{gelu, softmax},
        module::conv1d,
        ops::ConvOptions,
        s,
    },
};

use crate::tokenizer::hubert::config::HubertConfig;

// Combined Wav2Vec2GroupNormConvLayer and Wav2Vec2NoLayerNormConvLayer
// as the only difference is the layer_norm. Burn will automatically plug in weights during loading.
#[derive(Module, Debug)]
pub struct Wav2Vec2ConvLayer {
    conv: Conv1d,
    layer_norm: Option<GroupNorm>,
}

impl Wav2Vec2ConvLayer {
    pub fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let mut hidden_states = self.conv.forward(hidden_states);
        if let Some(layer_norm) = &self.layer_norm {
            hidden_states = layer_norm.forward(hidden_states);
        }
        gelu(hidden_states)
    }
}

// feat_extract_activation = gelu,
#[derive(Config, Debug)]
pub struct Wav2Vec2ConvLayerConfig {}

impl Wav2Vec2ConvLayerConfig {
    pub fn init(
        &self,
        layer_id: usize,
        layer_norm: bool,
        config: &HubertConfig,
        device: &Device,
    ) -> Wav2Vec2ConvLayer {
        let in_conv_dim = if layer_id > 0 {
            config.conv_dim[layer_id - 1]
        } else {
            1
        };
        let out_conv_dim = config.conv_dim[layer_id];
        let kernel_size = config.conv_kernel[layer_id];
        let stride = config.conv_stride[layer_id];
        let bias = config.conv_bias;

        Wav2Vec2ConvLayer {
            conv: Conv1dConfig::new(in_conv_dim, out_conv_dim, kernel_size)
                .with_stride(stride)
                .with_bias(bias)
                .init(device),
            layer_norm: match layer_norm {
                true => Some(GroupNormConfig::new(out_conv_dim, out_conv_dim).init(device)),
                false => None,
            },
        }
    }
}

// feat_extract_norm = "group"
#[derive(Module, Debug)]
pub struct Wav2Vec2FeatureEncoder {
    conv_layers: Vec<Wav2Vec2ConvLayer>,
}

impl Wav2Vec2FeatureEncoder {
    pub fn forward(&self, input_values: Tensor<3>) -> Tensor<3> {
        let mut hidden_states = input_values;
        for conv_layer in &self.conv_layers {
            hidden_states = conv_layer.forward(hidden_states);
        }
        hidden_states
    }
}

#[derive(Config, Debug)]
pub struct Wav2Vec2FeatureEncoderConfig {}

impl Wav2Vec2FeatureEncoderConfig {
    pub fn init(&self, config: &HubertConfig, device: &Device) -> Wav2Vec2FeatureEncoder {
        let mut conv_layers = vec![Wav2Vec2ConvLayerConfig::new().init(0, true, config, device)];
        let num_feat_extract_layers = config.num_feat_extract_layers;
        for i in 1..num_feat_extract_layers {
            conv_layers.push(Wav2Vec2ConvLayerConfig::new().init(i, false, config, device));
        }
        Wav2Vec2FeatureEncoder { conv_layers }
    }
}

#[derive(Module, Debug)]
pub struct Wav2Vec2FeatureProjection {
    layer_norm: LayerNorm,
    projection: Linear,
}

impl Wav2Vec2FeatureProjection {
    // ⚠️: layer_norm expects (B, T, C) not (B, C, T)
    // Make sure to transpose before calling this forward
    pub fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let norm_hidden_states = self.layer_norm.forward(hidden_states);
        self.projection.forward(norm_hidden_states)
    }
}

#[derive(Config, Debug)]
pub struct Wav2Vec2FeatureProjectionConfig {}

impl Wav2Vec2FeatureProjectionConfig {
    pub fn init(&self, config: &HubertConfig, device: &Device) -> Wav2Vec2FeatureProjection {
        let d_model = config
            .conv_dim
            .last()
            .expect("conv_dim shouldn't be empty!")
            .to_owned();
        let hidden_size = config.hidden_size;
        let eps = config.layer_norm_eps;
        Wav2Vec2FeatureProjection {
            layer_norm: LayerNormConfig::new(d_model).with_epsilon(eps).init(device),
            projection: LinearConfig::new(d_model, hidden_size).init(device),
        }
    }
}

// is_decoder = False
// is_causal = False
// num_heads = 12
// embed_dim = 768
// head_dim = embed_dim // num_heads = 768 // 12 = 64
#[derive(Module, Debug)]
pub struct Wav2Vec2Attention {
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
}

impl Wav2Vec2Attention {
    pub fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        // hidden_states: [B, T, 768]
        let [b, t, embed_dim] = hidden_states.dims();
        let keys = self.k_proj.forward(hidden_states.clone()); // (B, T, 768)
        let queries = self.q_proj.forward(hidden_states.clone()); // (B, T, 768)
        let values = self.v_proj.forward(hidden_states.clone()); // (B, T, 768)

        // Hardcoding values here
        let num_heads: usize = 12;
        let head_dim: usize = 64;

        let mut values = values.reshape([b, t, num_heads, head_dim]); // (B, T, 12, 64)
        values = values.swap_dims(1, 2); // (B, 12, T, 64)
        let mut keys = keys.reshape([b, t, num_heads, head_dim]); // (B, T, 12, 64)
        keys = keys.swap_dims(1, 2); // (B, 12, T, 64)
        let mut queries = queries.reshape([b, t, num_heads, head_dim]); // (B, T, 12, 64)
        queries = queries.swap_dims(1, 2); // (B, 12, T, 64)

        // Transpose keys for attention
        keys = keys.transpose(); // (B, 12, 64, T)
        // Multihead attention
        // (B, 12, T, 64) @ (B, 12, 64, T) => (B, 12, T, T)
        let scores = queries.matmul(keys) / (head_dim as f64).sqrt(); // (B, 12, T, T)
        let scores = softmax(scores, 3); // (B, 12, T, T)

        // (B, 12, T, T) @ (B, 12, T, 64) => (B, 12, T, 64)
        let attention = scores.matmul(values); // (B, 12, T, 64)
        let attention = attention.swap_dims(1, 2); // (B, T, 12, 64)
        let attention = attention.reshape([b, t, embed_dim]); // (B, T, 768)
        self.out_proj.forward(attention)
    }
}

#[derive(Config, Debug)]
pub struct Wav2Vec2AttentionConfig {}

impl Wav2Vec2AttentionConfig {
    pub fn init(&self, config: &HubertConfig, device: &Device) -> Wav2Vec2Attention {
        let embed_dim = config.hidden_size;
        Wav2Vec2Attention {
            k_proj: LinearConfig::new(embed_dim, embed_dim).init(device),
            v_proj: LinearConfig::new(embed_dim, embed_dim).init(device),
            q_proj: LinearConfig::new(embed_dim, embed_dim).init(device),
            out_proj: LinearConfig::new(embed_dim, embed_dim).init(device),
        }
    }
}

// hidden_act = gelu
#[derive(Module, Debug)]
pub struct Wav2Vec2FeedForward {
    intermediate_dense: Linear,
    output_dense: Linear,
}

impl Wav2Vec2FeedForward {
    pub fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let mut hidden_states = self.intermediate_dense.forward(hidden_states);
        hidden_states = gelu(hidden_states);
        self.output_dense.forward(hidden_states)
    }
}

#[derive(Config, Debug)]
pub struct Wav2Vec2FeedForwardConfig {}

impl Wav2Vec2FeedForwardConfig {
    pub fn init(&self, config: &HubertConfig, device: &Device) -> Wav2Vec2FeedForward {
        let intermediate_size = config.intermediate_size;
        let hidden_size = config.hidden_size;
        Wav2Vec2FeedForward {
            intermediate_dense: LinearConfig::new(hidden_size, intermediate_size).init(device),
            output_dense: LinearConfig::new(intermediate_size, hidden_size).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Wav2Vec2EncoderLayer {
    attention: Wav2Vec2Attention,
    layer_norm: LayerNorm,
    feed_forward: Wav2Vec2FeedForward,
    final_layer_norm: LayerNorm,
}

impl Wav2Vec2EncoderLayer {
    pub fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let attn_residual = hidden_states.clone();
        let mut hidden_states = self.attention.forward(hidden_states);
        hidden_states = hidden_states + attn_residual;

        hidden_states = self.layer_norm.forward(hidden_states);
        hidden_states = hidden_states.clone() + self.feed_forward.forward(hidden_states);
        self.final_layer_norm.forward(hidden_states)
    }
}

#[derive(Config, Debug)]
pub struct Wav2Vec2EncoderLayerConfig {}

impl Wav2Vec2EncoderLayerConfig {
    pub fn init(&self, config: &HubertConfig, device: &Device) -> Wav2Vec2EncoderLayer {
        let embed_dim = config.hidden_size;
        let eps = config.layer_norm_eps;
        Wav2Vec2EncoderLayer {
            attention: Wav2Vec2AttentionConfig::new().init(config, device),
            layer_norm: LayerNormConfig::new(embed_dim)
                .with_epsilon(eps)
                .init(device),
            feed_forward: Wav2Vec2FeedForwardConfig::new().init(config, device),
            final_layer_norm: LayerNormConfig::new(embed_dim)
                .with_epsilon(eps)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct WeightParametrization {
    pub original0: Param<Tensor<3>>, // g: [1, 1, kernel_size]
    pub original1: Param<Tensor<3>>, // v: [out_channels, in_channels/groups, kernel_size]
}

#[derive(Module, Debug)]
pub struct Parametrizations {
    pub weight: WeightParametrization,
}

#[derive(Module, Debug)]
pub struct WeightNormConv1d {
    pub parametrizations: Parametrizations,
    pub bias: Param<Tensor<1>>,
    pub groups: usize,
    pub padding: usize,
}

impl WeightNormConv1d {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let g = self.parametrizations.weight.original0.val();
        let v = self.parametrizations.weight.original1.val();

        let v_norm = v.clone().powf_scalar(2.0).sum_dim(0).sum_dim(1).sqrt();

        let weight = v * (g / v_norm);
        let options = ConvOptions::new([1], [self.padding], [1], self.groups);

        conv1d(x, weight, Some(self.bias.val()), options)
    }
}

#[derive(Config, Debug)]
pub struct WeightNormConv1dConfig {
    pub channels: usize,
    pub kernel_size: usize,
    pub groups: usize,
    pub padding: usize,
}

impl WeightNormConv1dConfig {
    pub fn init(&self, device: &Device) -> WeightNormConv1d {
        let in_channels_per_group = self.channels / self.groups;

        let original1 = Tensor::zeros(
            [self.channels, in_channels_per_group, self.kernel_size],
            device,
        );
        let original0 = Tensor::zeros([1, 1, self.kernel_size], device);
        let bias = Tensor::zeros([self.channels], device);

        WeightNormConv1d {
            parametrizations: Parametrizations {
                weight: WeightParametrization {
                    original0: Param::from_tensor(original0),
                    original1: Param::from_tensor(original1),
                },
            },
            bias: Param::from_tensor(bias),
            groups: self.groups,
            padding: self.padding,
        }
    }
}

// feat_extract_activation = gelu
#[derive(Module, Debug)]
pub struct Wav2Vec2PositionalConvEmbedding {
    // ⚠️: Original safetensors model stores them as weight_v (The Filter Shape) and weight_g (The Magnitude) by the name original1 and original0
    // When converting to safetensors, make sure to remove parametrizations
    // Actually, updated to calculate the conv at runtime so don't have to convert safetensors
    conv: WeightNormConv1d,
}

impl Wav2Vec2PositionalConvEmbedding {
    pub fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let mut hidden_states = hidden_states.transpose();
        hidden_states = self.conv.forward(hidden_states);
        // Removed Wav2Vec2SamePadLayer and doing its operation directly
        hidden_states = hidden_states.slice(s![.., .., ..-1]);
        hidden_states = gelu(hidden_states);
        hidden_states.transpose()
    }
}

#[derive(Config, Debug)]
pub struct Wav2Vec2PositionalConvEmbeddingConfig {}

impl Wav2Vec2PositionalConvEmbeddingConfig {
    pub fn init(&self, config: &HubertConfig, device: &Device) -> Wav2Vec2PositionalConvEmbedding {
        let hidden_size = config.hidden_size;
        let kernel_size = config.num_conv_pos_embeddings;
        let padding = (config.num_conv_pos_embeddings as u32 / 2) as usize;
        let groups = config.num_conv_pos_embedding_groups;
        Wav2Vec2PositionalConvEmbedding {
            conv: WeightNormConv1dConfig::new(hidden_size, kernel_size, groups, padding)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Wav2Vec2Encoder {
    pos_conv_embed: Wav2Vec2PositionalConvEmbedding,
    layer_norm: LayerNorm,
    layers: Vec<Wav2Vec2EncoderLayer>,
}

impl Wav2Vec2Encoder {
    pub fn forward(&self, hidden_states: Tensor<3>) -> Vec<Tensor<3>> {
        let position_embeddings = self.pos_conv_embed.forward(hidden_states.clone());
        let mut hidden_states = hidden_states + position_embeddings;
        hidden_states = self.layer_norm.forward(hidden_states);

        let mut all_hidden_states = vec![];
        all_hidden_states.push(hidden_states.clone());

        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states);
            all_hidden_states.push(hidden_states.clone());
        }
        all_hidden_states
    }
}

#[derive(Config, Debug)]
pub struct Wav2Vec2EncoderConfig {}

impl Wav2Vec2EncoderConfig {
    pub fn init(&self, config: &HubertConfig, device: &Device) -> Wav2Vec2Encoder {
        let hidden_size = config.hidden_size;
        let eps = config.layer_norm_eps;
        let num_hidden_layers = config.num_hidden_layers;
        let mut layers = vec![];
        for _ in 0..num_hidden_layers {
            layers.push(Wav2Vec2EncoderLayerConfig::new().init(config, device));
        }

        Wav2Vec2Encoder {
            pos_conv_embed: Wav2Vec2PositionalConvEmbeddingConfig::new().init(config, device),
            layer_norm: LayerNormConfig::new(hidden_size)
                .with_epsilon(eps)
                .init(device),
            layers,
        }
    }
}
