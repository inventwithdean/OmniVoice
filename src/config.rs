use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Serialize, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: usize,
    pub rope_type: String,
}

#[derive(Serialize, Deserialize)]
pub struct LLMConfig {
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub bos_token_id: usize,
    pub chunk_size_feed_forward: usize,
    pub eos_token_id: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub id2label: Value,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub is_encoder_decoder: bool,
    pub label2id: Value,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: usize,
    pub max_window_layers: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub output_attentions: bool,
    pub output_hidden_states: bool,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub use_sliding_window: bool,
    pub vocab_size: usize,
}

#[derive(Serialize, Deserialize)]
pub struct OmniVoiceConfig {
    pub audio_codebook_weights: Vec<usize>,
    pub audio_mask_id: usize,
    pub audio_vocab_size: usize,
    pub eos_token_id: usize,
    pub llm_config: LLMConfig,
    pub num_audio_codebook: usize,
    pub pad_token_id: usize,
}

pub fn get_config() -> OmniVoiceConfig {
    serde_json::from_value::<OmniVoiceConfig>(json!(
            {
      "architectures": [
        "OmniVoice"
      ],
      "audio_codebook_weights": [
        8,
        8,
        6,
        6,
        4,
        4,
        2,
        2
      ],
      "audio_mask_id": 1024,
      "audio_vocab_size": 1025,
      "bos_token_id": null,
      "dtype": "float32",
      "eos_token_id": 151645,
      "llm_config": {
        "_name_or_path": "",
        "architectures": [
          "Qwen3ForCausalLM"
        ],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "chunk_size_feed_forward": 0,
        "dtype": "float32",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "id2label": {
          "0": "LABEL_0",
          "1": "LABEL_1"
        },
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "is_encoder_decoder": false,
        "label2id": {
          "LABEL_0": 0,
          "LABEL_1": 1
        },
        "layer_types": [
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention",
          "full_attention"
        ],
        "max_position_embeddings": 40960,
        "max_window_layers": 28,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "output_attentions": false,
        "output_hidden_states": false,
        "pad_token_id": null,
        "problem_type": null,
        "return_dict": true,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
          "rope_theta": 1000000,
          "rope_type": "default"
        },
        "sliding_window": null,
        "tie_word_embeddings": true,
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 151676
      },
      "model_type": "omnivoice",
      "num_audio_codebook": 8,
      "pad_token_id": 151643,
      "transformers_version": "5.3.0"
    }
        ))
    .unwrap()
}
