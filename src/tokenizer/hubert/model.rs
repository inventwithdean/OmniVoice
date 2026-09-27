// Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/hubert/modular_hubert.py
use burn::{Tensor, config::Config, module::Module, tensor::Device};

use crate::tokenizer::hubert::{
    config::HubertConfig,
    wav2vec2::{
        Wav2Vec2Encoder, Wav2Vec2EncoderConfig, Wav2Vec2FeatureEncoder,
        Wav2Vec2FeatureEncoderConfig, Wav2Vec2FeatureProjection, Wav2Vec2FeatureProjectionConfig,
    },
};

#[derive(Module, Debug)]
pub struct HubertModel {
    feature_extractor: Wav2Vec2FeatureEncoder,
    feature_projection: Wav2Vec2FeatureProjection,
    pub encoder: Wav2Vec2Encoder,
}

impl HubertModel {
    pub fn forward(&self, input_values: Tensor<3>) -> Vec<Tensor<3>> {
        // input_values: (B, C, T)
        let mut extract_features = self.feature_extractor.forward(input_values);
        extract_features = extract_features.transpose(); // (B, T, C)
        let hidden_states = self.feature_projection.forward(extract_features);
        self.encoder.forward(hidden_states)
    }
}

#[derive(Config, Debug)]
pub struct HubertModelConfig {}

impl HubertModelConfig {
    pub fn init(&self, config: &HubertConfig, device: &Device) -> HubertModel {
        HubertModel {
            feature_extractor: Wav2Vec2FeatureEncoderConfig::new().init(config, device),
            feature_projection: Wav2Vec2FeatureProjectionConfig::new().init(config, device),
            encoder: Wav2Vec2EncoderConfig::new().init(config, device),
        }
    }
}
