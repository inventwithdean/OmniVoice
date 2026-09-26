// Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Device, Int, TensorData, ops::PadMode, s},
};

use crate::{
    resample_audio,
    tokenizer::config::HiggsAudioV2TokenizerConfig,
    tokenizer::dac::{
        decoder::{DacDecoder, DacDecoderConfig},
        encoder::{DacEncoder, DacEncoderConfig},
    },
    tokenizer::hubert::model::{HubertModel, HubertModelConfig},
    tokenizer::residual_vector_quantization::{
        HiggsAudioV2TokenizerResidualVectorQuantization,
        HiggsAudioV2TokenizerResidualVectorQuantizationConfig,
    },
    tokenizer::semantic_encoder::{SemanticEncoder, SemanticEncoderConfig},
};

#[derive(Module, Debug)]
pub struct HiggsAudioV2TokenizerModel {
    acoustic_encoder: DacEncoder,
    acoustic_decoder: DacDecoder,
    encoder_semantic: SemanticEncoder,
    semantic_model: HubertModel,
    fc: Linear,
    // fc1: Linear<B>, // Used for training for HuBERT features reconstruction
    fc2: Linear,
    quantizer: HiggsAudioV2TokenizerResidualVectorQuantization,
    semantic_downsample_factor: f64,
}

impl HiggsAudioV2TokenizerModel {
    pub fn extract_semantic_features(&self, input_values: Tensor<3>) -> Tensor<3> {
        // (B, 1, T)

        let input_mono = input_values.slice(s![.., 0..1, ..]);

        let [batch_size, num_channels, _original_time] = input_mono.dims();

        let tensor_data = input_mono.to_data();
        let input_slice = tensor_data.as_slice::<f32>().unwrap();

        let resampled_vec = resample_audio(input_slice, 24000, 16000, num_channels).unwrap();
        let new_time = resampled_vec.len() / num_channels;

        let mut input_values = Tensor::<3>::from_data(
            TensorData::new(resampled_vec, [batch_size, num_channels, new_time]),
            &input_mono.device(),
        );

        input_values = input_values.pad([(160, 160)], PadMode::Constant(0.0));

        let hidden_states = self.semantic_model.forward(input_values); // Get hidden_states
        // Stack into [B, 13, T, C]
        let stacked: Tensor<4> = Tensor::stack(hidden_states, 1);
        let mut semantic_features = stacked.mean_dim(1).squeeze_dim(1);
        if self.semantic_downsample_factor > 1.0 {
            semantic_features =
                semantic_features.slice(s![..,..;self.semantic_downsample_factor as usize,..]);
        }
        semantic_features
    }

    pub fn encode(&self, input_values: Tensor<3>) -> Tensor<3, Int> {
        // input_values: (B, 1, T)
        let e_semantic_input = self.extract_semantic_features(input_values.clone()); // (B, T, C)
        let mut e_semantic = self.encoder_semantic.forward(e_semantic_input.transpose());

        let mut e_acoustic = self.acoustic_encoder.forward(input_values);

        let t_semantic = e_semantic.dims()[2];
        let t_acoustic = e_acoustic.dims()[2];
        let min_len = std::cmp::min(t_semantic, t_acoustic);

        e_semantic = e_semantic.slice(s![.., .., 0..min_len]);
        e_acoustic = e_acoustic.slice(s![.., .., 0..min_len]);

        let mut embeddings = Tensor::cat(vec![e_acoustic, e_semantic], 1); // (B, hidden_size, T)
        embeddings = self.fc.forward(embeddings.transpose()).transpose(); // (B, hidden_size, T)
        let audio_codes = self.quantizer.encode(embeddings); // (B, NumQuantizers, T)

        audio_codes
    }

    pub fn decode(&self, audio_codes: Tensor<3, Int>) -> Tensor<3> {
        // audio_codes: (B, NumQuantizers, T)
        let quantized = self.quantizer.decode(audio_codes);
        let quantized_acoustic = self.fc2.forward(quantized.transpose()).transpose();
        self.acoustic_decoder.forward(quantized_acoustic)
    }
}

#[derive(Config, Debug)]
pub struct HiggsAudioV2TokenizerModelConfig {}

impl HiggsAudioV2TokenizerModelConfig {
    pub fn init(
        &self,
        config: &HiggsAudioV2TokenizerConfig,
        device: &Device,
    ) -> HiggsAudioV2TokenizerModel {
        let hidden_size =
            config.acoustic_model_config.hidden_size + config.semantic_model_config.hidden_size;

        let hop_length: usize = config
            .acoustic_model_config
            .downsampling_ratios
            .iter()
            .product();

        let semantic_downsample_factor = hop_length as f64
            / (config.sample_rate as f64 / config.semantic_sample_rate as f64)
            / config.downsample_factor as f64;

        HiggsAudioV2TokenizerModel {
            acoustic_encoder: DacEncoderConfig::new().init(&config.acoustic_model_config, device),
            acoustic_decoder: DacDecoderConfig::new().init(&config.acoustic_model_config, device),
            encoder_semantic: SemanticEncoderConfig::new().init(config, device),
            semantic_model: HubertModelConfig::new().init(&config.semantic_model_config, device),
            fc: LinearConfig::new(hidden_size, hidden_size).init(device),
            // fc1: LinearConfig::new(hidden_size, config.semantic_model_config.hidden_size)
            //     .init(device),
            fc2: LinearConfig::new(hidden_size, config.acoustic_model_config.hidden_size)
                .init(device),
            quantizer: HiggsAudioV2TokenizerResidualVectorQuantizationConfig::new()
                .init(config, device),
            semantic_downsample_factor,
        }
    }
}
