use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{Device, activation::elu},
};

use crate::tokenizer::config::HiggsAudioV2TokenizerConfig;

#[derive(Module, Debug)]
pub struct HiggsAudioV2TokenizerResidualUnit {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl HiggsAudioV2TokenizerResidualUnit {
    pub fn forward(&self, hidden_state: Tensor<3>) -> Tensor<3> {
        let mut output_tensor = elu(hidden_state.clone(), 1.0);
        output_tensor = self.conv1.forward(output_tensor);
        output_tensor = elu(output_tensor, 1.0);
        output_tensor = self.conv2.forward(output_tensor);
        hidden_state + output_tensor
    }
}

#[derive(Config, Debug)]
pub struct HiggsAudioV2TokenizerResidualUnitConfig {
    in_channels: usize,
    out_channels: usize,
    dilation: usize,
}

impl HiggsAudioV2TokenizerResidualUnitConfig {
    pub fn init(
        &self,
        config: &HiggsAudioV2TokenizerConfig,
        device: &Device,
    ) -> HiggsAudioV2TokenizerResidualUnit {
        let padding = ((config.unit_kernel_size - 1) / 2) * self.dilation;
        HiggsAudioV2TokenizerResidualUnit {
            conv1: Conv1dConfig::new(self.in_channels, self.out_channels, config.unit_kernel_size)
                .with_padding(PaddingConfig1d::Explicit(padding, padding))
                .with_dilation(self.dilation)
                .with_bias(false)
                .init(device),
            conv2: Conv1dConfig::new(self.out_channels, self.out_channels, 1)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct HiggsAudioV2TokenizerSemanticEncoderBlock {
    res_units: Vec<HiggsAudioV2TokenizerResidualUnit>,
    conv: Conv1d,
}

impl HiggsAudioV2TokenizerSemanticEncoderBlock {
    pub fn forward(&self, mut hidden_state: Tensor<3>) -> Tensor<3> {
        for unit in &self.res_units {
            hidden_state = unit.forward(hidden_state);
        }
        self.conv.forward(hidden_state)
    }
}

#[derive(Config, Debug)]
pub struct HiggsAudioV2TokenizerSemanticEncoderBlockConfig {
    in_channels: usize,
    out_channels: usize,
    stride: usize,
}

impl HiggsAudioV2TokenizerSemanticEncoderBlockConfig {
    pub fn init(
        &self,
        config: &HiggsAudioV2TokenizerConfig,
        device: &Device,
    ) -> HiggsAudioV2TokenizerSemanticEncoderBlock {
        let mut res_units = vec![];
        for dilation in &config.block_dilations {
            res_units.push(
                HiggsAudioV2TokenizerResidualUnitConfig::new(
                    self.in_channels,
                    self.in_channels,
                    *dilation,
                )
                .init(config, device),
            );
        }
        let kernel = if self.stride == 1 { 3 } else { 2 * self.stride };
        let padding = (kernel - 1) / 2;
        HiggsAudioV2TokenizerSemanticEncoderBlock {
            res_units,
            conv: Conv1dConfig::new(self.in_channels, self.out_channels, kernel)
                .with_stride(self.stride)
                .with_padding(PaddingConfig1d::Explicit(padding, padding))
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct SemanticEncoder {
    conv: Conv1d,
    conv_blocks: Vec<HiggsAudioV2TokenizerSemanticEncoderBlock>,
}

impl SemanticEncoder {
    pub fn forward(&self, hidden_state: Tensor<3>) -> Tensor<3> {
        // hidden_state: (B, C, T)
        let mut hidden_state = self.conv.forward(hidden_state);
        for block in &self.conv_blocks {
            hidden_state = block.forward(hidden_state);
        }
        hidden_state
    }
}

#[derive(Config, Debug)]
pub struct SemanticEncoderConfig {}

impl SemanticEncoderConfig {
    pub fn init(&self, config: &HiggsAudioV2TokenizerConfig, device: &Device) -> SemanticEncoder {
        let semantic_hidden_size = config.semantic_model_config.hidden_size;
        let kernel = config.kernel_size;
        let padding = kernel / 2;
        let mut in_channels = semantic_hidden_size;
        let mut conv_blocks = vec![];
        for (i, stride) in config.strides.iter().enumerate() {
            let out_channels = semantic_hidden_size * config.channel_ratios[i];
            conv_blocks.push(
                HiggsAudioV2TokenizerSemanticEncoderBlockConfig::new(
                    in_channels,
                    out_channels,
                    *stride,
                )
                .init(config, device),
            );
            in_channels = out_channels;
        }
        SemanticEncoder {
            conv: Conv1dConfig::new(semantic_hidden_size, semantic_hidden_size, kernel)
                .with_padding(PaddingConfig1d::Explicit(padding, padding))
                .with_bias(false)
                .init(device),
            conv_blocks,
        }
    }
}
