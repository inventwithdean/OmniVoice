use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct DacConfig {
    pub codebook_dim: usize,
    pub codebook_loss_weight: f64,
    pub codebook_size: usize,
    pub commitment_loss_weight: f64,
    pub decoder_hidden_size: usize,
    pub downsampling_ratios: Vec<usize>,
    pub encoder_hidden_size: usize,
    pub hidden_size: usize,
    pub hop_length: usize,
    pub model_type: String,
    pub n_codebooks: usize,
    pub quantizer_dropout: usize,
    pub sampling_rate: usize,
    pub upsampling_ratios: Vec<usize>,
}
