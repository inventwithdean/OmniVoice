use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{Bool, Device, Distribution, FloatDType, Int, TensorData, s},
};
use tokenizers::Tokenizer;

use crate::{
    duration::RuleDurationEstimator,
    qwen3::{Qwen3Model, Qwen3ModelConfig},
    tokenizer::model::{HiggsAudioV2TokenizerModel, HiggsAudioV2TokenizerModelConfig},
};

#[derive(Module, Debug)]
pub struct OmniVoiceModel {
    pub llm: Qwen3Model,
    audio_embeddings: Embedding,
    codebook_layer_offsets: Tensor<1, Int>,
    audio_heads: Linear,
    normalized_audio_codebook_weights: Vec<f64>,
    pub audio_tokenizer: HiggsAudioV2TokenizerModel,
    #[module(skip)]
    pub text_tokenizer: Tokenizer,
    #[module(skip)]
    duration_estimator: RuleDurationEstimator,
}

impl OmniVoiceModel {
    pub fn forward(
        &self,
        text_ids: Tensor<2, Int>,
        audio_ids: Tensor<3, Int>,
        attention_mask: Tensor<4, Bool>,
    ) -> Tensor<4> {
        // text_ids: (B, T)
        // audio_ids: (B, 8, audio_T)
        let text_embeddings = self.llm.embed_tokens.forward(text_ids.clone()); // (B, T, C)

        let offsets = self.codebook_layer_offsets.clone().reshape([
            1,
            self.codebook_layer_offsets.dims()[0],
            1,
        ]);
        let shifted_audio_ids = audio_ids + offsets;
        let [b, num_codebooks, audio_seq] = shifted_audio_ids.dims();
        let flattended_ids = shifted_audio_ids.reshape([b * num_codebooks, audio_seq]);
        let flat_embeds = self.audio_embeddings.forward(flattended_ids); // (B * num_codebooks, audio_seq, C)
        let c = flat_embeds.dims()[2];

        let audio_embeds = flat_embeds.reshape([b, num_codebooks, audio_seq, c]);

        let audio_embeds = audio_embeds.sum_dim(1).squeeze_dim(1); // (B, audio_seq, C)
        let input_embeds = Tensor::cat(vec![text_embeddings, audio_embeds], 1); // (B, text_seq+audio+seq, C)
        let total_seq = input_embeds.dims()[1];

        let position_ids = Tensor::arange(0..total_seq as i64, &text_ids.device())
            .unsqueeze_dim::<2>(0)
            .expand([b, total_seq]); // (B, total_seq)

        let hidden_states = self.llm.forward(position_ids, input_embeds, attention_mask);
        let logits = self.audio_heads.forward(hidden_states); // (B, T, num_audio_codebooks * audio_vocab_size)
        let audio_vocab_size = 1025;
        let logits = logits.reshape([b, total_seq, num_codebooks, audio_vocab_size]); // (B, total_seq, 8, 1025)

        logits.swap_dims(1, 2) // (B, 8, total_seq, 1025)
    }

    pub fn generate_iterative(
        &self,
        text_ids: Tensor<2, Int>,
        ref_audio_ids: Tensor<3, Int>,
        target_seq_len: usize,
        num_steps: usize,
        cfg_scale: f32,
    ) -> Tensor<3, Int> {
        let device = &text_ids.device();
        let [b, text_seq] = text_ids.dims();
        let [_, num_codebooks, ref_seq] = ref_audio_ids.dims();
        let mask_id: i64 = 1024;

        // Match OmniVoiceGenerationConfig defaults
        let t_shift = 0.1_f32;
        let layer_penalty_factor = 5.0_f32;
        let position_temperature = 5.0_f32;

        let mut target_audio: Tensor<3, Int> =
            Tensor::full([b, num_codebooks, target_seq_len], mask_id, device);

        let cond_total = text_seq + ref_seq + target_seq_len;
        let cond_mask = Tensor::<4>::ones([b, 1, cond_total, cond_total], device).bool();
        let uncond_mask = Tensor::<4>::ones([b, 1, target_seq_len, target_seq_len], device).bool();

        // Schedule over the pooled (C * T) token budget, matching Python.
        let total = target_seq_len * num_codebooks;
        let t_sched = get_time_steps(num_steps, t_shift);
        let mut schedule = vec![0usize; num_steps];
        let mut rem = total;
        for i in 0..num_steps {
            let n = if i == num_steps - 1 {
                rem
            } else {
                let delta = t_sched[i + 1] - t_sched[i];
                (((total as f32) * delta).ceil() as usize).min(rem)
            };
            schedule[i] = n;
            rem -= n;
        }

        let layer_ids: Tensor<3> = Tensor::arange(0..num_codebooks as i64, device)
            .reshape([1, num_codebooks, 1])
            .cast(FloatDType::F32);

        let empty_text = Tensor::<2, Int>::zeros([b, 0], device);

        for step in 0..num_steps {
            // Conditioned
            let cond_audio = Tensor::cat(vec![ref_audio_ids.clone(), target_audio.clone()], 2);
            let cond_logits = self.forward(text_ids.clone(), cond_audio, cond_mask.clone());
            let start = text_seq + ref_seq;
            let c_logits = cond_logits.slice(s![.., .., start..cond_total, ..]);

            // Unconditioned
            let u_logits = self.forward(
                empty_text.clone(),
                target_audio.clone(),
                uncond_mask.clone(),
            );

            // CFG
            let c_lp = burn::tensor::activation::log_softmax(c_logits, 3);

            let u_lp = burn::tensor::activation::log_softmax(u_logits, 3);

            let combined = c_lp.clone() + (c_lp - u_lp) * cfg_scale;

            let lp = burn::tensor::activation::log_softmax(combined, 3);

            // Mask the mask token column
            let lp = mask_out_token(lp, mask_id as usize);

            let pred: Tensor<3, Int> = lp.clone().argmax(3).squeeze_dim(3); // (B, C, T)
            let conf = lp.max_dim(3).squeeze_dim(3); // (B, C, T)

            // Layer penalty
            let mut scores = conf - layer_ids.clone() * layer_penalty_factor;

            // Gumbel noise for position selection
            if position_temperature > 0.0 {
                scores = gumbel_noise(scores, position_temperature, device);
            }

            // Freeze already unmasked positions
            let is_unmasked = target_audio.clone().not_equal_elem(mask_id);
            let scores = scores.mask_fill(is_unmasked, f32::NEG_INFINITY);

            let k = schedule[step];

            if k == 0 {
                println!("step {step}: k=0, skipping");
                continue;
            }

            let [_, c, t] = scores.dims();
            let total = c * t;

            let flat_scores = scores.reshape([b, total]); // (B, total)

            let flat_pred = pred.reshape([b, total]); // (B, total)

            // This is too slow.
            // let topk_vals = flat_scores.clone().topk(k, 1); // (B, k)
            // let threshold = topk_vals.clone().slice(s![.., (k - 1)..k]); // (B, 1)

            // This is fast. Works like a charm.
            let sorted_scores = flat_scores.clone().sort_descending(1);

            let threshold = sorted_scores.slice(s![.., (k - 1)..k]);

            let threshold = threshold.expand([b, total]); // (B, total)

            let mask = flat_scores.greater_equal(threshold); // (B, total) bool

            let flat_target = target_audio.clone().reshape([b, total]);
            let new_flat = flat_target.clone().mask_where(mask, flat_pred);

            target_audio = new_flat.reshape([b, num_codebooks, target_seq_len]);
        }

        target_audio
    }

    pub fn prepare_inference_tokens(
        &self,
        text: &str,
        ref_text: Option<&str>,
        lang: Option<&str>,
        instruct: Option<&str>,
        device: &Device,
    ) -> Tensor<2, Int> {
        let lang_str = lang.filter(|s| !s.is_empty()).unwrap_or("None");
        let instruct_str = instruct.filter(|s| !s.is_empty()).unwrap_or("None");

        let style_text = format!(
            "<|denoise|><|lang_start|>{lang_str}<|lang_end|>\
         <|instruct_start|>{instruct_str}<|instruct_end|>"
        );

        let full_text = match ref_text {
            Some(rt) if !rt.trim().is_empty() => format!("{} {}", rt.trim(), text.trim()),
            _ => text.trim().to_string(),
        };
        let wrapped = format!("<|text_start|>{full_text}<|text_end|>");

        let combined = format!("{}{}", style_text, wrapped);

        let encoding = self.text_tokenizer.encode(combined, true).expect("encode");
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let seq_len = ids.len();

        Tensor::<2, Int>::from_data(TensorData::new(ids, [1, seq_len]), device)
    }

    pub fn generate_speech(
        &self,
        target_text: &str,
        ref_text: &str,
        ref_audio_ids: Tensor<3, Int>,
        ref_duration: f32,
        lang: Option<&str>,
        num_steps: usize,
        cfg_scale: f32,
    ) -> Tensor<2> {
        let device = self.codebook_layer_offsets.device();

        let est_duration_seconds = self.duration_estimator.estimate_duration(
            target_text,
            ref_text,
            ref_duration,
            Some(2.0),
            1.0,
        );

        let target_seq_len = (est_duration_seconds * 25.0).round() as usize;

        let text_ids =
            self.prepare_inference_tokens(target_text, Some(ref_text), lang, None, &device);

        let target_audio_ids = self.generate_iterative(
            text_ids,
            ref_audio_ids,
            target_seq_len,
            num_steps,
            cfg_scale,
        );

        self.audio_tokenizer.decode(target_audio_ids).squeeze_dim(1) // (1, generated_samples)
    }

    // For caching
    pub fn get_ref_audio_codes(&self, ref_audio_waveform: Tensor<2>) -> Tensor<3, Int> {
        self.audio_tokenizer
            .encode(ref_audio_waveform.unsqueeze_dim(1)) // (1, 8, ref_seq)
    }
}

fn get_time_steps(num_step: usize, t_shift: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_step + 1);
    for i in 0..=num_step {
        let t = i as f32 / num_step as f32;
        out.push(t_shift * t / (1.0 + (t_shift - 1.0) * t));
    }
    out
}

// Was so wrong!
// fn gumbel_noise(scores: Tensor<3>, temperature: f32, device: &Device) -> Tensor<3> {
//     let [b, c, t] = scores.dims();
//     let u: Tensor<3> = Tensor::random([b, c, t], Distribution::Uniform(1e-10, 1.0), device);
//     let noise = -(u.log()).log();
//     scores / temperature + noise
// }

fn gumbel_noise(scores: Tensor<3>, temperature: f32, device: &Device) -> Tensor<3> {
    let [b, c, t] = scores.dims();
    let u: Tensor<3> = Tensor::random([b, c, t], Distribution::Uniform(1e-10, 1.0), device);
    let noise = -((-u.log()).log());
    scores / temperature + noise
}

fn mask_out_token(log_probs: Tensor<4>, mask_id: usize) -> Tensor<4> {
    let [_, _, _, v] = log_probs.dims();
    let device = log_probs.device();
    let mut data = vec![0.0f32; v];
    data[mask_id] = f32::NEG_INFINITY;
    let col = Tensor::<1>::from_floats(data.as_slice(), &device).reshape([1, 1, 1, v]);
    log_probs + col
}

#[derive(Config, Debug)]
pub struct OmniVoiceModelConfig {}

impl OmniVoiceModelConfig {
    pub fn init(
        &self,
        config: &crate::config::OmniVoiceConfig,
        tokenizer_config: &crate::tokenizer::config::HiggsAudioV2TokenizerConfig,
        text_tokenizer: Tokenizer,
        device: &Device,
    ) -> OmniVoiceModel {
        let mut normalized_audio_codebook_weights = vec![];
        let weights_sum = config.audio_codebook_weights.iter().sum::<usize>() as f64;
        for w in &config.audio_codebook_weights {
            normalized_audio_codebook_weights.push(*w as f64 / weights_sum);
        }

        OmniVoiceModel {
            llm: Qwen3ModelConfig::new().init(&config.llm_config, device),
            audio_embeddings: EmbeddingConfig::new(
                config.num_audio_codebook * config.audio_vocab_size,
                config.llm_config.hidden_size,
            )
            .init(device),
            codebook_layer_offsets: Tensor::arange(0..config.num_audio_codebook as i64, device)
                * config.audio_vocab_size as i64,
            audio_heads: LinearConfig::new(
                config.llm_config.hidden_size,
                config.num_audio_codebook * config.audio_vocab_size,
            )
            .with_bias(false)
            .init(device),
            normalized_audio_codebook_weights,
            text_tokenizer,
            audio_tokenizer: HiggsAudioV2TokenizerModelConfig::new().init(tokenizer_config, device),
            duration_estimator: RuleDurationEstimator {},
        }
    }
}
