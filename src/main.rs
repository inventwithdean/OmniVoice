#![recursion_limit = "256"]

use burn::{
    Tensor,
    tensor::{Device, TensorData},
};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

mod config;
mod duration;
mod omnivoice;
mod qwen3;
mod tokenizer;

use audioadapter::Adapter;
use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler, audioadapter};
use serde::Deserialize;
use std::{error::Error, fs::File, io::BufReader, time::Instant, vec};

use crate::omnivoice::OmniVoiceModelConfig;

pub fn resample_audio(
    input: &[f32],
    source_rate: usize,
    target_rate: usize,
    channels: usize,
) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
    if source_rate == target_rate {
        return Ok(input.to_vec());
    }
    let input_frames = input.len() / channels;

    let mut resampler = Fft::<f32>::new(source_rate, target_rate, 1024, channels, FixedSync::Both)?;

    let input_adapter = InterleavedSlice::new(input, channels, input_frames)?;

    let output_adapter = resampler.process_all(&input_adapter, input_frames, None)?;

    let out_frames = output_adapter.frames();
    let out_channels = output_adapter.channels();
    let mut out_vec = Vec::with_capacity(out_frames * out_channels);

    for frame in 0..out_frames {
        for chan in 0..out_channels {
            out_vec.push(output_adapter.read_sample(chan, frame).unwrap());
        }
    }

    Ok(out_vec)
}

#[derive(Deserialize)]
struct InferenceArgs {
    ref_audio_file: String,
    ref_text: String,
    target_text: String,
    lang: Option<String>,
}

fn main() {
    let device = &Device::wgpu(Default::default());
    let model_config = config::get_config();
    let tokenizer_config = tokenizer::config::get_config();

    let mut omnivoice = OmniVoiceModelConfig::new().init(&model_config, &tokenizer_config, device);

    let mut tokenizer_store = SafetensorsStore::from_file("tokenizer.safetensors")
        .with_from_adapter(PyTorchToBurnAdapter);

    let tokenizer_result = omnivoice
        .audio_tokenizer
        .load_from(&mut tokenizer_store)
        .unwrap();
    println!(
        "Loaded tokenizer.safetensors: {} applied",
        tokenizer_result.applied.len()
    );

    let mut store = SafetensorsStore::from_file("model.safetensors")
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true);

    let result = omnivoice.load_from(&mut store).unwrap();
    println!("{}", result);

    println!("Applied: {} tensors", result.applied.len());
    println!("Errors: {:?}", result.errors);

    if result.is_success() {
        println!("All tensors loaded successfully!");
    }
    let args_file = File::open("args.json").expect("No args.json found!");
    let reader = BufReader::new(args_file);
    let args: InferenceArgs = serde_json::from_reader(reader).unwrap();

    let mut reader = WavReader::open(args.ref_audio_file).unwrap();
    let spec = reader.spec();
    println!(
        "Loaded test.wav: {} Hz, {} channels, {} bits",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );
    let mut num_channels = spec.channels as usize;
    let original_sample_rate = spec.sample_rate as usize;

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
    };
    let samples: Vec<f32> = if num_channels > 1 {
        samples.iter().step_by(num_channels).copied().collect()
    } else {
        samples
    };
    num_channels = 1;

    let target_base_rate = 24000;
    let resampled = resample_audio(
        &samples,
        original_sample_rate,
        target_base_rate,
        num_channels,
    )
    .unwrap();
    let num_frames = resampled.len() / num_channels;

    let mut planar_samples = vec![0.0f32; resampled.len()];
    for t in 0..num_frames {
        for c in 0..num_channels {
            planar_samples[c * num_frames + t] = resampled[t * num_channels + c];
        }
    }

    let input_tensor = Tensor::<3>::from_data(
        TensorData::new(planar_samples, [1, num_channels, num_frames]),
        device,
    );

    let ref_audio_waveform = input_tensor.squeeze_dim(1);

    println!("Starting generation...");
    let timer = Instant::now();

    let generated_waveform = omnivoice.generate_speech(
        &args.target_text,
        &args.ref_text,
        ref_audio_waveform,
        args.lang.as_deref(),
        12,
        3.0,
    );

    println!("Generation completed in: {:?}", timer.elapsed());

    let recon_data = generated_waveform.to_data();

    let recon_slice = recon_data.as_slice::<f32>().unwrap();

    let out_spec = WavSpec {
        channels: num_channels as u16,
        sample_rate: target_base_rate as u32,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create("output.wav", out_spec).unwrap();

    for &sample in recon_slice {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();

    println!("Successfully saved output.wav!");
}
