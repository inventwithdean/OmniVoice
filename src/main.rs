#![recursion_limit = "256"]

use burn::{
    Tensor,
    module::{Module, Quantizer},
    store::ModuleRecord,
    tensor::{
        Device, TensorData,
        quantization::{Calibration, QuantScheme, ScaleDtype},
    },
};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use hf_hub::{
    HFClientSync,
    progress::{DownloadEvent, ProgressEvent, ProgressHandler},
};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

mod config;
mod duration;
mod omnivoice;
mod qwen3;
mod tokenizer;

use audioadapter::Adapter;
use audioadapter_buffers::direct::InterleavedSlice;
use rodio::{DeviceSinkBuilder, Player, buffer::SamplesBuffer};
use rubato::{Fft, FixedSync, Resampler, audioadapter};
use serde::Deserialize;
use std::{
    error::Error,
    fs::{self, File},
    io::{self, BufReader, Write},
    num::NonZero,
    sync::mpsc,
    thread, vec,
};
use tokenizers::Tokenizer;

use crate::omnivoice::{OmniVoiceModel, OmniVoiceModelConfig};

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
    lang: Option<String>,
}

struct PrintHandler;

impl ProgressHandler for PrintHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        match event {
            ProgressEvent::Download(DownloadEvent::Start {
                total_files: _,
                total_bytes,
            }) => {
                let total_size = *total_bytes as f64 / 1_000_000.0;
                println!("Downloading {total_size:.2} MB.");
            }
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
            }) => {
                let total_mb = *total_bytes as f64 / 1_000_000.0;
                let mb_completed = *bytes_completed as f64 / 1_000_000.0;
                let mut mb_per_sec = None;
                if let Some(bytes_per_sec) = bytes_per_sec {
                    mb_per_sec = Some(*bytes_per_sec as f64 / 1_000_000.0);
                }
                print!(
                    "\rProgress: {:.2}/{:.2} bytes ({:.2?} MB/s)",
                    mb_completed, total_mb, mb_per_sec
                );
                io::stdout().flush().unwrap();
            }
            ProgressEvent::Download(DownloadEvent::Complete) => {
                println!("\nDownload complete!");
            }
            _ => {}
        }
    }
}

fn main() {
    let device = Device::wgpu(Default::default());
    let model_config = config::get_config();
    let tokenizer_config = tokenizer::config::get_config();

    let client = HFClientSync::new().unwrap();
    let repo = client.model("inventwithdean", "OmniVoice");
    let model_path = repo
        .download_file()
        .filename("omnivoice.mpk")
        .progress(PrintHandler)
        .send()
        .unwrap();
    let tokenizer_path = repo
        .download_file()
        .filename("tokenizer.json")
        .progress(PrintHandler)
        .send()
        .unwrap();

    let text_tokenizer =
        Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer.json");

    let mut omnivoice =
        OmniVoiceModelConfig::new().init(&model_config, &tokenizer_config, text_tokenizer, &device);

    let record = ModuleRecord::load(model_path).expect("Failed to load the MPK file");
    omnivoice = omnivoice.load_record(record);

    // let mpk_path = "omnivoice.mpk";

    // if Path::new(mpk_path).exists() {
    //     println!("Loading {mpk_path}...");
    //     let record = ModuleRecord::load(mpk_path).expect("Failed to load the MPK file");
    //     omnivoice = omnivoice.load_record(record);
    // } else {
    //     println!("No MPK found. Loading safetensors, quantizing and saving...");
    //     convert_safetensors_to_mpk_save(omnivoice);
    //     println!("Saved. Exiting now!");
    //     exit(0);
    // }

    let args_file = File::open("args.json").expect("No args.json found!");
    let reader = BufReader::new(args_file);
    let args: InferenceArgs = serde_json::from_reader(reader).unwrap();

    let mut reader = WavReader::open(args.ref_audio_file).unwrap();
    let spec = reader.spec();
    println!(
        "Loaded wav: {} Hz, {} channels, {} bits",
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
        &device,
    );

    let ref_audio_waveform = input_tensor.squeeze_dim(1);

    let content_file_path = "data.txt";
    let content = fs::read_to_string(content_file_path).expect("Failed to read data.txt");

    process_play_and_save_paragraph(
        &omnivoice,
        &content,
        &args.ref_text,
        ref_audio_waveform,
        args.lang.as_deref(),
        "paragraph.wav",
    );
}

fn chunk_text(text: &str) -> Vec<String> {
    text.replace('\n', "|||")
        .replace(". ", ". |||")
        .replace("? ", "? |||")
        .replace("! ", "! |||")
        .split("|||")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

pub fn process_play_and_save_paragraph(
    omnivoice: &OmniVoiceModel,
    paragraph: &str,
    ref_text: &str,
    ref_audio_waveform: Tensor<2>,
    lang: Option<&str>,
    output_filename: &str,
) {
    let sentences = chunk_text(paragraph);

    let stream_handle =
        DeviceSinkBuilder::open_default_sink().expect("Failed to open default audio stream");
    let player = Player::connect_new(&stream_handle.mixer());

    let sample_rate = 24_000u32;
    println!("Starting generation and playback...");

    let num_samples = ref_audio_waveform.dims()[1];
    let ref_audio_duration = num_samples as f32 / sample_rate as f32;

    let omnivoice = std::sync::Arc::new(omnivoice.clone());
    let ref_audio_codes = omnivoice.get_ref_audio_codes(ref_audio_waveform);

    let (tx, rx) = mpsc::channel::<Vec<f32>>();

    let gen_model = omnivoice.clone();
    let gen_ref_text = ref_text.to_string();
    let gen_lang = lang.map(|s| s.to_string());
    thread::spawn(move || {
        for sentence in sentences {
            println!("Processing: {sentence}");
            let generated = gen_model.generate_speech(
                &sentence,
                &gen_ref_text,
                ref_audio_codes.clone(),
                ref_audio_duration,
                gen_lang.as_deref(),
                16,
                3.0,
            );
            let data = generated.into_data();
            let slice = data.as_slice::<f32>().unwrap();
            if tx.send(slice.to_vec()).is_err() {
                break;
            }
        }
    });

    let mut complete_audio_data = Vec::new();
    for audio_vec in rx {
        let _ = audio_vec.len();
        complete_audio_data.extend_from_slice(&audio_vec);

        let buffer = SamplesBuffer::new(
            NonZero::new(1).unwrap(),
            NonZero::new(sample_rate).unwrap(),
            audio_vec,
        );
        player.append(buffer);

        player.sleep_until_end();
    }

    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(output_filename, spec).unwrap();
    for sample in complete_audio_data {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
    println!("Successfully saved {output_filename}!");
}

fn _convert_safetensors_to_mpk_save(mut omnivoice: OmniVoiceModel) {
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

    // Quantization
    let mut quantizer = Quantizer::new(
        Calibration::MinMax,
        QuantScheme::default().per_tensor(ScaleDtype::F16),
    );

    // Quantize the transformers
    omnivoice.audio_tokenizer.semantic_model.encoder = omnivoice
        .audio_tokenizer
        .semantic_model
        .encoder
        .quantize_weights(&mut quantizer);

    omnivoice.llm = omnivoice.llm.quantize_weights(&mut quantizer);

    let quantized_record = omnivoice.clone().into_record();

    quantized_record
        .save("omnivoice.mpk")
        .expect("Failed to save model");
}
