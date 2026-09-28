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
    error::Error, fs::File, io::BufReader, num::NonZero, path::Path, process::exit, sync::mpsc,
    thread, vec,
};

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

fn main() {
    let device = Device::wgpu(Default::default());
    let model_config = config::get_config();
    let tokenizer_config = tokenizer::config::get_config();

    let mut omnivoice = OmniVoiceModelConfig::new().init(&model_config, &tokenizer_config, &device);

    let mpk_path = "omnivoice.mpk";

    if Path::new(mpk_path).exists() {
        println!("Loading {mpk_path}...");
        let record = ModuleRecord::load(mpk_path).expect("Failed to load the MPK file");
        omnivoice = omnivoice.load_record(record);
    } else {
        println!("No MPK found. Loading safetensors, quantizing and saving...");
        convert_safetensors_to_mpk_save(omnivoice);
        println!("Saved. Exiting now!");
        exit(0);
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
        &device,
    );

    let ref_audio_waveform = input_tensor.squeeze_dim(1);

    let paragraph = r#"
The morning sun pierced the thick emerald canopy of the Great Amazon, casting golden spotlights on a jungle that was already wide awake and extremely loud.

Swinging from a sturdy liana vine with far more enthusiasm than grace was Leo, a young jaguar whose spots were slightly lopsided and whose heart was roughly the size of a watermelon. He landed on a mossy branch with a heavy *thud*, nearly dislodging his best friend, Maya, a scarlet macaw with a beak sharper than a thorn and a wit to match.

"Smooth landing, fearless leader," Maya squawked, adjusting her ruffled feathers. "Only scared away three flocks of toucans and a family of capybaras with that one."

"The jungle needs its protectors, Maya!" Leo declared, puffing out his chest. "We have to be ready! Who knows what cries for help echo through the vines today? A lost monkey? A stranded frog? A—"

"HEEEEEELP!"

The cry was agonizingly drawn out, echoing from the branches below. It didn't sound like a creature in mortal peril; it sounded like a creature in deep, theatrical despair.

Leo and Maya scrambled down the great mahogany tree, bursting through the giant fern leaves to find Paco. Paco was a three-toed sloth who currently had one arm draped over his eyes in a pose of ultimate tragedy, hanging upside down at a speed that could only be described as geological.

"Paco! What is it?" Leo gasped, his claws gripping the bark. "Are the army ants marching? Did you drop your favorite leaf?"

"Worse," Paco whispered, opening one teary, oversized brown eye. "It’s... Valentina."

Maya rolled her eyes so hard she nearly fell off her perch. "Oh, feathers. Not the sloth romance again."

Valentina was the most beautiful sloth in the western canopy, known for her uniquely mossy fur and the fact that she could chew a hibiscus flower faster than anyone else in her family. Paco had been hopelessly in love with her since the rainy season, but his attempts to woo her had been disastrously slow. By the time he waved hello, she had usually migrated to another tree.

"I need the Moon-Kissed Bromeliad," Paco groaned, slowly extending a claw toward the absolute highest peak of the canopy, a dizzying height above them. "It blooms today. If I bring it to her, she will know my heart beats only for her... at a steady resting rate of ten beats per minute."

"A rescue mission for love!" Leo’s eyes went wide with starry-eyed heroism. "Do not fear, Paco! The Amazon Rescue Duo is on the case! To the Emergent Layer!"

"Leo, wait," Maya warned, but the jaguar was already scooping the sloth onto his back. Paco let out a slow-motion *“Whooooaaaa”* as Leo bounded up the trunk, his powerful legs propelling them upward through the humid air.

The journey was a vertical obstacle course. They dodged grumpy iguanas sunbathing on branches and leaped over rivers of marching ants. As they climbed higher, the air grew thinner and the branches much more precarious.

"Left, Leo! Watch the slippery orchids!" Maya directed from above, flying point. "Okay, now right! No, your *other* right!"

Leo misstepped, his back paw slipping on a patch of slick lichen. He scrambled, claws tearing bark, while Paco, completely unfazed, murmured, "Ah, the wind in my fur. So thrilling."

Finally, they reached the roof of the jungle. The view was breathtaking—a boundless sea of green stretching to the horizon. And there, perched on the very edge of a slender, swaying branch, was the Moon-Kissed Bromeliad. It was a radiant, glowing pink flower that smelled like vanilla and rain.

But there was a problem. Guarding the branch was a troupe of Howler Monkeys, currently engaged in a high-stakes game of toss with a half-eaten mango.

"Halt!" shrieked Gonzo, the lead monkey, hanging by his tail. "Nobody passes the Howler Gang without paying the fruit toll!"

"We don't have time for this, Gonzo," Maya squawked, fluttering in his face. "This is an emergency of the heart!"

Gonzo blew a raspberry. "No fruit, no flower, bird-brain."

Leo bared his teeth, trying to look intimidating. "Listen here! I am the protector of—"

Before Leo could attempt his fearsome roar (which usually sounded more like a startled hiccup), Paco did the unthinkable. Driven by the sheer power of romance, the sloth detached himself from Leo’s back. Moving with a sudden, uncharacteristic burst of adrenaline, Paco reached into his own fur, pulled out a perfectly ripe, secretly stashed wild fig, and held it out.

The monkeys gasped. A wild fig was the currency of kings.

Gonzo snatched it, tipped an imaginary hat, and swung away, his gang hollering in pursuit of the snack.

"Paco!" Leo cheered. "You saved the mission!"

Paco didn't answer. He was already inching his way across the precarious branch. The wood groaned under his weight, but the sloth’s legendary balance held true. With a tenderness that made even Maya’s cynical heart melt, Paco plucked the glowing pink flower.

Now came the hardest part: the descent.

By the time they found Valentina, the sun was beginning to set, painting the Amazon in strokes of fiery orange and deep purple. She was hanging gracefully from a cecropia tree, slowly blinking at the sunset.

Leo and Maya hid behind a curtain of Spanish moss, holding their breath.

Paco approached, taking a solid five minutes to bridge the gap between their branches. When he was finally face-to-face with her, he extended the Moon-Kissed Bromeliad.

"Valentina," Paco said, his voice smooth and slow. "I crossed the jungle for you. Because... you make my heart race. Look. I am slightly out of breath."

Valentina slowly turned her head. Her large eyes fixed on the beautiful flower, and then on Paco. A slow, gentle smile spread across her face.

She reached out, taking the flower, and then, in a move that shocked the entire canopy, she leaned forward and gave Paco a very slow, very deliberate kiss on the cheek.

"My hero," she whispered.

Behind the moss, Leo wiped a single, dramatic tear from his eye. "Mission accomplished, Maya. True love wins."

Maya chuckled, landing on Leo's shoulder and preening his ear. "Yeah, yeah, it’s a beautiful thing, fuzzball. Now come on. I think I saw a tapir stuck in a mud pit on the way down, and you know how much you love a dramatic rescue."

Leo perked up instantly, the romance of the moment immediately replaced by the call of duty. "To the mud pit!" he roared, missing the branch entirely and tumbling into a soft pile of ferns below, while Paco and Valentina continued their slow-motion date in the beautiful, bustling canopy above.
    "#;
    process_play_and_save_paragraph(
        &omnivoice,
        paragraph,
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

fn convert_safetensors_to_mpk_save(mut omnivoice: OmniVoiceModel) {
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
