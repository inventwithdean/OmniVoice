# OmniVoice (Burn / Rust)

A lightweight, zero-dependency, cross-platform Text-to-Speech (TTS) and voice cloning inference engine implemented in Rust using the **Burn** deep learning framework.

OmniVoice runs directly on your GPU or CPU using the **wgpu** backend—delivering fast, high-fidelity voice cloning without requiring Python, CUDA toolkits, or PyTorch runtimes.

---

## ✨ Features

* **Ultralight Distribution:** Standalone executable is only **74.4 MB** (~**21.9 MB** zipped for Windows).
* **Runs Anywhere (`wgpu`):** Powered by Burn's `wgpu` backend, supporting Vulkan, DirectX 12, Metal, and native CPU fallbacks out of the box.
* **Automatic Model Downloads:** Model weights (`omnivoice.mpk`) and tokenizer (`tokenizer.json`) are pulled directly and cached from [Hugging Face Hub](https://huggingface.co/inventwithdean/OmniVoice) on first launch.
* **Zero-Shot Voice Cloning:** Clone any voice using a short reference audio file (`.wav`) and its transcript.
* **Streaming Generation & Playback:** Sentences are synthesized and streamed directly to your audio device via `rodio` while the remaining sentences process in parallel in the background.
* **Full Paragraph Support:** Automatically chunks long texts by punctuation, resamples reference audio (24 kHz mono), and stitches everything into a clean final `paragraph.wav`.

---

## 🚀 Quick Start (Pre-built Windows Binary)

1. Download the latest release `.zip` (**~21.9 MB**) from the [**Releases**](https://github.com/inventwithdean/omnivoice/releases/latest) page.
2. Extract the archive into a folder.
3. Place your reference audio (`ref.wav`), reference text, and target speech in the same directory:
* Configure `args.json`
* Write your target paragraph in `data.txt`


4. Run the executable:
```cmd
omnivoice.exe
```


*On the first run, the app will automatically download the quantized model weights (`omnivoice.mpk`) and tokenizer from HuggingFace.*

---

## ⚙️ Configuration

### 1. `args.json`

Define your reference voice sample, matching transcript, and optional language code:

```json
{
  "ref_audio_file": "samples/reference.wav",
  "ref_text": "This is the exact spoken text inside the reference audio file.",
  "lang": "en"
}

```

* `ref_audio_file`: Path to any standard `.wav` file (automatically converted and resampled to 24 kHz mono).
* `ref_text`: Ground-truth transcript corresponding to `ref_audio_file`.
* `lang`: Target language tag (e.g., `"en"`).

### 2. `data.txt`

Add the text or paragraphs you want synthesized into `data.txt`:

```text
OmniVoice is running entirely in Rust without Python or Torch. 
It streams audio through your default speakers as it computes, then exports the full output to paragraph.wav.
```

---

## 🛠️ Building from Source

### Prerequisites

* [Rust & Cargo](https://rustup.rs/) (latest stable toolchain)
* A GPU driver supporting Vulkan, DirectX 12, or Metal

### Steps

1. **Clone the repository:**
```bash
git clone https://github.com/inventwithdean/omnivoice
cd omnivoice
```


2. **Prepare configuration files:**
Ensure `args.json`, `data.txt`, and your reference `.wav` are in your working directory.
3. **Run in release mode:**
```bash
cargo run --release
```



---

## 🧠 Architecture Overview

OmniVoice combines transformer-based language modeling with discrete neural audio codecs:

```text
[Reference WAV] ───> [HiggsAudioV2 Tokenizer] ───> [Reference Audio Codes (8 Codebooks)]
                                                                  │
[Text Prompt]   ───> [Qwen3 LLM Backbone]     ───> [Iterative Masked Generation]
                                                                  │
                                                [Higgs Audio Decoder (24 kHz)]
                                                                  │
                                                        ┌─────────┴─────────┐
                                                        ▼                   ▼
                                                [Real-time Stream]   [paragraph.wav]

```

* **Backbone:** Custom Qwen3 architecture driving conditional acoustic token prediction.
* **Audio Codec:** HiggsAudio V2 multi-codebook RVQ encoder/decoder running at 24,000 Hz.
* **Sampling:** Iterative mask-predict scheduler with Classifier-Free Guidance (CFG: `3.0`) and Gumbel position perturbation.
* **Quantization:** Quantized weights packaged in Burn's native `.mpk` format for minimal memory overhead and rapid cold starts.

---

## 📦 Model Hub

All model weights and tokenizer configurations are hosted on Hugging Face:

* **Repo:** [inventwithdean/OmniVoice](https://huggingface.co/inventwithdean/OmniVoice)


---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more details.