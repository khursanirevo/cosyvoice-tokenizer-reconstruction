# CosyVoice Tokenizer Reconstruction

Tools for encoding audio to speech tokens, decoding back to audio, and performing voice conversion using CosyVoice3 components.

## Features

- **Speech Tokenization**: Encode audio to discrete speech tokens using CosyVoice3's tokenizer
  - Regular tokenizer (speech_tokenizer_v3.onnx) - single input processing
  - Batch tokenizer (speech_tokenizer_v3.batch.onnx) - optimized for batch processing
  - Perceiver tokenizer (SpeechTokenExtractor) - same as Flow/LLM modules use internally
- **Speech Decoding**: Decode speech tokens back to audio using Flow model + HiFi-GAN
- **Voice Conversion**: Convert speech from source voice to target voice (bypasses LLM)
- **Intermediate Representations**: Access mel-spectrograms and tokens at each stage

## Installation

```bash
cd tokenizer_reconstruction
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from tokenizer_reconstruction import SpeechReconstructor

# Initialize
reconstructor = SpeechReconstructor(
    model_dir='pretrained_models/Fun-CosyVoice3-0.5B'
)

# Encode and reconstruct audio
audio = reconstructor.reconstruct('input.wav')
reconstructor.save_audio(audio, 'output.wav')

# Calculate quality metrics
metrics = reconstructor.calculate_metrics('input.wav', audio)
print(f"SRER: {metrics['srer']:.2f} dB")
```

### Voice Conversion

```python
from tokenizer_reconstruction import VoiceConverter

converter = VoiceConverter()

# Convert: jenny's content + anwar's voice quality
converter.convert_to_file(
    source_audio_path='jenny.wav',      # What to say
    target_audio_path='anwar.wav',      # Voice to match
    output_path='output.wav'
)
```

**Result**: Source speaker's prosody + target speaker's timbre

## API Reference

### SpeechTokenizer

Encodes audio to speech tokens using CosyVoice3's speech tokenizer.

**Regular Tokenizer (default):**
```python
from tokenizer_reconstruction import SpeechTokenizer

tokenizer = SpeechTokenizer()

# Audio → Speech tokens
tokens = tokenizer.encode('audio.wav')

# Audio → Mel-spectrogram (intermediate)
mel = tokenizer.audio_to_mel('audio.wav')

# Mel-spectrogram → Speech tokens
tokens = tokenizer.mel_to_tokens(mel)
```

**Batch Tokenizer (v3 batch):**
```python
from tokenizer_reconstruction import SpeechTokenizer

# Initialize with batch tokenizer
tokenizer = SpeechTokenizer(use_batch_tokenizer=True)

# Single file with batch tokenizer
tokens = tokenizer.encode_with_batch_tokenizer('audio.wav')

# Batch processing multiple files
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
tokens_list = tokenizer.encode_batch(audio_files)
```

**Perceiver Tokenizer (SpeechTokenExtractor):**
```python
from tokenizer_reconstruction import SpeechTokenizer

# Initialize with perceiver tokenizer (uses SpeechTokenExtractor class)
tokenizer = SpeechTokenizer(use_perceiver=True)

# Single file with perceiver tokenizer
tokens = tokenizer.encode_with_perceiver('audio.wav')

# Batch processing with perceiver tokenizer
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
tokens_list = tokenizer.encode_batch_perceiver(audio_files)
```

**Note:** The Perceiver tokenizer uses `SpeechTokenExtractor`, which is the exact same class used internally by CosyVoice3's Flow and LLM modules. This provides identical tokenization to what's used during model inference.

### SpeechDecoder

Decodes speech tokens to audio using Flow model + HiFi-GAN.

```python
from tokenizer_reconstruction import SpeechDecoder

decoder = SpeechDecoder()

# Speech tokens → Audio
audio = decoder.decode(tokens, prompt_audio='reference.wav')

# Speech tokens → Mel-spectrogram (intermediate)
mel = decoder.tokens_to_mel(tokens, prompt_audio='reference.wav')

# Mel-spectrogram → Audio
audio = decoder.mel_to_audio(mel)
```

### SpeechReconstructor

High-level interface combining tokenizer and decoder.

```python
from tokenizer_reconstruction import SpeechReconstructor

reconstructor = SpeechReconstructor()

# Full pipeline
audio = reconstructor.reconstruct('input.wav')

# Encode/decode separately
tokens = reconstructor.encode('input.wav')
audio = reconstructor.decode(tokens, 'input.wav')

# Calculate metrics
metrics = reconstructor.calculate_metrics('input.wav', audio)
# Returns: {'mse': float, 'mae': float, 'srer': float}
```

### VoiceConverter

Voice conversion using CosyVoice's VC mode (bypasses LLM).

```python
from tokenizer_reconstruction import VoiceConverter

converter = VoiceConverter()

# Convert voice
converter.convert_to_file(
    source_audio_path='content.wav',
    target_audio_path='voice_style.wav',
    output_path='output.wav'
)

# Stream output
for output in converter.convert('content.wav', 'voice_style.wav', stream=True):
    audio = output['tts_speech']
    # Process chunk...
```

## Voice Conversion Details

Voice conversion (VC mode) works by:
1. Extracting speech tokens from source audio (preserves content and prosody)
2. Using target audio for speaker embedding (provides timbre)
3. Bypassing the LLM - direct token-to-audio generation

**Characteristics:**
- ✅ Preserves source speaker's prosody (pitch, rhythm, timing)
- ✅ Changes to target speaker's timbre (voice quality)
- ✅ Fast - no LLM inference needed
- ✅ Content preservation - speech tokens encode what was said

## Architecture

```
Input Audio
    ↓
[Speech Tokenizer] → Mel-spectrogram (16kHz, 128 bands)
    ↓
[Speech Tokenizer] → Speech tokens (discrete)
    ↓
[Flow Model + HiFi-GAN] → Output Audio (24kHz)
```

### Voice Conversion Path

```
Source Audio → Speech tokens (content + prosody)
                      ↓
Target Audio → Speaker embedding → Flow Model → Output
                                      ↓
                                  (timbre transfer)
```

## Quality Metrics

- **MSE** (Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better
- **SRER** (Signal-to-Reconstruction Error Ratio): Higher is better
  - > 20 dB: Excellent
  - 15-20 dB: Good
  - 10-15 dB: Fair
  - < 10 dB: Poor

**Note**: For fair comparison, use mel-spectrogram metrics (16kHz) rather than waveform metrics, as input/output sample rates differ.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- CosyVoice3 model (Fun-CosyVoice3-0.5B or similar)

## License

Same as CosyVoice project.
