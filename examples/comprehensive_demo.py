#!/usr/bin/env python3
"""
Comprehensive Demo: CosyVoice Tokenizer Reconstruction

Demonstrates all features:
1. Speech tokenization (encode/decode)
2. Voice conversion (VC mode)
3. Quality metrics
4. Intermediate representations (mel-spectrograms)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tokenizer_reconstruction import (
    SpeechTokenizer,
    SpeechDecoder,
    SpeechReconstructor,
    VoiceConverter
)


def demo_tokenization():
    """Demo: Encode and decode audio using speech tokens."""
    print("\n" + "=" * 80)
    print("DEMO 1: Speech Tokenization")
    print("=" * 80)

    tokenizer = SpeechTokenizer()
    decoder = SpeechDecoder()

    # Encode
    print("\n[1/3] Encoding audio to speech tokens...")
    tokens = tokenizer.encode('jenny.wav')
    print(f"  Generated {len(tokens)} tokens")

    # Decode
    print("\n[2/3] Decoding tokens to audio...")
    tokens_tensor = torch.from_numpy(tokens).unsqueeze(0).long()
    audio = decoder.decode(tokens_tensor, 'jenny.wav')

    # Save
    print("\n[3/3] Saving reconstructed audio...")
    decoder.save_audio(audio, 'demo_reconstructed.wav')

    # Metrics
    reconstructor = SpeechReconstructor()
    metrics = reconstructor.calculate_metrics('jenny.wav', audio)
    print(f"\nQuality Metrics:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  SRER: {metrics['srer']:.2f} dB")

    print("\n✓ Demo 1 complete: demo_reconstructed.wav")


def demo_voice_conversion():
    """Demo: Voice conversion (VC mode - preserves source prosody)."""
    print("\n" + "=" * 80)
    print("DEMO 2: Voice Conversion (VC Mode)")
    print("=" * 80)

    converter = VoiceConverter()

    # Convert
    print("\n[1/2] Converting voice...")
    print("  Source: jenny.wav (content)")
    print("  Target: anwar.wav (voice style)")
    audio = converter.convert_to_file(
        source_audio_path='jenny.wav',
        target_audio_path='anwar.wav',
        output_path='demo_vc_output.wav'
    )

    # Metrics
    print("\n[2/2] Calculating metrics...")
    metrics = converter.calculate_metrics('jenny.wav', audio)
    print(f"\nQuality Metrics:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  SRER: {metrics['srer']:.2f} dB")

    print("\nNote: VC mode preserves source prosody, changes timbre only")
    print("✓ Demo 2 complete: demo_vc_output.wav")


def demo_intermediate_representations():
    """Demo: Access intermediate representations (mel-spectrograms)."""
    print("\n" + "=" * 80)
    print("DEMO 3: Intermediate Representations")
    print("=" * 80)

    tokenizer = SpeechTokenizer()

    # Extract mel-spectrogram
    print("\n[1/2] Extracting mel-spectrogram...")
    mel = tokenizer.audio_to_mel('jenny.wav')
    print(f"  Mel shape: {mel.shape} (bands, time)")

    # Save mel-spectrogram
    print("\n[2/2] Saving intermediate representation...")
    tokenizer.save_mel(mel, 'demo_mel_spectrogram.npy')
    print("  Saved: demo_mel_spectrogram.npy")

    # Load and verify
    mel_loaded = tokenizer.load_mel('demo_mel_spectrogram.npy')
    print(f"  Loaded mel shape: {mel_loaded.shape}")

    print("\n✓ Demo 3 complete: Mel-spectrogram saved")


def main():
    """Run all demos."""
    print("=" * 80)
    print("CosyVoice Tokenizer Reconstruction - Comprehensive Demo")
    print("=" * 80)

    import torch  # Import here to avoid issues

    demos = [
        ("Speech Tokenization", demo_tokenization),
        ("Voice Conversion", demo_voice_conversion),
        ("Intermediate Representations", demo_intermediate_representations),
    ]

    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")

    print("\nRunning all demos...")

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All demos complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - demo_reconstructed.wav        (tokenization demo)")
    print("  - demo_vc_output.wav            (voice conversion demo)")
    print("  - demo_mel_spectrogram.npy      (intermediate representation)")
    print("=" * 80)


if __name__ == '__main__':
    main()
