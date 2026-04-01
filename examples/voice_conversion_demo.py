#!/usr/bin/env python3
"""
Example: Voice Conversion (No LLM)

Demonstrates voice conversion - converting speech from one voice to another
without using the LLM, similar to the chatterbox example.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tokenizer_reconstruction import VoiceConverter


def main():
    print("=" * 80)
    print("Voice Conversion Demo - No LLM (Like Chatterbox)")
    print("=" * 80)

    # Paths
    source_audio = project_root / 'jenny.wav'  # What to say
    target_audio = project_root / 'jenny.wav'   # Voice to match
    output_file = 'voice_conversion_output.wav'

    # Initialize
    print("\n[1/3] Initializing VoiceConverter...")
    converter = VoiceConverter(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

    # Convert voice
    print("\n[2/3] Converting voice...")
    print(f"  Source audio: {source_audio}")
    print(f"  Target voice: {target_audio}")

    audio = converter.convert_to_file(
        source_audio_path=str(source_audio),
        target_audio_path=str(target_audio),
        output_path=output_file
    )

    # Calculate metrics
    print("\n[3/3] Calculating metrics...")
    metrics = converter.calculate_metrics(str(source_audio), audio)

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"MSE:  {metrics['mse']:.8f}")
    print(f"MAE:  {metrics['mae']:.8f}")
    print(f"SRER: {metrics['srer']:.2f} dB")
    print(f"{'=' * 80}")

    if metrics['srer'] > 20:
        print("✓ Excellent quality (> 20 dB)")
    elif metrics['srer'] > 15:
        print("✓ Good quality (15-20 dB)")
    elif metrics['srer'] > 10:
        print("○ Fair quality (10-15 dB)")
    elif metrics['srer'] > 0:
        print("✗ Poor quality (0-10 dB)")
    else:
        print(f"✗ Very Poor quality ({metrics['srer']:.2f} dB)")

    print(f"\n✓ Output: {output_file}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
