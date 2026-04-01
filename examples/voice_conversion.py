#!/usr/bin/env python3
"""
Voice Conversion: jenny.wav content + anwar.wav voice

Uses VoiceConverter to convert speech content from jenny.wav
to sound like it's spoken by the voice from anwar.wav.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tokenizer_reconstruction import VoiceConverter


def main():
    print("=" * 80)
    print("Voice Conversion: jenny.wav content + anwar.wav voice")
    print("=" * 80)

    source_audio = 'jenny.wav'   # What to say (content)
    target_audio = 'anwar.wav'  # Voice style to match
    output_file = 'jenny_content_anwar_voice.wav'

    print(f"\nContent source: {source_audio}")
    print(f"Voice target:  {target_audio}")
    print(f"Output file:   {output_file}")

    # Initialize converter
    print(f"\n[1/2] Initializing VoiceConverter...")
    converter = VoiceConverter(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

    # Convert
    print(f"\n[2/2] Converting voice...")
    audio = converter.convert_to_file(
        source_audio_path=source_audio,
        target_audio_path=target_audio,
        output_path=output_file
    )

    # Calculate metrics
    print(f"\nCalculating metrics...")
    metrics = converter.calculate_metrics(source_audio, audio)

    print(f"\n{'=' * 80}")
    print(f"RESULTS")
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
    print(f"  → Content from: {source_audio}")
    print(f"  → Voice style from: {target_audio}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
