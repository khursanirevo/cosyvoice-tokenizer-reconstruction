#!/usr/bin/env python3
"""
Quick Start Example: CosyVoice Tokenizer Reconstruction

Basic usage example for getting started quickly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tokenizer_reconstruction import SpeechReconstructor


def main():
    print("=" * 80)
    print("Quick Start: Speech Tokenization & Reconstruction")
    print("=" * 80)

    # Initialize
    print("\n[1/2] Initializing SpeechReconstructor...")
    reconstructor = SpeechReconstructor(
        model_dir='pretrained_models/Fun-CosyVoice3-0.5B'
    )

    # Reconstruct audio
    print("\n[2/2] Encoding and reconstructing audio...")
    print("  Input: jenny.wav")

    audio = reconstructor.reconstruct('jenny.wav')
    reconstructor.save_audio(audio, 'quick_start_output.wav')

    # Calculate metrics
    metrics = reconstructor.calculate_metrics('jenny.wav', audio)

    print(f"\nResults:")
    print(f"  Output: quick_start_output.wav")
    print(f"  MSE:    {metrics['mse']:.6f}")
    print(f"  MAE:    {metrics['mae']:.6f}")
    print(f"  SRER:   {metrics['srer']:.2f} dB")
    print("=" * 80)


if __name__ == '__main__':
    main()
