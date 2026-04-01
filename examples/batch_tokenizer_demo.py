#!/usr/bin/env python3
"""
Batch Tokenizer Demo: Using speech_tokenizer_v3.batch.onnx

Demonstrates the batch tokenizer which is used internally by
CosyVoice3's Flow and LLM modules.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tokenizer_reconstruction import SpeechTokenizer


def demo_batch_tokenizer_single():
    """Demo: Batch tokenizer for single file."""
    print("\n" + "=" * 80)
    print("DEMO 1: Batch Tokenizer (Single File)")
    print("=" * 80)

    # Initialize with batch tokenizer
    print("\n[1/2] Initializing SpeechTokenizer with batch tokenizer...")
    tokenizer = SpeechTokenizer(use_batch_tokenizer=True)

    # Encode single file
    print("\n[2/2] Encoding audio with batch tokenizer...")
    tokens = tokenizer.encode_with_batch_tokenizer('jenny.wav')

    print(f"\nResults:")
    print(f"  Tokens: {len(tokens)}")
    print(f"  First 10 tokens: {tokens[:10]}")
    print("\n✓ Demo 1 complete")


def demo_batch_tokenizer_multi():
    """Demo: Batch tokenizer for multiple files."""
    print("\n" + "=" * 80)
    print("DEMO 2: Batch Tokenizer (Multiple Files)")
    print("=" * 80)

    # Initialize with batch tokenizer
    print("\n[1/2] Initializing SpeechTokenizer with batch tokenizer...")
    tokenizer = SpeechTokenizer(use_batch_tokenizer=True)

    # Encode multiple files
    print("\n[2/2] Encoding multiple audio files...")
    audio_files = ['jenny.wav', 'anwar.wav']

    tokens_list = tokenizer.encode_batch(audio_files)

    print(f"\nResults:")
    for i, (audio_path, tokens) in enumerate(zip(audio_files, tokens_list)):
        print(f"  {audio_path}: {len(tokens)} tokens")

    print("\n✓ Demo 2 complete")


def demo_comparison():
    """Demo: Compare regular vs batch tokenizer."""
    print("\n" + "=" * 80)
    print("DEMO 3: Regular vs Batch Tokenizer Comparison")
    print("=" * 80)

    audio_path = 'jenny.wav'

    # Regular tokenizer
    print("\n[1/2] Using regular tokenizer...")
    tokenizer_regular = SpeechTokenizer(use_batch_tokenizer=False)
    tokens_regular = tokenizer_regular.encode(audio_path)

    # Batch tokenizer
    print("\n[2/2] Using batch tokenizer...")
    tokenizer_batch = SpeechTokenizer(use_batch_tokenizer=True)
    tokens_batch = tokenizer_batch.encode_with_batch_tokenizer(audio_path)

    # Compare
    print(f"\nComparison:")
    print(f"  Regular tokenizer: {len(tokens_regular)} tokens")
    print(f"  Batch tokenizer:   {len(tokens_batch)} tokens")
    print(f"  Difference:        {abs(len(tokens_regular) - len(tokens_batch))} tokens")

    # Check if tokens are similar
    if len(tokens_regular) == len(tokens_batch):
        matches = np.sum(tokens_regular == tokens_batch)
        print(f"  Token match rate: {matches / len(tokens_regular) * 100:.1f}%")

    print("\n✓ Demo 3 complete")


def main():
    """Run all demos."""
    import numpy as np

    print("=" * 80)
    print("CosyVoice Tokenizer Reconstruction - Batch Tokenizer Demo")
    print("=" * 80)

    demos = [
        ("Batch Tokenizer (Single)", demo_batch_tokenizer_single),
        ("Batch Tokenizer (Multiple)", demo_batch_tokenizer_multi),
        ("Regular vs Batch Comparison", demo_comparison),
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
    print("\nNote: speech_tokenizer_v3.batch.onnx is the same tokenizer used")
    print("      internally by CosyVoice3's Flow and LLM modules.")
    print("=" * 80)


if __name__ == '__main__':
    main()
