#!/usr/bin/env python3
"""
Perceiver Tokenizer Demo: Using SpeechTokenExtractor

Demonstrates the Perceiver tokenizer (SpeechTokenExtractor) which is
the exact same implementation used internally by CosyVoice3's Flow and LLM modules.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tokenizer_reconstruction import SpeechTokenizer


def demo_perceiver_single():
    """Demo: Perceiver tokenizer for single file."""
    print("\n" + "=" * 80)
    print("DEMO 1: Perceiver Tokenizer (Single File)")
    print("=" * 80)

    # Initialize with perceiver tokenizer
    print("\n[1/2] Initializing SpeechTokenizer with perceiver tokenizer...")
    tokenizer = SpeechTokenizer(use_perceiver=True)

    # Encode single file
    print("\n[2/2] Encoding audio with perceiver tokenizer...")
    tokens = tokenizer.encode_with_perceiver('jenny.wav')

    print(f"\nResults:")
    print(f"  Tokens: {len(tokens)}")
    print(f"  First 10 tokens: {tokens[:10]}")
    print(f"  Token dtype: {tokens.dtype}")
    print("\n✓ Demo 1 complete")
    print("Note: This is identical to Flow/LLM internal tokenization")


def demo_perceiver_batch():
    """Demo: Perceiver tokenizer for multiple files."""
    print("\n" + "=" * 80)
    print("DEMO 2: Perceiver Tokenizer (Multiple Files)")
    print("=" * 80)

    # Initialize with perceiver tokenizer
    print("\n[1/2] Initializing SpeechTokenizer with perceiver tokenizer...")
    tokenizer = SpeechTokenizer(use_perceiver=True)

    # Encode multiple files
    print("\n[2/2] Encoding multiple audio files...")
    audio_files = ['jenny.wav', 'anwar.wav']

    tokens_list = tokenizer.encode_batch_perceiver(audio_files)

    print(f"\nResults:")
    for i, (audio_path, tokens) in enumerate(zip(audio_files, tokens_list)):
        print(f"  {audio_path}: {len(tokens)} tokens")

    print("\n✓ Demo 2 complete")


def demo_comparison_all():
    """Demo: Compare all three tokenizers."""
    print("\n" + "=" * 80)
    print("DEMO 3: Comparison - Regular vs Batch vs Perceiver")
    print("=" * 80)

    audio_path = 'jenny.wav'

    # Regular tokenizer
    print("\n[1/3] Using regular tokenizer...")
    tokenizer_regular = SpeechTokenizer(use_batch_tokenizer=False, use_perceiver=False)
    tokens_regular = tokenizer_regular.encode(audio_path)

    # Batch tokenizer
    print("\n[2/3] Using batch tokenizer...")
    tokenizer_batch = SpeechTokenizer(use_batch_tokenizer=True, use_perceiver=False)
    tokens_batch = tokenizer_batch.encode_with_batch_tokenizer(audio_path)

    # Perceiver tokenizer
    print("\n[3/3] Using perceiver tokenizer...")
    tokenizer_perceiver = SpeechTokenizer(use_perceiver=True)
    tokens_perceiver = tokenizer_perceiver.encode_with_perceiver(audio_path)

    # Compare
    import numpy as np

    print(f"\nComparison:")
    print(f"  Regular tokenizer:  {len(tokens_regular)} tokens")
    print(f"  Batch tokenizer:    {len(tokens_batch)} tokens")
    print(f"  Perceiver tokenizer: {len(tokens_perceiver)} tokens")

    # Check if batch and perceiver produce same results
    if len(tokens_batch) == len(tokens_perceiver):
        matches = np.sum(tokens_batch == tokens_perceiver)
        print(f"\n  Batch vs Perceiver match rate: {matches / len(tokens_batch) * 100:.1f}%")
        print(f"  (Should be 100% - same underlying model)")

    print("\n✓ Demo 3 complete")
    print("Note: Perceiver tokenizer is identical to Flow/LLM internal implementation")


def main():
    """Run all demos."""
    print("=" * 80)
    print("CosyVoice Tokenizer Reconstruction - Perceiver Tokenizer Demo")
    print("=" * 80)

    demos = [
        ("Perceiver Tokenizer (Single)", demo_perceiver_single),
        ("Perceiver Tokenizer (Batch)", demo_perceiver_batch),
        ("All Tokenizers Comparison", demo_comparison_all),
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
    print("\nKey Points:")
    print("  1. Perceiver tokenizer uses SpeechTokenExtractor class")
    print("  2. This is identical to Flow/LLM internal tokenization")
    print("  3. Use use_perceiver=True for exact model alignment")
    print("=" * 80)


if __name__ == '__main__':
    main()
