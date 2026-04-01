#!/usr/bin/env python3
"""
Example: All Pipeline Stages

Demonstrates all available stages in the tokenizer_reconstruction package.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tokenizer_reconstruction import SpeechTokenizer, SpeechDecoder
import torch


def main():
    print("=" * 80)
    print("All Pipeline Stages Demo")
    print("=" * 80)

    audio_path = project_root / 'jenny.wav'
    model_dir = project_root / 'pretrained_models/Fun-CosyVoice3-0.5B'

    # Initialize
    tokenizer = SpeechTokenizer(model_dir=str(model_dir))
    decoder = SpeechDecoder(model_dir=str(model_dir))

    # Stage 1: Audio → Mel-Spectrogram (EMBEDDING)
    print("\n[1/4] Audio → Mel-Spectrogram (EMBEDDING)")
    mel_input = tokenizer.audio_to_mel(str(audio_path))
    print(f"  Input mel: {mel_input.shape}")

    # Stage 2: Mel → Tokens
    print("\n[2/4] Mel → Tokens")
    tokens = tokenizer.mel_to_tokens(mel_input)
    print(f"  Tokens: {len(tokens)}")

    # Stage 3: Tokens → Mel (Flow)
    print("\n[3/4] Tokens → Mel (Flow Model)")
    tokens_tensor = torch.from_numpy(tokens).unsqueeze(0).long()
    mel_output = decoder.tokens_to_mel(tokens_tensor, str(audio_path))
    print(f"  Output mel: {mel_output.shape}")

    # Stage 4: Mel → Audio (Vocoder)
    print("\n[4/4] Mel → Audio (Vocoder)")
    audio = decoder.mel_to_audio(mel_output)
    print(f"  Audio: {audio.shape}")

    print("\n✓ All stages demonstrated!")


if __name__ == '__main__':
    main()
