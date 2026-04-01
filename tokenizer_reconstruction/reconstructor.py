"""
Speech Reconstructor

High-level interface for encoding and decoding audio using speech tokens.
"""

import torch
import torchaudio
import numpy as np

from .encoder import SpeechTokenizer
from .decoder import SpeechDecoder


class SpeechReconstructor:
    """
    High-level interface for speech tokenization and reconstruction.

    Combines SpeechTokenizer and SpeechDecoder for end-to-end processing.

    Args:
        model_dir: Path to CosyVoice3 model directory
    """

    def __init__(self, model_dir: str = 'pretrained_models/Fun-CosyVoice3-0.5B'):
        self.tokenizer = SpeechTokenizer(model_dir=model_dir)
        self.decoder = SpeechDecoder(model_dir=model_dir)
        self.sample_rate = self.decoder.sample_rate

    def encode(self, audio_path: str) -> np.ndarray:
        """Encode audio to speech tokens."""
        return self.tokenizer.encode(audio_path)

    def decode(self, tokens: np.ndarray, prompt_audio_path: str) -> torch.Tensor:
        """Decode speech tokens to audio."""
        tokens_tensor = torch.from_numpy(tokens).unsqueeze(0).long()
        return self.decoder.decode(tokens_tensor, prompt_audio_path)

    def reconstruct(self, audio_path: str) -> torch.Tensor:
        """
        Encode and reconstruct audio.

        Args:
            audio_path: Path to audio file

        Returns:
            Reconstructed audio tensor [1, samples]
        """
        # Encode
        tokens = self.encode(audio_path)

        # Decode (use same audio as prompt)
        reconstructed = self.decode(tokens, audio_path)

        return reconstructed

    def save_tokens(self, tokens: np.ndarray, output_path: str):
        """Save tokens to file."""
        self.tokenizer.save_tokens(tokens, output_path)

    def load_tokens(self, tokens_path: str) -> np.ndarray:
        """Load tokens from file."""
        return self.tokenizer.load_tokens(tokens_path)

    def save_audio(self, audio: torch.Tensor, output_path: str):
        """Save audio to file."""
        self.decoder.save_audio(audio, output_path)

    def calculate_metrics(self, original_path: str, reconstructed: torch.Tensor) -> dict:
        """
        Calculate reconstruction quality metrics.

        Args:
            original_path: Path to original audio file
            reconstructed: Reconstructed audio tensor [1, samples]

        Returns:
            Dictionary with MSE, MAE, SRER metrics
        """
        # Load original
        original, sr = torchaudio.load(original_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            original = resampler(original)

        # Match lengths
        min_len = min(original.shape[1], reconstructed.shape[1])
        orig = original[:, :min_len].cpu().numpy()
        gen = reconstructed[:, :min_len].cpu().numpy()

        # Calculate metrics
        mse = np.mean((orig - gen) ** 2)
        mae = np.mean(np.abs(orig - gen))
        signal_power = np.mean(orig ** 2)
        srer = 10 * np.log10(signal_power / (mse + 1e-10))

        return {
            'mse': float(mse),
            'mae': float(mae),
            'srer': float(srer)
        }
