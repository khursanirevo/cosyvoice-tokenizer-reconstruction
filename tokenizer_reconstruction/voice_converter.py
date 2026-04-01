"""
Voice Converter

Converts speech from one voice to another (voice conversion / voice cloning).
Uses CosyVoice's inference_vc method which bypasses the LLM.
"""

import types
import sys
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Fix pkg_resources BEFORE any CosyVoice imports
pkg_resources = types.ModuleType('pkg_resources')
pkg_resources.declare_namespace = lambda x: None


class FakeDistribution:
    def __init__(self, name):
        self.name = name
        self.version = '0.0.0'

    @property
    def location(self):
        return ''


pkg_resources.get_distribution = lambda name: FakeDistribution(name)
pkg_resources.iter_entry_points = lambda group=None: []
pkg_resources.working_set = []
pkg_resources.require = lambda *args, **kwargs: None
sys.modules['pkg_resources'] = pkg_resources


class VoiceConverter:
    """
    Voice Converter using CosyVoice's VC (Voice Conversion) mode.

    Converts speech from source voice to target voice without using LLM.
    This directly uses speech tokens from both audios.

    Args:
        model_dir: Path to CosyVoice3 model directory
    """

    def __init__(self, model_dir: str = 'pretrained_models/Fun-CosyVoice3-0.5B'):
        # Add Matcha-TTS to path
        project_root = Path(__file__).parent.parent.parent
        sys.path.append(str(project_root / 'third_party' / 'Matcha-TTS'))

        # Import CosyVoice
        from cosyvoice.cli.cosyvoice import CosyVoice3

        # Load model
        self.model = CosyVoice3(model_dir=model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = self.model.sample_rate

        print(f"[VoiceConverter] Initialized on {self.device}")
        print(f"[VoiceConverter] Model sample rate: {self.sample_rate} Hz")
        print("[VoiceConverter] Mode: Voice Conversion (no LLM)")

    def convert(self, source_audio_path: str, target_audio_path: str, stream=False):
        """
        Convert source speech to sound like target voice.

        Args:
            source_audio_path: Path to source audio (what to say)
            target_audio_path: Path to target audio (voice to match)
            stream: Whether to stream output

        Yields:
            Dict with 'tts_speech' key containing converted audio tensor
        """
        print("[VoiceConverter] Converting voice...")
        print(f"  Source: {source_audio_path}")
        print(f"  Target: {target_audio_path}")

        # Run voice conversion
        for i, output in enumerate(self.model.inference_vc(
            source_wav=source_audio_path,
            prompt_wav=target_audio_path,
            stream=stream
        )):
            audio = output['tts_speech']
            duration = audio.shape[1] / self.sample_rate
            print(f"  Generated chunk {i}: {duration:.2f}s @ {self.sample_rate}Hz")
            yield output

    def convert_to_file(self, source_audio_path: str, target_audio_path: str,
                       output_path: str):
        """
        Convert source speech to sound like target voice and save to file.

        Args:
            source_audio_path: Path to source audio (what to say)
            target_audio_path: Path to target audio (voice to match)
            output_path: Path to output audio file

        Returns:
            Audio tensor [1, samples]
        """
        # Get first chunk (non-streaming)
        for output in self.convert(source_audio_path, target_audio_path, stream=False):
            audio = output['tts_speech']
            break

        # Save to file
        self.save_audio(audio, output_path)

        return audio

    def save_audio(self, audio: torch.Tensor, output_path: str):
        """Save audio to file."""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torchaudio.save(output_path, audio.cpu(), self.sample_rate)
        print(f"[VoiceConverter] Saved audio to {output_path}")

    def calculate_metrics(self, original_path: str, converted: torch.Tensor) -> dict:
        """
        Calculate voice conversion quality metrics.

        Args:
            original_path: Path to original source audio file
            converted: Converted audio tensor [1, samples]

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
        min_len = min(original.shape[1], converted.shape[1])
        orig = original[:, :min_len].cpu().numpy()
        conv = converted[:, :min_len].cpu().numpy()

        # Calculate metrics
        mse = np.mean((orig - conv) ** 2)
        mae = np.mean(np.abs(orig - conv))
        signal_power = np.mean(orig ** 2)
        srer = 10 * np.log10(signal_power / (mse + 1e-10))

        return {
            'mse': float(mse),
            'mae': float(mae),
            'srer': float(srer)
        }
