"""
Speech Tokenizer Decoder

Decodes speech tokens to audio using CosyVoice3's Flow model and HiFi-GAN vocoder.
"""

import types
import sys
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


class SpeechDecoder:
    """
    Decodes speech tokens to audio using CosyVoice3's Flow model and HiFi-GAN.

    Args:
        model_dir: Path to CosyVoice3 model directory
    """

    def __init__(self, model_dir: str = 'pretrained_models/Fun-CosyVoice3-0.5B'):
        # Add Matcha-TTS to path
        project_root = Path(__file__).parent.parent.parent
        sys.path.append(str(project_root / 'third_party' / 'Matcha-TTS'))

        # Import CosyVoice
        from cosyvoice.cli.cosyvoice import CosyVoice3
        from cosyvoice.utils.file_utils import load_wav
        import torchaudio.compliance.kaldi as kaldi

        self.load_wav = load_wav
        self.kaldi = kaldi

        # Load model
        self.model = CosyVoice3(model_dir=model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = self.model.sample_rate

        print(f"[SpeechDecoder] Initialized on {self.device}")
        print(f"[SpeechDecoder] Model sample rate: {self.sample_rate} Hz")

    def _extract_speaker_embedding(self, audio_path: str) -> torch.Tensor:
        """Extract speaker embedding from audio."""
        audio_16k = self.load_wav(audio_path, 16000)
        feat = self.kaldi.fbank(audio_16k,
                                 num_mel_bins=80,
                                 dither=0,
                                 sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding_np = self.model.frontend.campplus_session.run(None, {
            self.model.frontend.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()
        })[0].flatten().tolist()
        embedding = torch.tensor([embedding_np]).to(self.device)
        return embedding

    def _extract_speech_features(self, audio_path: str) -> tuple:
        """Extract speech features from audio."""
        audio_24k = self.load_wav(audio_path, 24000)
        speech_feat = self.model.frontend.feat_extractor(audio_24k)
        speech_feat = speech_feat.squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def tokens_to_mel(self, tokens: torch.Tensor, prompt_audio_path: str) -> torch.Tensor:
        """
        Convert speech tokens to mel-spectrogram using Flow model.

        Args:
            tokens: Speech tokens tensor [1, num_tokens]
            prompt_audio_path: Path to reference audio for conditioning

        Returns:
            Mel-spectrogram tensor [1, 80, time]
        """
        # Extract conditioning from prompt audio
        embedding = self._extract_speaker_embedding(prompt_audio_path)
        speech_feat, speech_feat_len = self._extract_speech_features(prompt_audio_path)

        # Prepare inputs
        token = tokens.to(self.device)
        token_len = torch.tensor([tokens.shape[1]], dtype=torch.int32).to(self.device)
        prompt_token = tokens.to(self.device)
        prompt_token_len = torch.tensor([tokens.shape[1]], dtype=torch.int32).to(self.device)

        # Run Flow model
        with torch.no_grad():
            tts_mel, _ = self.model.model.flow.inference(
                token=token,
                token_len=token_len,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=speech_feat,
                prompt_feat_len=speech_feat_len,
                embedding=embedding,
                streaming=False,
                finalize=True
            )

        print(f"[SpeechDecoder] Generated mel: {tts_mel.shape} (freq: {tts_mel.shape[1]}, time: {tts_mel.shape[2]})")
        return tts_mel

    def mel_to_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to audio using HiFi-GAN.

        Args:
            mel: Mel-spectrogram tensor [1, 80, time]

        Returns:
            Audio tensor [1, samples]
        """
        with torch.no_grad():
            tts_speech, _ = self.model.model.hift.inference(
                speech_feat=mel,
                finalize=True
            )

        duration = tts_speech.shape[1] / self.sample_rate
        print(f"[SpeechDecoder] Generated audio: {duration:.2f}s @ {self.sample_rate}Hz")

        return tts_speech

    def decode(self, tokens: torch.Tensor, prompt_audio_path: str) -> torch.Tensor:
        """
        Decode speech tokens to audio.

        Args:
            tokens: Speech tokens tensor [1, num_tokens]
            prompt_audio_path: Path to reference audio for conditioning

        Returns:
            Audio tensor [1, samples]
        """
        print("[SpeechDecoder] Decoding tokens...")

        # Tokens → Mel
        mel = self.tokens_to_mel(tokens, prompt_audio_path)

        # Mel → Audio
        audio = self.mel_to_audio(mel)

        return audio

    def save_audio(self, audio: torch.Tensor, output_path: str):
        """Save audio to file."""
        torchaudio.save(output_path, audio.cpu(), self.sample_rate)
        print(f"[SpeechDecoder] Saved audio to {output_path}")

    def save_mel(self, mel: torch.Tensor, output_path: str):
        """Save mel-spectrogram to file."""
        np.save(output_path, mel.cpu().numpy())
        print(f"[SpeechDecoder] Saved mel to {output_path}")
