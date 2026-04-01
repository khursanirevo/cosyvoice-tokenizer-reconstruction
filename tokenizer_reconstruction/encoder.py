"""
Speech Tokenizer Encoder

Encodes audio to speech tokens using CosyVoice3's ONNX tokenizer.

Two tokenizer modes available:
1. Regular tokenizer (speech_tokenizer_v3.onnx) - default, single input processing
2. Batch tokenizer (speech_tokenizer_v3.batch.onnx) - optimized for batch processing
"""

import types
import sys
import torch
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


class SpeechTokenizer:
    """
    Encodes audio to speech tokens using CosyVoice3's speech tokenizer.

    Args:
        model_dir: Path to CosyVoice3 model directory
        use_batch_tokenizer: If True, use batch tokenizer (speech_tokenizer_v3.batch.onnx).
                           If False, use regular tokenizer (speech_tokenizer_v3.onnx).
    """

    def __init__(self, model_dir: str = 'pretrained_models/Fun-CosyVoice3-0.5B',
                 use_batch_tokenizer: bool = False):
        # Add Matcha-TTS to path
        project_root = Path(__file__).parent.parent.parent
        sys.path.append(str(project_root / 'third_party' / 'Matcha-TTS'))

        # Import CosyVoice
        from cosyvoice.cli.cosyvoice import CosyVoice3
        from cosyvoice.utils.file_utils import load_wav
        import whisper
        import onnxruntime

        self.load_wav = load_wav
        self.whisper = whisper
        self.use_batch_tokenizer = use_batch_tokenizer
        self.model_dir = model_dir

        # Load model
        self.model = CosyVoice3(model_dir=model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize batch tokenizer if requested
        if use_batch_tokenizer:
            batch_tokenizer_path = f"{model_dir}/speech_tokenizer_v3.batch.onnx"
            option = onnxruntime.SessionOptions()
            option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            option.intra_op_num_threads = 1
            self.batch_tokenizer_session = onnxruntime.InferenceSession(
                batch_tokenizer_path,
                sess_options=option,
                providers=["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"]
            )
            print(f"[SpeechTokenizer] Initialized batch tokenizer: {batch_tokenizer_path}")

        print(f"[SpeechTokenizer] Initialized on {self.device}")
        print(f"[SpeechTokenizer] Model sample rate: {self.model.sample_rate} Hz")
        print(f"[SpeechTokenizer] Using batch tokenizer: {use_batch_tokenizer}")

    def audio_to_mel(self, audio_path: str) -> torch.Tensor:
        """
        Convert audio to mel-spectrogram embedding.

        This is the intermediate representation before tokenization.
        Uses Whisper-style log mel-spectrogram at 16kHz with 128 bands.

        Args:
            audio_path: Path to audio file

        Returns:
            Mel-spectrogram tensor (shape: [1, 128, time])
        """
        # Load and preprocess audio
        audio_16k = self.load_wav(audio_path, 16000)

        # Extract mel spectrogram
        mel = self.whisper.log_mel_spectrogram(audio_16k, n_mels=128)

        print(f"[SpeechTokenizer] Extracted mel-spectrogram from {audio_path}")
        print(f"[SpeechTokenizer]   Mel shape: {mel.shape} (bands=128, time={mel.shape[2]})")

        return mel

    def mel_to_tokens(self, mel: torch.Tensor) -> np.ndarray:
        """
        Convert mel-spectrogram to speech tokens.

        Args:
            mel: Mel-spectrogram tensor (shape: [1, 128, time])

        Returns:
            Speech tokens as numpy array (shape: [num_tokens])
        """
        # Run speech tokenizer
        tokens_np = self.model.frontend.speech_tokenizer_session.run(None, {
            self.model.frontend.speech_tokenizer_session.get_inputs()[0].name: mel.cpu().numpy(),
            self.model.frontend.speech_tokenizer_session.get_inputs()[1].name: np.array([mel.shape[2]], dtype=np.int32)
        })[0].flatten()

        print(f"[SpeechTokenizer] Converted mel to tokens: {len(tokens_np)} tokens")

        return tokens_np

    def encode(self, audio_path: str) -> np.ndarray:
        """
        Encode audio to speech tokens (combined operation).

        This internally calls audio_to_mel() then mel_to_tokens().

        Args:
            audio_path: Path to audio file

        Returns:
            Speech tokens as numpy array (shape: [num_tokens])
        """
        # Load and preprocess audio
        audio_16k = self.load_wav(audio_path, 16000)

        # Extract mel spectrogram
        mel = self.whisper.log_mel_spectrogram(audio_16k, n_mels=128)

        # Run speech tokenizer
        tokens_np = self.model.frontend.speech_tokenizer_session.run(None, {
            self.model.frontend.speech_tokenizer_session.get_inputs()[0].name: mel.cpu().numpy(),
            self.model.frontend.speech_tokenizer_session.get_inputs()[1].name: np.array([mel.shape[2]], dtype=np.int32)
        })[0].flatten()

        compression_ratio = audio_16k.shape[1] / len(tokens_np)

        print(f"[SpeechTokenizer] Encoded {audio_path}")
        print(f"[SpeechTokenizer]   Tokens: {len(tokens_np)}")
        print(f"[SpeechTokenizer]   Compression: {compression_ratio:.1f}x")

        return tokens_np

    def encode_to_tensor(self, audio_path: str) -> torch.Tensor:
        """
        Encode audio to speech tokens as tensor.

        Args:
            audio_path: Path to audio file

        Returns:
            Speech tokens as tensor (shape: [1, num_tokens])
        """
        tokens_np = self.encode(audio_path)
        tokens_tensor = torch.from_numpy(tokens_np).unsqueeze(0).long()
        return tokens_tensor

    def encode_with_batch_tokenizer(self, audio_path: str) -> np.ndarray:
        """
        Encode audio to speech tokens using batch tokenizer (v3 batch).

        This uses the speech_tokenizer_v3.batch.onnx model which is optimized
        for batch processing and used internally by Flow and LLM modules.

        Args:
            audio_path: Path to audio file

        Returns:
            Speech tokens as numpy array (shape: [num_tokens])
        """
        if not self.use_batch_tokenizer:
            raise ValueError("Batch tokenizer not initialized. Set use_batch_tokenizer=True during initialization.")

        # Load and preprocess audio
        audio_16k = self.load_wav(audio_path, 16000)

        # Extract mel spectrogram
        mel = self.whisper.log_mel_spectrogram(audio_16k, n_mels=128)

        # Add batch dimension for batch tokenizer
        mel_batch = mel.unsqueeze(0)  # [1, 1, 128, time]

        # Run batch speech tokenizer
        tokens_np = self.batch_tokenizer_session.run(None, {
            self.batch_tokenizer_session.get_inputs()[0].name: mel_batch.cpu().numpy(),
            self.batch_tokenizer_session.get_inputs()[1].name: np.array([mel.shape[2]], dtype=np.int32)
        })[0].flatten()

        compression_ratio = audio_16k.shape[1] / len(tokens_np)

        print(f"[SpeechTokenizer] Encoded {audio_path} (batch tokenizer)")
        print(f"[SpeechTokenizer]   Tokens: {len(tokens_np)}")
        print(f"[SpeechTokenizer]   Compression: {compression_ratio:.1f}x")

        return tokens_np

    def encode_batch(self, audio_paths: list) -> list:
        """
        Encode multiple audio files to speech tokens using batch tokenizer.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of speech token arrays
        """
        if not self.use_batch_tokenizer:
            raise ValueError("Batch tokenizer not initialized. Set use_batch_tokenizer=True during initialization.")

        # Process all audio files
        mels = []
        mel_lengths = []
        for audio_path in audio_paths:
            audio_16k = self.load_wav(audio_path, 16000)
            mel = self.whisper.log_mel_spectrogram(audio_16k, n_mels=128)
            mels.append(mel)
            mel_lengths.append(mel.shape[2])

        # Pad to same length
        max_len = max(mel_lengths)
        mels_padded = []
        for mel in mels:
            if mel.shape[2] < max_len:
                pad = torch.zeros(1, 128, max_len - mel.shape[2])
                mel = torch.cat([mel, pad], dim=2)
            mels_padded.append(mel)

        # Stack into batch
        mel_batch = torch.stack(mels_padded, dim=0)  # [batch, 1, 128, time]

        # Run batch speech tokenizer
        tokens_batch = self.batch_tokenizer_session.run(None, {
            self.batch_tokenizer_session.get_inputs()[0].name: mel_batch.cpu().numpy(),
            self.batch_tokenizer_session.get_inputs()[1].name: np.array(mel_lengths, dtype=np.int32)
        })[0]

        # Extract individual token sequences
        results = []
        for i, audio_path in enumerate(audio_paths):
            tokens = tokens_batch[i, :mel_lengths[i] // 4]  # Account for downsampling
            results.append(tokens.flatten())
            print(f"[SpeechTokenizer] Encoded {audio_path} -> {len(tokens)} tokens")

        return results

    def save_tokens(self, tokens: np.ndarray, output_path: str):
        """Save speech tokens to numpy file."""
        np.save(output_path, tokens)
        print(f"[SpeechTokenizer] Saved tokens to {output_path}")

    def load_tokens(self, tokens_path: str) -> np.ndarray:
        """Load speech tokens from numpy file."""
        tokens = np.load(tokens_path)
        print(f"[SpeechTokenizer] Loaded tokens from {tokens_path}")
        return tokens

    def save_mel(self, mel: torch.Tensor, output_path: str):
        """Save mel-spectrogram to numpy file."""
        np.save(output_path, mel.cpu().numpy())
        print(f"[SpeechTokenizer] Saved mel-spectrogram to {output_path}")

    def load_mel(self, mel_path: str) -> torch.Tensor:
        """Load mel-spectrogram from numpy file."""
        mel = np.load(mel_path)
        mel_tensor = torch.from_numpy(mel)
        print(f"[SpeechTokenizer] Loaded mel-spectrogram from {mel_path}")
        return mel_tensor
