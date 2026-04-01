"""
CosyVoice Tokenizer Reconstruction Package

Provides tools for:
- Encoding audio to speech tokens using CosyVoice3's tokenizer
- Decoding speech tokens back to audio using Flow model + HiFi-GAN
- Voice conversion (VC mode): preserves source prosody, changes timbre

Main classes:
- SpeechTokenizer: Encode audio to speech tokens
- SpeechDecoder: Decode speech tokens to audio
- SpeechReconstructor: High-level interface for encode/decode
- VoiceConverter: Voice conversion using CosyVoice's VC mode
"""

from .encoder import SpeechTokenizer
from .decoder import SpeechDecoder
from .reconstructor import SpeechReconstructor
from .voice_converter import VoiceConverter

__version__ = "0.1.0"
__all__ = ["SpeechTokenizer", "SpeechDecoder", "SpeechReconstructor", "VoiceConverter"]
