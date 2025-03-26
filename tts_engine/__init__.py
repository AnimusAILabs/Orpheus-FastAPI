"""
TTS Engine package for Orpheus text-to-speech system.

This package contains the core components for audio generation:
- inference.py: Token generation and API handling
- speechpipe.py: Audio conversion pipeline
"""

# Make key components available at package level
from .inference import (
    generate_speech_from_api,
    generate_tokens_from_api,
    tokens_decoder,
    load_model,
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    list_available_voices
)

__all__ = [
    'generate_speech_from_api',
    'generate_tokens_from_api',
    'tokens_decoder',
    'load_model',
    'AVAILABLE_VOICES',
    'DEFAULT_VOICE'
]
