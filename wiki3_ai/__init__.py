"""Python wrapper for Chrome's built-in AI Prompt API."""

from .language_model import LanguageModel, LanguageModelWidget

__version__ = "0.5.1"

__all__ = [
    "LanguageModel",
    "LanguageModelWidget",
]
