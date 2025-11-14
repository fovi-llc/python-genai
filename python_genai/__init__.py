"""Python wrapper for Chrome's built-in AI Prompt API."""

from .models import (
    Availability,
    LanguageModelMessageRole,
    LanguageModelMessageType,
    LanguageModelMessage,
    LanguageModelMessageContent,
    LanguageModelExpected,
    LanguageModelTool,
    LanguageModelCreateOptions,
    LanguageModelPromptOptions,
    LanguageModelAppendOptions,
    LanguageModelCloneOptions,
    LanguageModelParams,
)
from .language_model import LanguageModel

__version__ = "0.1.0"

__all__ = [
    "LanguageModel",
    "LanguageModelParams",
    "Availability",
    "LanguageModelMessageRole",
    "LanguageModelMessageType",
    "LanguageModelMessage",
    "LanguageModelMessageContent",
    "LanguageModelExpected",
    "LanguageModelTool",
    "LanguageModelCreateOptions",
    "LanguageModelPromptOptions",
    "LanguageModelAppendOptions",
    "LanguageModelCloneOptions",
]
