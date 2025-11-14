"""Data models for the Prompt API matching the IDL specification."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class Availability(str, Enum):
    """Availability status for language model features."""

    UNAVAILABLE = "unavailable"
    DOWNLOADABLE = "downloadable"
    DOWNLOADING = "downloading"
    AVAILABLE = "available"


class LanguageModelMessageRole(str, Enum):
    """Role of a message in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LanguageModelMessageType(str, Enum):
    """Type of content in a message."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass
class LanguageModelMessageContent:
    """Content of a message with type and value."""

    type: LanguageModelMessageType
    value: Any  # Can be str, ImageBitmapSource, AudioBuffer, BufferSource, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"type": self.type.value, "value": self.value}


@dataclass
class LanguageModelMessage:
    """A message in the conversation."""

    role: LanguageModelMessageRole
    content: Union[str, List[LanguageModelMessageContent]]
    prefix: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        if isinstance(self.content, str):
            content = self.content
        else:
            content = [c.to_dict() for c in self.content]

        return {"role": self.role.value, "content": content, "prefix": self.prefix}


@dataclass
class LanguageModelExpected:
    """Expected input or output type and languages."""

    type: LanguageModelMessageType
    languages: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {"type": self.type.value}
        if self.languages is not None:
            result["languages"] = self.languages
        return result


@dataclass
class LanguageModelTool:
    """A tool that the language model can invoke."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    execute: Callable[..., Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excluding execute function)."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class LanguageModelCreateOptions:
    """Options for creating a language model session."""

    top_k: Optional[float] = None
    temperature: Optional[float] = None
    expected_inputs: Optional[List[LanguageModelExpected]] = None
    expected_outputs: Optional[List[LanguageModelExpected]] = None
    tools: Optional[List[LanguageModelTool]] = None
    initial_prompts: Optional[List[LanguageModelMessage]] = None
    monitor: Optional[Callable[[Any], None]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        if self.top_k is not None:
            result["topK"] = self.top_k
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.expected_inputs is not None:
            result["expectedInputs"] = [e.to_dict() for e in self.expected_inputs]
        if self.expected_outputs is not None:
            result["expectedOutputs"] = [e.to_dict() for e in self.expected_outputs]
        if self.tools is not None:
            result["tools"] = [t.to_dict() for t in self.tools]
        if self.initial_prompts is not None:
            result["initialPrompts"] = [p.to_dict() for p in self.initial_prompts]
        return result


@dataclass
class LanguageModelPromptOptions:
    """Options for prompting the language model."""

    response_constraint: Optional[Union[Dict[str, Any], Any]] = None  # Can be JSON schema or RegExp
    omit_response_constraint_input: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        if self.response_constraint is not None:
            result["responseConstraint"] = self.response_constraint
        if self.omit_response_constraint_input:
            result["omitResponseConstraintInput"] = self.omit_response_constraint_input
        return result


@dataclass
class LanguageModelAppendOptions:
    """Options for appending messages to the session."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {}


@dataclass
class LanguageModelCloneOptions:
    """Options for cloning a session."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {}


@dataclass
class LanguageModelParams:
    """Parameters for the language model."""

    default_top_k: int
    max_top_k: int
    default_temperature: float
    max_temperature: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LanguageModelParams":
        """Create from dictionary."""
        return cls(
            default_top_k=data["defaultTopK"],
            max_top_k=data["maxTopK"],
            default_temperature=data["defaultTemperature"],
            max_temperature=data["maxTemperature"],
        )
