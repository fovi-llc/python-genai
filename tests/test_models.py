"""Tests for data models."""

import pytest
from python_genai.models import (
    Availability,
    LanguageModelMessage,
    LanguageModelMessageContent,
    LanguageModelMessageRole,
    LanguageModelMessageType,
    LanguageModelExpected,
    LanguageModelTool,
    LanguageModelCreateOptions,
    LanguageModelPromptOptions,
    LanguageModelParams,
)


def test_availability_enum():
    """Test Availability enum values."""
    assert Availability.UNAVAILABLE.value == "unavailable"
    assert Availability.DOWNLOADABLE.value == "downloadable"
    assert Availability.DOWNLOADING.value == "downloading"
    assert Availability.AVAILABLE.value == "available"


def test_message_role_enum():
    """Test LanguageModelMessageRole enum values."""
    assert LanguageModelMessageRole.SYSTEM.value == "system"
    assert LanguageModelMessageRole.USER.value == "user"
    assert LanguageModelMessageRole.ASSISTANT.value == "assistant"


def test_message_type_enum():
    """Test LanguageModelMessageType enum values."""
    assert LanguageModelMessageType.TEXT.value == "text"
    assert LanguageModelMessageType.IMAGE.value == "image"
    assert LanguageModelMessageType.AUDIO.value == "audio"


def test_message_content_creation():
    """Test LanguageModelMessageContent creation and serialization."""
    content = LanguageModelMessageContent(type=LanguageModelMessageType.TEXT, value="Hello, world!")
    assert content.type == LanguageModelMessageType.TEXT
    assert content.value == "Hello, world!"

    content_dict = content.to_dict()
    assert content_dict["type"] == "text"
    assert content_dict["value"] == "Hello, world!"


def test_message_creation_with_string():
    """Test LanguageModelMessage creation with string content."""
    message = LanguageModelMessage(role=LanguageModelMessageRole.USER, content="Test message")
    assert message.role == LanguageModelMessageRole.USER
    assert message.content == "Test message"
    assert message.prefix is False

    message_dict = message.to_dict()
    assert message_dict["role"] == "user"
    assert message_dict["content"] == "Test message"
    assert message_dict["prefix"] is False


def test_message_creation_with_content_list():
    """Test LanguageModelMessage creation with content list."""
    content1 = LanguageModelMessageContent(type=LanguageModelMessageType.TEXT, value="Hello")
    content2 = LanguageModelMessageContent(type=LanguageModelMessageType.TEXT, value="World")
    message = LanguageModelMessage(
        role=LanguageModelMessageRole.ASSISTANT, content=[content1, content2]
    )

    assert message.role == LanguageModelMessageRole.ASSISTANT
    assert len(message.content) == 2

    message_dict = message.to_dict()
    assert message_dict["role"] == "assistant"
    assert len(message_dict["content"]) == 2
    assert message_dict["content"][0]["value"] == "Hello"
    assert message_dict["content"][1]["value"] == "World"


def test_expected_creation():
    """Test LanguageModelExpected creation and serialization."""
    expected = LanguageModelExpected(type=LanguageModelMessageType.TEXT, languages=["en", "es"])
    assert expected.type == LanguageModelMessageType.TEXT
    assert expected.languages == ["en", "es"]

    expected_dict = expected.to_dict()
    assert expected_dict["type"] == "text"
    assert expected_dict["languages"] == ["en", "es"]


def test_tool_creation():
    """Test LanguageModelTool creation and serialization."""

    def my_function(arg1, arg2):
        return f"{arg1} + {arg2}"

    tool = LanguageModelTool(
        name="my_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {}},
        execute=my_function,
    )

    assert tool.name == "my_tool"
    assert tool.description == "A test tool"
    assert callable(tool.execute)

    tool_dict = tool.to_dict()
    assert tool_dict["name"] == "my_tool"
    assert tool_dict["description"] == "A test tool"
    assert "execute" not in tool_dict  # Execute function should not be serialized


def test_create_options_serialization():
    """Test LanguageModelCreateOptions serialization."""
    options = LanguageModelCreateOptions(
        temperature=0.8,
        top_k=40,
        expected_inputs=[
            LanguageModelExpected(type=LanguageModelMessageType.TEXT, languages=["en"])
        ],
    )

    options_dict = options.to_dict()
    assert options_dict["temperature"] == 0.8
    assert options_dict["topK"] == 40
    assert len(options_dict["expectedInputs"]) == 1
    assert options_dict["expectedInputs"][0]["type"] == "text"


def test_prompt_options_serialization():
    """Test LanguageModelPromptOptions serialization."""
    schema = {"type": "object", "properties": {"rating": {"type": "number"}}}
    options = LanguageModelPromptOptions(
        response_constraint=schema, omit_response_constraint_input=True
    )

    options_dict = options.to_dict()
    assert options_dict["responseConstraint"] == schema
    assert options_dict["omitResponseConstraintInput"] is True


def test_params_from_dict():
    """Test LanguageModelParams creation from dictionary."""
    data = {
        "defaultTopK": 40,
        "maxTopK": 128,
        "defaultTemperature": 0.8,
        "maxTemperature": 2.0,
    }

    params = LanguageModelParams.from_dict(data)
    assert params.default_top_k == 40
    assert params.max_top_k == 128
    assert params.default_temperature == 0.8
    assert params.max_temperature == 2.0
