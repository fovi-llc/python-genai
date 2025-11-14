# python-ai

Python wrapper for Chrome's built-in AI Prompt API, enabling direct access to browser-provided language models from Jupyter notebooks.

## Features

- 🤖 Direct access to Chrome's built-in AI via the Prompt API
- 📓 Seamless Jupyter integration using AnyWidget
- 🔄 Bidirectional communication via traitlets
- 🎯 Type-safe Python API matching the Web IDL specification
- 🌊 Support for streaming responses
- 🛠️ Tool use capabilities
- 🖼️ Multimodal input support (text, images, audio)
- 📊 Session management and cloning
- 🔒 Privacy-focused on-device processing

## Installation

```bash
pip install python-ai
```

## Requirements

- Python 3.10+
- Chrome browser with Prompt API enabled
- Running in a Jupyter environment (JupyterLab, Jupyter Notebook, etc.)

## Quick Start

```python
from python_ai import LanguageModel

# Create a language model session
session = await LanguageModel.create()

# Simple prompt
result = await session.prompt("Write me a poem about Python.")
print(result)

# Streaming response
async for chunk in session.prompt_streaming("Write me a long story."):
    print(chunk, end="", flush=True)
```

## Usage Examples

### System Prompts

```python
from python_ai import LanguageModel, LanguageModelMessage, LanguageModelMessageRole

session = await LanguageModel.create({
    "initialPrompts": [
        {
            "role": "system",
            "content": "You are a helpful Python programming assistant."
        }
    ]
})

response = await session.prompt("How do I read a file in Python?")
print(response)
```

### Checking Availability

```python
from python_ai import LanguageModel, Availability

# Check if the API is available
availability = await LanguageModel.availability()
print(f"Model availability: {availability}")

if availability == Availability.AVAILABLE:
    session = await LanguageModel.create()
    # Use the session...
```

### Configuring Temperature and Top-K

```python
# Get default parameters
params = await LanguageModel.params()
print(f"Default temperature: {params.default_temperature}")
print(f"Max temperature: {params.max_temperature}")

# Create session with custom parameters
session = await LanguageModel.create({
    "temperature": 0.8,
    "topK": 40
})
```

### Session Management

```python
# Create a session
session = await LanguageModel.create()

# Use the session
result1 = await session.prompt("Tell me about AI.")

# Clone for different conversation branches
session2 = await session.clone()
result2 = await session2.prompt("Now tell me about machine learning.")

# Destroy when done
await session.destroy()
```

### Measuring Token Usage

```python
session = await LanguageModel.create()

# Check current usage
print(f"Current usage: {session.input_usage}/{session.input_quota}")

# Measure potential usage before prompting
usage = await session.measure_input_usage("This is my prompt")
print(f"This prompt would use: {usage} tokens")

# Prompt if there's enough quota
if session.input_usage + usage < session.input_quota:
    result = await session.prompt("This is my prompt")
```

### Structured Output with JSON Schema

```python
schema = {
    "type": "object",
    "required": ["rating"],
    "properties": {
        "rating": {
            "type": "number",
            "minimum": 0,
            "maximum": 5
        }
    }
}

result = await session.prompt(
    "Rate this: The food was excellent!",
    {"responseConstraint": schema}
)

import json
data = json.parse(result)
print(f"Rating: {data['rating']}")
```

## API Reference

### LanguageModel

Main class for interacting with the language model.

#### Class Methods

- `create(options=None)` - Create a new language model session
- `availability(options=None)` - Check model availability
- `params()` - Get model parameters

#### Instance Methods

- `prompt(input, options=None)` - Send a prompt and get response
- `prompt_streaming(input, options=None)` - Send a prompt and stream response
- `append(input, options=None)` - Append messages without getting response
- `measure_input_usage(input, options=None)` - Measure token usage
- `clone(options=None)` - Clone the session
- `destroy()` - Destroy the session

#### Properties

- `input_usage` - Current token usage
- `input_quota` - Maximum token quota
- `top_k` - Top-K sampling parameter
- `temperature` - Temperature sampling parameter

### Data Models

- `LanguageModelMessage` - A message in the conversation
- `LanguageModelMessageContent` - Content with type and value
- `LanguageModelMessageRole` - Message role (system/user/assistant)
- `LanguageModelMessageType` - Content type (text/image/audio)
- `LanguageModelCreateOptions` - Options for creating sessions
- `LanguageModelPromptOptions` - Options for prompting
- `LanguageModelParams` - Model parameters
- `Availability` - Availability status enum

## Architecture

This package uses:

- **AnyWidget** for Jupyter integration
- **Traitlets** for bidirectional Python ↔ JavaScript communication
- **Chrome Prompt API** for accessing built-in language models

The communication flow:
1. Python code calls methods on `LanguageModel`
2. Requests are serialized via traitlets to JavaScript
3. JavaScript calls Chrome's native Prompt API
4. Results are sent back via traitlets to Python
5. Python code receives async results

## Specifications

This implementation follows:
- [Chrome Prompt API Specification](https://webmachinelearning.github.io/prompt-api/)
- [WebIDL Interface Definitions](https://webmachinelearning.github.io/prompt-api/#idl-index)
- [Chrome Developer Documentation](https://developer.chrome.com/docs/ai/prompt-api)

## License

Apache License 2.0 - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Links

- [GitHub Repository](https://github.com/fovi-llc/python-ai)
- [Chrome Prompt API Documentation](https://developer.chrome.com/docs/ai/prompt-api)
- [Web Machine Learning Community Group](https://github.com/webmachinelearning/prompt-api)