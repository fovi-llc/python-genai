# Architecture

This document describes the architecture of the python-genai package.

## Overview

python-genai is a Python wrapper for Chrome's built-in AI Prompt API that enables seamless access to browser-provided language models from Jupyter notebooks. The implementation uses minimal code and directly maps to the Web IDL specifications.

## Components

### 1. Data Models (`python_genai/models.py`)

Pure Python dataclasses that mirror the IDL specification:

- **Enums**: `Availability`, `LanguageModelMessageRole`, `LanguageModelMessageType`
- **Message structures**: `LanguageModelMessage`, `LanguageModelMessageContent`
- **Configuration options**: `LanguageModelCreateOptions`, `LanguageModelPromptOptions`, etc.
- **Parameters**: `LanguageModelParams`

Each model includes:
- Type-safe Python definitions
- `to_dict()` methods for serialization to JavaScript
- `from_dict()` methods for deserialization from JavaScript (where applicable)

### 2. Language Model (`python_genai/language_model.py`)

#### LanguageModelWidget (AnyWidget)

The bridge between Python and JavaScript:

```
┌─────────────────────┐
│  Python (Backend)   │
│  LanguageModel      │
│  class              │
└──────────┬──────────┘
           │ traitlets
           │ (request/response)
┌──────────▼──────────┐
│   AnyWidget         │
│   JavaScript ESM    │
└──────────┬──────────┘
           │ Chrome API
┌──────────▼──────────┐
│  Chrome Prompt API  │
│  (Browser Native)   │
└─────────────────────┘
```

**Key traits:**
- `request`: Dict trait for Python → JavaScript requests
- `response`: Dict trait for JavaScript → Python responses
- `error`: Dict trait for global errors
- `quota_overflow_event`: Dict trait for quota overflow notifications
- `stream_chunk`: Dict trait for streaming chunks

#### LanguageModel Class

The main Python API that users interact with:

**Class methods:**
- `create(options)` - Create a new session
- `availability(options)` - Check model availability
- `params()` - Get model parameters

**Instance methods:**
- `prompt(input, options)` - Send a prompt and get response
- `prompt_streaming(input, options)` - Stream responses
- `append(input, options)` - Append messages without response
- `measure_input_usage(input, options)` - Measure token usage
- `clone(options)` - Clone the session
- `destroy()` - Destroy the session

**Properties:**
- `input_usage` - Current token usage
- `input_quota` - Maximum token quota
- `top_k` - Top-K sampling parameter
- `temperature` - Temperature sampling parameter

## Communication Flow

### Request Flow (Python → JavaScript)

1. User calls Python method (e.g., `session.prompt("Hello")`)
2. Python method prepares request data with unique ID
3. Request is set on the `request` traitlet
4. AnyWidget syncs the traitlet to JavaScript
5. JavaScript handler receives request via `model.on('change:request', ...)`
6. JavaScript calls Chrome Prompt API
7. Result is set on the `response` traitlet
8. AnyWidget syncs the response back to Python
9. Python resolves the awaited Future with the result

### Error Handling

Errors are propagated through the response mechanism:
- JavaScript catches exceptions and includes them in the response
- Python checks for errors in the response and raises Python exceptions
- Type information is preserved (e.g., `NotSupportedError`, `QuotaExceededError`)

### Streaming

Streaming uses an additional `stream_chunk` traitlet:
1. JavaScript sends chunks via `stream_chunk` as they arrive
2. Python collects chunks in a buffer
3. Python yields chunks as an async iterator
4. Final response includes the complete text

## Session Management

Sessions are managed with unique IDs:
- Each session gets a UUID generated in Python
- JavaScript stores sessions in a map by ID
- Clone operations create new session IDs
- Destroy operations remove sessions from the map

## Type Safety

The implementation uses Python's type system:
- All public methods have type hints
- Enums prevent invalid values
- Dataclasses validate structure
- Union types for flexible inputs (e.g., `str | List[LanguageModelMessage]`)

## Minimal Code Philosophy

The implementation follows the principle of using the least extraneous code:

1. **Direct IDL mapping**: Data structures mirror the Web IDL specification
2. **No wrapper abstractions**: Methods directly call Chrome API equivalents
3. **Standard tools**: Uses AnyWidget and traitlets (industry-standard Jupyter tools)
4. **No intermediate formats**: Direct serialization to/from JavaScript objects
5. **Thin Python layer**: Python code primarily handles async coordination and type safety

## Dependencies

- **anywidget** (>=0.9.0): Jupyter widget framework
- **traitlets** (>=5.0.0): Reactive Python properties for syncing
- **ipywidgets** (via anywidget): Jupyter widget infrastructure

All dependencies are mature, widely-used libraries in the Jupyter ecosystem.

## Browser Requirements

- Chrome browser with Prompt API enabled
- Running in a Jupyter environment (JupyterLab, Jupyter Notebook, Google Colab, etc.)
- Chrome must support `window.ai.languageModel` API

## Testing

The package includes unit tests for:
- Data model creation and serialization
- Enum values
- Options handling
- Parameter conversion

Tests use pytest and cover all data models comprehensively.

## Future Extensibility

The architecture supports future extensions:

1. **Additional IDL features**: New Chrome API features can be added by:
   - Adding corresponding methods to JavaScript handler
   - Adding Python methods to LanguageModel class
   - No changes needed to the traitlet communication layer

2. **Error types**: New error types automatically propagate through the system

3. **Events**: Additional events can be added as new traitlets

4. **Multimodal support**: Already structured to support images, audio, and future modalities

## Security Considerations

- Cross-origin restrictions enforced by Chrome API
- No sensitive data stored in the package
- All processing happens in the user's browser
- Session IDs are UUIDs (not predictable)
- JavaScript code runs in the browser's sandbox

## Performance

- Async/await prevents blocking
- Direct communication (no HTTP overhead)
- Sessions reuse browser's loaded model
- Cloning is efficient (browser-level optimization)
- Streaming reduces latency for long responses
