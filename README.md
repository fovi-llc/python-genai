# python-genai

Python bindings for Chrome's built-in Prompt API using [AnyWidget](https://github.com/manzt/anywidget).

## Installation

```bash
pip install python-genai
```

## Usage

The package is designed for Jupyter environments. Creating a `PromptAPI` automatically displays the
underlying AnyWidget that bridges calls to `window.ai`.

```python
from python_genai import PromptAPI

api = PromptAPI()
capabilities = await api.get_capabilities()

session = await api.create_text_session()
response = await session.prompt("Write a haiku about autumn sunsets.")
print(response)
```

### Streaming

```python
stream = await session.prompt_stream("Draft a product update email." )
async for chunk in stream:
    print(chunk, end="")
await stream.result()
```

### Context managers

```python
async with await api.create_text_session() as session:
    print(await session.prompt("Summarize the meeting notes."))
```

Refer to the [Prompt API specification](https://webmachinelearning.github.io/prompt-api/) and
[Chrome documentation](https://developer.chrome.com/docs/ai/prompt-api) for supported options.
