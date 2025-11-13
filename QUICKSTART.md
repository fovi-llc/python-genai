# Quick Start Guide

Get started with python-genai in 5 minutes!

## Prerequisites

1. **Chrome Browser**: You need Chrome with the Prompt API enabled
   - Download [Chrome Canary](https://www.google.com/chrome/canary/) or [Chrome Dev](https://www.google.com/chrome/dev/)
   - Enable the Prompt API flag: `chrome://flags/#prompt-api-for-gemini-nano`

2. **Jupyter Environment**: JupyterLab, Jupyter Notebook, or Google Colab

3. **Python 3.8+**

## Installation

```bash
pip install python-genai
```

## Your First Prompt

Open a Jupyter notebook in Chrome and run:

```python
from python_genai import LanguageModel

# Create a session
session = await LanguageModel.create()

# Send a prompt
result = await session.prompt("Write a haiku about Python.")
print(result)
```

That's it! You're now using Chrome's built-in AI.

## Common Use Cases

### 1. Check Availability

Always check if the API is available:

```python
from python_genai import LanguageModel, Availability

availability = await LanguageModel.availability()

if availability == Availability.AVAILABLE:
    session = await LanguageModel.create()
    # Use the session...
elif availability == Availability.DOWNLOADING:
    print("Model is downloading. Please wait...")
else:
    print("API not available in this browser")
```

### 2. Streaming Responses

For long responses, stream them as they're generated:

```python
async for chunk in session.prompt_streaming("Write a short story."):
    print(chunk, end="", flush=True)
```

### 3. Set Context with System Prompts

Guide the AI's behavior:

```python
session = await LanguageModel.create({
    "initialPrompts": [{
        "role": "system",
        "content": "You are a helpful Python tutor."
    }]
})

answer = await session.prompt("How do I use list comprehensions?")
```

### 4. Multi-turn Conversations

The session remembers context:

```python
# First question
await session.prompt("What's the capital of France?")

# Follow-up (AI remembers we're talking about France)
answer = await session.prompt("What's the population?")
```

### 5. Structured Output

Get JSON responses:

```python
import json

schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string"},
        "score": {"type": "number"}
    }
}

result = await session.prompt(
    "Analyze: This is amazing!",
    {"responseConstraint": schema}
)

data = json.loads(result)
print(f"Sentiment: {data['sentiment']}, Score: {data['score']}")
```

### 6. Monitor Token Usage

Keep track of your usage:

```python
print(f"Used: {session.input_usage}/{session.input_quota} tokens")

# Check before prompting
usage = await session.measure_input_usage("My prompt")
if session.input_usage + usage < session.input_quota:
    result = await session.prompt("My prompt")
```

## Best Practices

1. **Always destroy sessions** when done:
   ```python
   await session.destroy()
   ```

2. **Check availability** before creating sessions

3. **Handle errors** gracefully:
   ```python
   try:
       result = await session.prompt("Hello")
   except Exception as e:
       print(f"Error: {e}")
   ```

4. **Use streaming** for better UX with long responses

5. **Clone sessions** for conversation branches:
   ```python
   branch1 = await session.clone()
   branch2 = await session.clone()
   ```

## Troubleshooting

### "Chrome Prompt API is not available"

- Make sure you're using Chrome (not another browser)
- Enable the flag: `chrome://flags/#prompt-api-for-gemini-nano`
- Restart Chrome
- Check if you're running in Jupyter (not a regular Python script)

### "Model is downloading"

The first time you use the API, Chrome needs to download the model:
- Wait for the download to complete
- Check progress in Chrome's task manager
- You only need to download once

### "NotSupportedError"

Some features may not be available:
- Check model capabilities with `await LanguageModel.params()`
- Ensure your options are within supported ranges
- Some multimodal features may require specific model versions

### Jupyter not recognizing `await`

Make sure you're using:
- Jupyter with IPython 7.0+
- Top-level `await` (not inside a function without async)
- Jupyter notebook cells (not a regular Python file)

## Next Steps

- Check out the full [README](README.md) for detailed documentation
- Explore [examples/demo.ipynb](examples/demo.ipynb) for more examples
- Read the [ARCHITECTURE](ARCHITECTURE.md) to understand how it works
- Browse the [Chrome Prompt API docs](https://developer.chrome.com/docs/ai/prompt-api)

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/fovi-llc/python-genai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fovi-llc/python-genai/discussions)
- **Chrome API**: [Chrome AI Documentation](https://developer.chrome.com/docs/ai/prompt-api)

Happy coding! 🚀
