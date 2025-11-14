"""
Basic usage examples for python-genai.

This file demonstrates the core functionality of the Python wrapper
for Chrome's Prompt API.
"""

import asyncio
from python_genai import LanguageModel, Availability


async def check_availability():
    """Check if the Prompt API is available."""
    print("Checking availability...")
    availability = await LanguageModel.availability()
    print(f"Model availability: {availability}")
    return availability


async def get_params():
    """Get model parameters."""
    print("\nGetting model parameters...")
    params = await LanguageModel.params()
    if params:
        print(f"Default temperature: {params.default_temperature}")
        print(f"Max temperature: {params.max_temperature}")
        print(f"Default top-K: {params.default_top_k}")
        print(f"Max top-K: {params.max_top_k}")
    else:
        print("Model parameters not available")
    return params


async def simple_prompt():
    """Simple prompt example."""
    print("\nSimple prompt example...")
    session = await LanguageModel.create()

    result = await session.prompt("Write a haiku about coding.")
    print(f"Response: {result}")

    await session.destroy()


async def streaming_prompt():
    """Streaming prompt example."""
    print("\nStreaming prompt example...")
    session = await LanguageModel.create()

    print("Response: ", end="", flush=True)
    async for chunk in session.prompt_streaming("Write a short story about a robot."):
        print(chunk, end="", flush=True)
    print()

    await session.destroy()


async def system_prompt_example():
    """System prompt example."""
    print("\nSystem prompt example...")
    session = await LanguageModel.create(
        {
            "initialPrompts": [
                {
                    "role": "system",
                    "content": "You are a helpful Python programming assistant who gives concise answers.",
                }
            ]
        }
    )

    result = await session.prompt("How do I read a file in Python?")
    print(f"Response: {result}")

    await session.destroy()


async def conversation_example():
    """Multi-turn conversation example."""
    print("\nConversation example...")
    session = await LanguageModel.create(
        {"initialPrompts": [{"role": "system", "content": "You are a friendly assistant."}]}
    )

    # First message
    result1 = await session.prompt("What's the weather like today?")
    print(f"Assistant: {result1}")

    # Follow-up message
    result2 = await session.prompt("What should I wear based on that?")
    print(f"Assistant: {result2}")

    # Check token usage
    print(f"\nToken usage: {session.input_usage}/{session.input_quota}")

    await session.destroy()


async def measure_usage_example():
    """Example of measuring token usage."""
    print("\nMeasure usage example...")
    session = await LanguageModel.create()

    prompt = "Explain quantum computing in simple terms."
    usage = await session.measure_input_usage(prompt)
    print(f"This prompt will use approximately {usage} tokens")

    print(f"Current quota: {session.input_usage}/{session.input_quota}")

    if session.input_usage + usage < session.input_quota:
        result = await session.prompt(prompt)
        print(f"Response: {result[:100]}...")  # Print first 100 chars
        print(f"New usage: {session.input_usage}/{session.input_quota}")

    await session.destroy()


async def session_cloning_example():
    """Example of session cloning."""
    print("\nSession cloning example...")
    session = await LanguageModel.create(
        {"initialPrompts": [{"role": "system", "content": "You are a creative storyteller."}]}
    )

    # Initial prompt
    await session.prompt("Once upon a time, there was a dragon.")

    # Clone for different story branches
    session1 = await session.clone()
    session2 = await session.clone()

    result1 = await session1.prompt("The dragon was friendly.")
    print(f"Branch 1: {result1}")

    result2 = await session2.prompt("The dragon was fierce.")
    print(f"Branch 2: {result2}")

    await session.destroy()
    await session1.destroy()
    await session2.destroy()


async def structured_output_example():
    """Example of structured output with JSON schema."""
    print("\nStructured output example...")
    session = await LanguageModel.create()

    schema = {
        "type": "object",
        "required": ["sentiment", "score"],
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }

    result = await session.prompt(
        "Analyze this review: The product was amazing, exceeded all expectations!",
        {"responseConstraint": schema},
    )
    print(f"Structured response: {result}")

    await session.destroy()


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Python GenAI - Basic Usage Examples")
    print("=" * 60)

    try:
        # Check availability first
        availability = await check_availability()
        if availability == Availability.UNAVAILABLE:
            print("\n❌ Chrome Prompt API is not available.")
            print("Make sure you're running in Chrome with the API enabled.")
            return

        if availability == Availability.DOWNLOADING:
            print("\n⏳ Model is downloading. Please wait and try again.")
            return

        # Get parameters
        await get_params()

        # Run examples
        await simple_prompt()
        await streaming_prompt()
        await system_prompt_example()
        await conversation_example()
        await measure_usage_example()
        await session_cloning_example()
        await structured_output_example()

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # For Jupyter notebooks, you can run: await main()
    # For regular Python scripts:
    asyncio.run(main())
