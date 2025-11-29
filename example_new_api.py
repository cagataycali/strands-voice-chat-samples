"""Example showcasing the new simplified Bidi API.

This demonstrates the cleaner import paths and automatic audio configuration
that comes with the updated SDK.
"""

import asyncio
from strands.experimental.bidi import (
    BidiAgent,
    BidiNovaSonicModel,
    BidiAudioIO,
    BidiTextIO,
)
from strands_tools import calculator, current_time


async def main():
    """Run a bidirectional agent with the new API."""
    
    # 1. Simple setup with defaults
    model = BidiNovaSonicModel()  # Uses default us-east-1 region
    agent = BidiAgent(model=model, tools=[calculator, current_time])
    
    # Audio I/O automatically uses model's audio config - no hardcoding!
    audio_io = BidiAudioIO()
    text_io = BidiTextIO()
    
    print("🎤 Simple Bidi Agent Example")
    print("Model automatically configures audio at:", model.config["audio"])
    print("\nTry: 'What is 25 times 8?' or 'What time is it?'")
    print("Press Ctrl+C to exit\n" + "-" * 50)
    
    await agent.run(
        inputs=[audio_io.input()],
        outputs=[audio_io.output(), text_io.output()]
    )


async def main_custom():
    """Run with custom configuration."""
    
    # 2. Custom voice and region
    model = BidiNovaSonicModel(
        region="us-west-2",
        config={
            "audio": {
                "voice": "tiffany",  # US feminine voice
                "input_rate": 48000,  # Custom high-quality input
            }
        }
    )
    
    # Audio I/O automatically adapts to model's config
    audio_io = BidiAudioIO()
    
    # Can also override specific audio device settings
    audio_io_custom = BidiAudioIO(
        input_device_index=1,  # Use specific mic
        output_device_index=2,  # Use specific speaker
    )
    
    agent = BidiAgent(
        model=model,
        tools=[calculator],
        system_prompt="You are a helpful math tutor. Speak clearly."
    )
    
    print("🎤 Custom Bidi Agent Example")
    print("Custom audio config:", model.config["audio"])
    print("\nTry asking math questions!")
    print("Press Ctrl+C to exit\n" + "-" * 50)
    
    await agent.run(
        inputs=[audio_io.input()],
        outputs=[audio_io.output(), text_io.output()],
        invocation_state={
            "user_id": "example_user",
            "session_id": "demo_session",
        }
    )


if __name__ == "__main__":
    import sys
    
    # Choose which example to run
    if len(sys.argv) > 1 and sys.argv[1] == "custom":
        print("Running custom configuration example...\n")
        asyncio.run(main_custom())
    else:
        print("Running simple default example...\n")
        print("(Run with 'python example_new_api.py custom' for custom config)\n")
        asyncio.run(main())
