# Installation Guide

This guide will help you install all dependencies needed to run the Strands Voice Chat tools.

## Quick Install

Install everything with one command:

```bash
pip install -e .
```

This will install:
- **Strands Agents SDK** from the forked repository with bidirectional streaming support
- **WebSockets** for browser-based voice chat server
- **PyAudio** for speech-to-speech functionality
- **AWS SDK** for Nova Sonic bidirectional streaming
- All required dependencies

## Install with All Providers

To include support for OpenAI Realtime API and Gemini Live:

```bash
pip install -e ".[all]"
```

## Install for Development

To include development tools (pytest, ruff, mypy):

```bash
pip install -e ".[dev]"
```

## Manual Installation

If you prefer to install components separately:

### 1. Install Strands Agents SDK (with bidirectional streaming)

```bash
pip install "strands-agents[bidirectional-streaming] @ git+https://github.com/mehtarac/sdk-python.git"
```

### 2. Install WebSocket Server Dependencies

```bash
pip install "websockets>=12.0,<14.0"
```

### 3. Install Audio Processing Dependencies

```bash
pip install pyaudio>=0.2.13
```

### 4. Install AWS Dependencies (for Nova Sonic)

```bash
pip install aws_sdk_bedrock_runtime smithy-aws-core>=0.0.1 rx>=3.2.0 pytz
```

### 5. Optional: OpenAI Realtime API

```bash
pip install "openai>=1.68.0,<2.0.0"
```

### 6. Optional: Gemini Live

```bash
pip install "google-genai>=1.32.0,<2.0.0"
```

## Troubleshooting

### PyAudio Installation Issues

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

### AWS SDK Installation Issues

If `aws_sdk_bedrock_runtime` fails to install:

1. Ensure you have Python 3.10 or higher
2. Try upgrading pip: `pip install --upgrade pip`
3. Install build tools:
   - **macOS:** `xcode-select --install`
   - **Ubuntu/Debian:** `sudo apt-get install build-essential`
   - **Windows:** Install Visual Studio Build Tools

## Verify Installation

After installation, verify everything is working:

```python
from strands import Agent

agent = Agent()

# Check if speech_to_speech tool is available
print("speech_to_speech" in agent.tool_registry.registry)

# Check if voice_chat_server tool is available
print("voice_chat_server" in agent.tool_registry.registry)
```

## Environment Variables

Set up your environment variables for different providers:

### Nova Sonic (AWS Bedrock)

```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"  # Optional, defaults to us-east-1
```

### OpenAI Realtime API

```bash
export OPENAI_API_KEY="sk-..."
```

### Gemini Live

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
# Or set API key directly
export GOOGLE_API_KEY="your_api_key"
```

## Next Steps

After successful installation:

1. **Read the documentation:** Check `README.md` for usage examples
2. **Test speech-to-speech:** Run `python -c "from strands import Agent; agent = Agent(); agent.tool.speech_to_speech(action='start')"`
3. **Test voice chat server:** Run `python -c "from strands import Agent; agent = Agent(); agent.tool.voice_chat_server(action='start')"`
4. **Open the web UI:** Navigate to `docs/index.html` in your browser

## Requirements

- **Python:** 3.10 or higher
- **Operating System:** macOS, Linux, or Windows
- **Audio:** Microphone and speakers for speech-to-speech
- **Browser:** Modern browser (Chrome, Firefox, Safari) for voice chat server

## Support

For issues or questions:
- Check the troubleshooting section above
- Review the main README.md
- Check Strands Agents documentation: https://strandsagents.com
- Open an issue on the SDK repository
