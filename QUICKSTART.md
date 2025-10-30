# Quick Start Guide

Get started with Strands Voice Chat in just a few steps!

## Installation

### Option 1: Automated Installation (Recommended)

```bash
./setup.sh
```

The script will:
- Check Python version (3.10+ required)
- Install PortAudio if needed (macOS only)
- Install all dependencies
- Verify the installation

### Option 2: Manual Installation

```bash
pip install -e .
```

### Option 3: Using requirements.txt

```bash
pip install -r requirements.txt
```

## Setup Environment

Set your AWS credentials for Nova Sonic:

```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"  # Optional
```

## Usage

### Speech-to-Speech (Terminal)

Talk directly to AI through your microphone and speakers:

```python
from strands import Agent

agent = Agent()

# Start speech session
agent.tool.speech_to_speech(action="start", provider="novasonic")

# Speak into your microphone!
# Say "Can you stop the conversation?" to end
```

**Voice-activated features:**
- 🎤 Continuous microphone input
- 🔊 Real-time audio output
- ⚡ Auto-interrupt when you speak (VAD)
- 🛠️ Full tool access via voice commands
- 🛑 Voice-activated stopping

### Voice Chat Server (Browser)

Multi-user web-based voice chat:

```python
from strands import Agent

agent = Agent()

# Start WebSocket server
agent.tool.voice_chat_server(action="start", provider="novasonic")

# Open docs/index.html in your browser
# Connect and start speaking!
```

**Features:**
- 🌐 Browser-based interface
- 👥 Multiple concurrent users
- 🎨 Beautiful gradient UI
- 📊 Audio visualizer
- 💬 Text transcript display

## Examples

### Use Different Voice IDs

```python
# Italian voice (Beatrice)
agent.tool.speech_to_speech(
    action="start",
    provider="novasonic",
    model_settings={"voice_id": "beatrice"}
)

# German voice (Lennart)
agent.tool.voice_chat_server(
    action="start",
    provider="novasonic",
    model_settings={"voice_id": "lennart"}
)
```

**Available voices:**
- **English (US):** tiffany (feminine), matthew (masculine)
- **English (GB):** amy
- **French:** ambre, florian
- **Italian:** beatrice, lorenzo
- **German:** greta, lennart
- **Spanish:** lupe, carlos

### Use Different Providers

```python
# OpenAI Realtime API
agent.tool.speech_to_speech(
    action="start",
    provider="openai",
    model_settings={
        "session": {
            "voice": "shimmer",
            "turn_detection": {
                "threshold": 0.6,
                "silence_duration_ms": 800
            }
        }
    }
)

# Gemini Live
agent.tool.voice_chat_server(
    action="start",
    provider="gemini_live",
    model_settings={
        "params": {
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": "Kore"}
                }
            }
        }
    }
)
```

### Custom System Prompt

```python
agent.tool.speech_to_speech(
    action="start",
    provider="novasonic",
    system_prompt="You are a friendly language tutor. Speak slowly and clearly."
)
```

### Specific Tools Only

```python
# Only allow calculator and time tools
agent.tool.speech_to_speech(
    action="start",
    provider="novasonic",
    tools=["calculator", "current_time"]
)
```

## Status and Control

```python
# Check active sessions
agent.tool.speech_to_speech(action="status")

# Stop specific session
agent.tool.speech_to_speech(action="stop", session_id="speech_20251029_150000")

# Stop all sessions
agent.tool.speech_to_speech(action="stop")

# Server status
agent.tool.voice_chat_server(action="status")

# Stop server
agent.tool.voice_chat_server(action="stop", port=8765)
```

## Testing Voice Commands

Try these voice commands to test tool usage:

- **Math:** "What is 25 times 17?"
- **Time:** "What's the current time?"
- **AWS:** "List my S3 buckets"
- **Stop:** "Can you stop the conversation?"

## Troubleshooting

### No audio input/output

**Check PyAudio installation:**
```bash
python3 -c "import pyaudio; print('PyAudio OK')"
```

**Reinstall PyAudio:**
```bash
# macOS
brew install portaudio
pip install --force-reinstall pyaudio

# Ubuntu/Debian
sudo apt-get install portaudio19-dev
pip install --force-reinstall pyaudio
```

### WebSocket connection fails

**Check if server is running:**
```python
agent.tool.voice_chat_server(action="status")
```

**Try different port:**
```python
agent.tool.voice_chat_server(action="start", port=8766)
```

**Check firewall settings** - allow connections on port 8765

### AWS credentials error

**Verify credentials:**
```bash
aws sts get-caller-identity
```

**Set credentials:**
```bash
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
```

### Audio is choppy

This is usually due to network latency or audio buffer settings. The tools are optimized for smooth playback with:
- Continuous audio streaming (no gaps)
- Precise scheduling (Web Audio API / sounddevice)
- Minimal blocking I/O

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Review [INSTALL.md](INSTALL.md) for advanced installation options
- Check out the example tools in `/tools` directory
- Explore the web UI at `docs/index.html`

## Support

For issues or questions:
- Review the troubleshooting section above
- Check Strands Agents documentation: https://strandsagents.com
- Open an issue on the repository

Happy voice chatting! 🎤🚀
