# 🎤 Strands Voice Chat

Real-time bidirectional voice conversations with AI agents using Strands Agents SDK with experimental bidirectional streaming support.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## 🌟 Features

### **Speech-to-Speech (Terminal)**
- 🎤 **Direct voice interaction** - Talk to AI through your microphone
- 🔊 **Real-time audio playback** - Hear AI responses instantly
- ⚡ **Auto-interruption** - Speak over the AI (VAD detection)
- 🛠️ **Full tool access** - Use any tool via voice commands
- 🛑 **Voice-activated stop** - Say "stop the conversation" to end
- 🌍 **Multiple voices** - Support for 11 voice IDs across 5 languages
- 🔧 **Customizable** - Custom system prompts and tool filtering

### **Voice Chat Server (Browser)**
- 🌐 **Web-based interface** - Beautiful gradient UI in the browser
- 👥 **Multi-client support** - Multiple users, isolated sessions
- 📊 **Audio visualizer** - Real-time audio level display
- 💬 **Text transcripts** - See what's being said
- 🎨 **Modern design** - Responsive, mobile-friendly interface
- 🔄 **Auto-reconnect** - Robust WebSocket handling
- ⚡ **Instant interruption** - Manual and automatic VAD

### **Model Providers**
- ✅ **Nova Sonic** (AWS Bedrock) - Default, 11 voice IDs
- ✅ **OpenAI Realtime API** - GPT-4o with voice
- ✅ **Gemini Live** - Google's conversational AI

## 📦 Installation

### Quick Install

```bash
./setup.sh
```

Or manually:

```bash
pip install -e .
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions and troubleshooting.

### Requirements

- **Python:** 3.10 or higher
- **Audio:** Microphone and speakers (for speech-to-speech)
- **Browser:** Modern browser (Chrome, Firefox, Safari) for web UI
- **AWS Credentials:** For Nova Sonic provider
- **PortAudio:** For PyAudio (auto-installed on macOS via setup.sh)

## 🚀 Quick Start

### 1. Set Environment Variables

```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"  # Optional
```

### 2. Speech-to-Speech (Terminal)

```python
from strands import Agent

agent = Agent()

# Start voice session
agent.tool.speech_to_speech(action="start", provider="novasonic")

# Speak into your microphone!
# The AI will respond with voice and can use tools
# Say "Can you stop the conversation?" to end
```

### 3. Voice Chat Server (Browser)

```python
from strands import Agent

agent = Agent()

# Start WebSocket server
agent.tool.voice_chat_server(action="start", provider="novasonic")

# Open docs/index.html in your browser
# Click "Connect" and "Start Speaking"
```

## 📖 Detailed Usage

### Speech-to-Speech Tool

#### Basic Usage

```python
# Start with default settings (Nova Sonic, US English)
agent.tool.speech_to_speech(action="start")

# Check status
agent.tool.speech_to_speech(action="status")

# Stop session
agent.tool.speech_to_speech(action="stop")
```

#### Custom Voice ID

```python
# Italian voice
agent.tool.speech_to_speech(
    action="start",
    provider="novasonic",
    model_settings={"voice_id": "beatrice"}
)

# German voice
agent.tool.speech_to_speech(
    action="start",
    provider="novasonic",
    model_settings={"voice_id": "lennart"}
)
```

#### Custom System Prompt

```python
agent.tool.speech_to_speech(
    action="start",
    provider="novasonic",
    system_prompt="You are a helpful language tutor. Speak slowly and clearly."
)
```

#### Specific Tools Only

```python
# Only allow calculator and time tools
agent.tool.speech_to_speech(
    action="start",
    provider="novasonic",
    tools=["calculator", "current_time", "weather"]
)
```

#### OpenAI Realtime API

```python
agent.tool.speech_to_speech(
    action="start",
    provider="openai",
    model_settings={
        "model": "gpt-4o-realtime-preview",
        "session": {
            "voice": "shimmer",
            "turn_detection": {
                "threshold": 0.6,
                "silence_duration_ms": 800
            }
        }
    }
)
```

#### Gemini Live

```python
agent.tool.speech_to_speech(
    action="start",
    provider="gemini_live",
    model_settings={
        "model_id": "models/gemini-2.0-flash-live-preview-04-09",
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

### Voice Chat Server Tool

#### Basic Usage

```python
# Start on default port (8765)
agent.tool.voice_chat_server(action="start")

# Start on custom port
agent.tool.voice_chat_server(action="start", port=8766)

# Check status
agent.tool.voice_chat_server(action="status")

# Stop server
agent.tool.voice_chat_server(action="stop")
```

#### Custom Configuration

```python
# With custom voice and system prompt
agent.tool.voice_chat_server(
    action="start",
    provider="novasonic",
    port=8765,
    model_settings={"voice_id": "tiffany"},
    system_prompt="You are a helpful customer support agent."
)
```

#### OpenAI Provider

```python
agent.tool.voice_chat_server(
    action="start",
    provider="openai",
    model_settings={
        "session": {
            "voice": "alloy",
            "instructions": "You are a friendly assistant."
        }
    }
)
```

#### Specific Tools

```python
# Only expose certain tools to voice chat users
agent.tool.voice_chat_server(
    action="start",
    provider="novasonic",
    tools=["calculator", "weather", "current_time"]
)
```

## 🎨 Voice IDs (Nova Sonic)

| Voice ID | Language | Gender | Description |
|----------|----------|--------|-------------|
| `tiffany` | English (US) | Feminine | Warm, professional |
| `matthew` | English (US) | Masculine | Clear, authoritative |
| `amy` | English (GB) | Feminine | British accent |
| `ambre` | French | Feminine | Native French |
| `florian` | French | Masculine | Native French |
| `beatrice` | Italian | Feminine | Native Italian |
| `lorenzo` | Italian | Masculine | Native Italian |
| `greta` | German | Feminine | Native German |
| `lennart` | German | Masculine | Native German |
| `lupe` | Spanish | Feminine | Native Spanish |
| `carlos` | Spanish | Masculine | Native Spanish |

## 🏗️ Architecture

### Speech-to-Speech Flow

```
Microphone (16kHz PCM)
    ↓
PyAudio Input Stream
    ↓
BidirectionalAgent (Nova Sonic / OpenAI / Gemini)
    ↓
Tool Execution (inherited from parent agent)
    ↓
Audio Output Queue (24kHz PCM)
    ↓
PyAudio Output Stream
    ↓
Speakers
```

### Voice Chat Server Flow

```
Browser Microphone (16kHz)
    ↓
WebSocket Client (JSON messages)
    ↓
WebSocket Server (port 8765)
    ↓
BidirectionalAgent (per-client isolation)
    ↓
Tool Execution (inherited from parent agent)
    ↓
WebSocket Server (JSON messages)
    ↓
Browser Audio Playback (Web Audio API)
    ↓
Speakers
```

### Key Components

**BidirectionalAgent:**
- Manages bidirectional streaming with AI models
- Handles tool execution
- Processes interruptions via VAD
- Maintains conversation context

**Audio Processing:**
- **Input:** 16kHz, mono, PCM 16-bit
- **Output:** 24kHz (Nova Sonic), 24kHz (OpenAI), variable (Gemini)
- **Latency:** ~200-500ms end-to-end
- **Chunk Size:** 1024 bytes (speech-to-speech), 4096 bytes (browser)

**Tool Inheritance:**
- Automatically inherits ALL tools from parent agent
- Filters out recursive tools (speech_to_speech, voice_chat_server)
- Adds stop tools (stop_speech, stop_voice_chat)
- Full tool execution during voice conversations

## 🎯 Use Cases

### 1. Voice-Controlled Data Analysis

```python
from strands import Agent
from strands_tools import calculator

agent = Agent(tools=[calculator])

# Voice session inherits calculator tool
agent.tool.speech_to_speech(action="start")

# Say: "What's the average of 45, 67, 89, and 23?"
# AI uses calculator tool and responds with voice!
```

### 2. Multi-User Customer Support

```python
agent = Agent(tools=[retrieve, http_request])

# Start server for multiple customers
agent.tool.voice_chat_server(
    action="start",
    system_prompt="You are a helpful customer support agent.",
    tools=["retrieve", "http_request"]
)

# Each browser client gets isolated session
# Open docs/index.html in multiple browsers
```

### 3. Language Learning

```python
agent.tool.speech_to_speech(
    action="start",
    provider="novasonic",
    model_settings={"voice_id": "ambre"},  # French voice
    system_prompt="You are a French language tutor. Speak slowly and correct pronunciation."
)
```

### 4. Voice-Activated Home Assistant

```python
from strands_tools import http_request

agent = Agent(tools=[http_request])

agent.tool.speech_to_speech(
    action="start",
    system_prompt="You control smart home devices. Use http_request tool to control lights, temperature, etc.",
    tools=["http_request", "current_time"]
)

# Say: "Turn on the living room lights"
# AI uses http_request to control smart home API
```

## 🔧 Configuration Reference

### speech_to_speech Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | str | - | Action: start, stop, status |
| `provider` | str | novasonic | Model provider |
| `system_prompt` | str | None | Custom system prompt |
| `tools` | List[str] | None | Specific tools (inherits all if None) |
| `model_settings` | Dict | None | Provider-specific config |
| `session_id` | str | None | Custom session ID |
| `audio_input` | bool | True | Enable microphone |
| `audio_output` | bool | True | Enable speakers |

### voice_chat_server Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | str | - | Action: start, stop, status |
| `port` | int | 8765 | WebSocket server port |
| `provider` | str | novasonic | Model provider |
| `system_prompt` | str | None | Custom system prompt |
| `tools` | List[str] | None | Specific tools (inherits all if None) |
| `model_settings` | Dict | None | Provider-specific config |

### Model Settings Examples

**Nova Sonic:**
```python
model_settings={
    "region": "us-west-2",
    "voice_id": "tiffany",
    "model_id": "amazon.nova-sonic-v1:0"
}
```

**OpenAI:**
```python
model_settings={
    "model": "gpt-4o-realtime-preview",
    "api_key": "sk-...",  # Optional if OPENAI_API_KEY set
    "session": {
        "voice": "shimmer",
        "instructions": "Custom instructions",
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "silence_duration_ms": 500
        }
    }
}
```

**Gemini:**
```python
model_settings={
    "model_id": "models/gemini-2.0-flash-live-preview-04-09",
    "api_key": "...",  # Optional
    "params": {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": "Kore"}
            }
        }
    }
}
```

## 🎤 Voice Commands Examples

Try these commands to test tool usage:

- **Math:** "What is 25 times 17?"
- **Time:** "What's the current time?"
- **AWS:** "List my S3 buckets"
- **Files:** "Read the README file"
- **Weather:** "What's the weather in San Francisco?"
- **Stop:** "Can you stop the conversation?"

## 🐛 Troubleshooting

### PyAudio Installation Issues

**macOS:**
```bash
brew install portaudio
pip install --force-reinstall pyaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev
pip install --force-reinstall pyaudio
```

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

### No Audio Input/Output

1. **Check PyAudio:**
   ```python
   import pyaudio
   p = pyaudio.PyAudio()
   print("Input devices:", p.get_device_count())
   ```

2. **Check permissions:** Ensure microphone access is granted
3. **Test audio:** `python -m pyaudio` to test audio devices

### WebSocket Connection Fails

1. **Check server status:**
   ```python
   agent.tool.voice_chat_server(action="status")
   ```

2. **Check firewall:** Allow connections on port 8765
3. **Try different port:** `agent.tool.voice_chat_server(action="start", port=8766)`

### AWS Credentials Error

```bash
# Verify credentials
aws sts get-caller-identity

# Set credentials
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
```

### Audio is Choppy

- **Issue:** Network latency or buffer underrun
- **Solution:** Tools are optimized for smooth playback with:
  - Continuous audio streaming (no gaps)
  - Precise scheduling (Web Audio API / sounddevice)
  - Minimal blocking I/O
- **Check:** Network connection to AWS/OpenAI/Google

### Tools Not Available in Voice Session

- **Issue:** Tools not inherited properly
- **Solution:** Check parent agent has tools:
  ```python
  print(agent.tool_registry.registry.keys())
  ```

## 📚 API Reference

### speech_to_speech(action, **kwargs)

**Actions:**
- `start` - Start new voice session
- `stop` - Stop session(s)
- `status` - Get session status

**Returns:** Status message string

### voice_chat_server(action, **kwargs)

**Actions:**
- `start` - Start WebSocket server
- `stop` - Stop server
- `status` - Get server status

**Returns:** Status message string

### stop_speech(session_id=None)

**Tool for voice-activated stopping** (available in speech-to-speech sessions)

**Args:**
- `session_id` (optional) - Specific session to stop

**Returns:** Confirmation message

### stop_voice_chat(client_id=None)

**Tool for voice-activated stopping** (available in voice chat sessions)

**Args:**
- `client_id` (optional) - Specific client to disconnect

**Returns:** Confirmation message

## 🧪 Testing

```python
# Test speech-to-speech
from strands import Agent
agent = Agent()
agent.tool.speech_to_speech(action="start")
# Speak: "What is 2 plus 2?"
# Expected: AI uses calculator and responds with voice

# Test voice chat server
agent.tool.voice_chat_server(action="start")
# Open docs/index.html in browser
# Click Connect, Start Speaking
# Expected: Web UI shows connection, audio works
```

## 📁 Project Structure

```
strands-web-voice-chat/
├── tools/
│   ├── speech_to_speech.py      # Terminal voice chat tool
│   └── voice_chat_server.py     # WebSocket voice chat server
├── docs/
│   └── index.html                # Web UI for voice chat
├── sdk-python/                   # Forked Strands SDK (submodule)
├── pyproject.toml                # Package configuration
├── requirements.txt              # Pip requirements
├── setup.sh                      # Automated installation script
├── README.md                     # This file
├── INSTALL.md                    # Installation guide
└── QUICKSTART.md                 # Quick start guide
```

## 🤝 Contributing

This project uses a forked version of Strands Agents SDK with experimental bidirectional streaming support:
- **Fork:** https://github.com/mehtarac/sdk-python.git
- **Original:** https://github.com/strands-agents/sdk-python

## 📄 License

Apache 2.0 License - See LICENSE file for details

## 🙏 Acknowledgments

- **Strands Agents SDK** - https://strandsagents.com
- **AWS Nova Sonic** - Amazon Bedrock bidirectional streaming
- **OpenAI Realtime API** - GPT-4o voice capabilities
- **Google Gemini Live** - Conversational AI

## 📞 Support

- **Documentation:** [INSTALL.md](INSTALL.md), [QUICKSTART.md](QUICKSTART.md)
- **Strands Agents:** https://strandsagents.com
- **Issues:** Open an issue on the repository

---

**Built with ❤️ using Strands Agents SDK**

Start talking to AI today! 🎤🚀
