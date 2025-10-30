"""Real-time speech-to-speech bidirectional streaming tool for Strands Agents.

Provides background speech-to-speech conversation capability using Strands
experimental bidirectional streaming with full model provider support, tool
inheritance, and comprehensive configuration options.

This tool creates isolated bidirectional agent sessions that run in background
threads, enabling real-time voice conversations with AI models while the parent
agent remains responsive.

Key Features:
- **Background Execution:** Runs in separate thread - parent agent stays responsive
- **Real-Time Audio:** Microphone input and speaker output with pyaudio
- **Tool Inheritance:** Automatically inherits ALL tools from parent agent
- **Multiple Providers:** Nova Sonic, OpenAI Realtime API, Gemini Live
- **Full Configuration:** Per-provider custom settings and parameters
- **Voice Activation:** Voice-activated stopping via stop_speech tool
- **Auto-Interruption:** Built-in VAD for natural conversation flow

Supported Providers:
-------------------
1. **Nova Sonic (AWS Bedrock):**
   - Region: us-east-1, us-west-2, etc.
   - Model: amazon.nova-sonic-v1:0
   - Requires AWS credentials

2. **OpenAI Realtime API:**
   - Models: gpt-realtime, gpt-4o-realtime-preview
   - Requires OPENAI_API_KEY
   - Custom session config support

3. **Gemini Live:**
   - Model: models/gemini-2.0-flash-live-preview-04-09
   - Requires Google AI API key
   - Live config customization

Usage Examples:
--------------
```python
from strands import Agent

agent = Agent()

# Basic usage with Nova Sonic
agent.tool.speech_to_speech(action="start", provider="novasonic")

# With custom model settings
agent.tool.speech_to_speech(
    action="start",
    provider="novasonic",
    model_settings={
        "region": "us-west-2",
        "model_id": "amazon.nova-sonic-v1:0"
    }
)

# OpenAI with custom configuration
agent.tool.speech_to_speech(
    action="start",
    provider="openai",
    model_settings={
        "model": "gpt-4o-realtime-preview",
        "api_key": "sk-...",
        "session": {
            "voice": "shimmer",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.6,
                "silence_duration_ms": 800
            }
        }
    }
)

# Gemini Live with custom system prompt
agent.tool.speech_to_speech(
    action="start",
    provider="gemini_live",
    system_prompt="You are a friendly language tutor. Speak slowly and clearly.",
    model_settings={
        "model_id": "models/gemini-2.0-flash-live-preview-04-09",
        "params": {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
            }
        }
    }
)

# Check session status
agent.tool.speech_to_speech(action="status")

# Stop specific session
agent.tool.speech_to_speech(action="stop", session_id="speech_20251029_150000")
```

See the speech_to_speech function docstring for complete parameter documentation.
"""

import asyncio
import logging
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyaudio

from strands import tool


from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent
from strands.experimental.bidirectional_streaming.models.gemini_live import GeminiLiveBidirectionalModel
from strands.experimental.bidirectional_streaming.models.novasonic import NovaSonicBidirectionalModel
from strands.experimental.bidirectional_streaming.models.openai import OpenAIRealtimeBidirectionalModel

logger = logging.getLogger(__name__)

# Audio configuration (matching test_bidi_novasonic.py)
INPUT_SAMPLE_RATE = 16000
INPUT_CHANNELS = 1
INPUT_CHUNK_SIZE = 1024  # 1024-byte buffers for responsive audio
OUTPUT_SAMPLE_RATE = 24000
OUTPUT_CHUNK_SIZE = 1024

# Global session tracking
_active_sessions = {}
_session_lock = threading.Lock()


class SpeechSession:
    """Manages a speech-to-speech conversation session with full lifecycle management."""
    
    def __init__(
        self,
        session_id: str,
        agent: BidirectionalAgent,
        audio_input_enabled: bool = True,
        audio_output_enabled: bool = True
    ):
        """Initialize speech session.
        
        Args:
            session_id: Unique session identifier
            agent: BidirectionalAgent instance
            audio_input_enabled: Whether to capture microphone input
            audio_output_enabled: Whether to play audio output
        """
        self.session_id = session_id
        self.agent = agent
        self.audio_input_enabled = audio_input_enabled
        self.audio_output_enabled = audio_output_enabled
        
        self.active = False
        self.thread = None
        self.loop = None
        self.interrupted = False
        
        # Use asyncio.Queue() - critical for async operations!
        self.audio_input_queue = None  # Created in async context
        self.audio_output_queue = None  # Created in async context
    
    def start(self) -> None:
        """Start the speech session in background thread."""
        if self.active:
            raise ValueError("Session already active")
        
        self.active = True
        self.thread = threading.Thread(target=self._run_session, daemon=True)
        self.thread.start()
    
    def stop(self) -> None:
        """Stop the speech session and cleanup resources."""
        if not self.active:
            return
        
        self.active = False
        
        if self.thread:
            self.thread.join(timeout=5.0)
    
    def _run_session(self) -> None:
        """Main session runner in background thread."""
        try:
            # Create event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Run the async session
            self.loop.run_until_complete(self._async_session())
        except Exception as e:
            error_msg = f"Session error: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
        finally:
            if self.loop:
                self.loop.close()
    
    async def _async_session(self) -> None:
        """Async session management - matches test_bidi_novasonic.py pattern."""
        # Create asyncio queues in async context
        self.audio_input_queue = asyncio.Queue()
        self.audio_output_queue = asyncio.Queue()
        
        try:
            # Start bidirectional agent
            await self.agent.start()
            
            # Start concurrent tasks (matching test pattern)
            await asyncio.gather(
                self._play_audio(),
                self._record_audio(),
                self._receive_from_agent(),
                self._send_to_agent(),
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Async session error: {e}\n{traceback.format_exc()}")
        finally:
            try:
                await self.agent.end()
            except Exception as e:
                logger.error(f"Error ending agent: {e}")
    
    async def _play_audio(self) -> None:
        """Play audio output - matches test pattern."""
        if not self.audio_output_enabled:
            return
        
        try:
            audio = pyaudio.PyAudio()
            speaker = audio.open(
                channels=1,
                format=pyaudio.paInt16,
                output=True,
                rate=OUTPUT_SAMPLE_RATE,
                frames_per_buffer=OUTPUT_CHUNK_SIZE,
            )
            
            try:
                while self.active:
                    try:
                        # Check for interruption first
                        if self.interrupted:
                            # Clear entire audio queue immediately
                            while not self.audio_output_queue.empty():
                                try:
                                    self.audio_output_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                            
                            self.interrupted = False
                            await asyncio.sleep(0.05)
                            continue
                        
                        # Get next audio data
                        audio_data = await asyncio.wait_for(
                            self.audio_output_queue.get(),
                            timeout=0.1
                        )
                        
                        if audio_data and self.active:
                            # Write in chunks for responsiveness
                            chunk_size = OUTPUT_CHUNK_SIZE
                            for i in range(0, len(audio_data), chunk_size):
                                # Check for interruption before each chunk
                                if self.interrupted or not self.active:
                                    break
                                
                                end = min(i + chunk_size, len(audio_data))
                                chunk = audio_data[i:end]
                                speaker.write(chunk)
                                await asyncio.sleep(0.001)
                        
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.01)
                    except asyncio.CancelledError:
                        break
            
            finally:
                speaker.close()
                audio.terminate()
        
        except ImportError:
            logger.error("pyaudio not installed. Install with: pip install pyaudio")
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
    
    async def _record_audio(self) -> None:
        """Record audio input - matches test pattern."""
        if not self.audio_input_enabled:
            return
        
        try:
            audio = pyaudio.PyAudio()
            microphone = audio.open(
                channels=1,
                format=pyaudio.paInt16,
                frames_per_buffer=INPUT_CHUNK_SIZE,
                input=True,
                rate=INPUT_SAMPLE_RATE,
            )
            
            try:
                while self.active:
                    try:
                        audio_bytes = microphone.read(
                            INPUT_CHUNK_SIZE,
                            exception_on_overflow=False
                        )
                        self.audio_input_queue.put_nowait(audio_bytes)
                        await asyncio.sleep(0.01)
                    except asyncio.CancelledError:
                        break
            
            finally:
                microphone.close()
                audio.terminate()
        
        except ImportError:
            logger.error("pyaudio not installed. Install with: pip install pyaudio")
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
    
    async def _receive_from_agent(self) -> None:
        """Receive events from agent - matches test pattern."""
        try:
            async for event in self.agent.receive():
                if not self.active:
                    break
                
                # Handle audio output
                if "audioOutput" in event:
                    if not self.interrupted:
                        self.audio_output_queue.put_nowait(
                            event["audioOutput"]["audioData"]
                        )
                
                # Handle interruption events
                elif "interruptionDetected" in event:
                    self.interrupted = True
                elif "interrupted" in event:
                    self.interrupted = True
                
                # Handle text output
                elif "textOutput" in event:
                    text_content = event["textOutput"].get("content", "")
                    role = event["textOutput"].get("role", "unknown")
                    
                    # Check for text-based interruption patterns
                    if '{ "interrupted" : true }' in text_content:
                        self.interrupted = True
                    elif "interrupted" in text_content.lower():
                        self.interrupted = True
                    
                    if role.upper() == "USER":
                        print(f"[USER] {text_content}")
                    elif role.upper() == "ASSISTANT":
                        print(f"[ASSISTANT] {text_content}")
                
                # Handle tool usage
                elif "toolUse" in event:
                    tool_use_data = event["toolUse"]
                    tool_name = tool_use_data["name"]
                    print(f"[TOOL] Calling: {tool_name}")
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive error: {e}")
    
    async def _send_to_agent(self) -> None:
        """Send audio to agent - matches test pattern."""
        try:
            while self.active:
                try:
                    audio_bytes = self.audio_input_queue.get_nowait()
                    audio_event = {
                        "audioData": audio_bytes,
                        "format": "pcm",
                        "sampleRate": INPUT_SAMPLE_RATE,
                        "channels": 1
                    }
                    await self.agent.send(audio_event)
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    break
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Send error: {e}")


@tool
def stop_speech(session_id: Optional[str] = None) -> str:
    """Stop the active speech-to-speech session.
    
    This tool allows the AI agent to stop the speech session when the user
    requests it (e.g., "please stop the conversation", "end this session").
    
    Args:
        session_id: Optional specific session to stop. If not provided, stops all active sessions.
    
    Returns:
        str: Confirmation message
    
    Example:
        User: "Can you stop the conversation?"
        Agent calls: stop_speech()
        Result: Session stopped
    """
    return _stop_speech_session(session_id)


@tool
def speech_to_speech(
    action: str,
    provider: str = "novasonic",
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None,
    audio_input: bool = True,
    audio_output: bool = True,
    model_settings: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None,
    agent: Optional[Any] = None
) -> str:
    """Start, stop, or manage speech-to-speech conversations with comprehensive configuration.
    
    Creates a background bidirectional streaming session for real-time voice
    conversations with AI. Supports full model configuration, tool inheritance,
    and multiple model providers with custom settings.
    
    How It Works:
    ------------
    1. Creates a bidirectional model based on provider and settings
    2. Inherits tools from parent agent (or uses specified tools)
    3. Creates BidirectionalAgent in isolated background thread
    4. Manages real-time audio I/O with pyaudio
    5. Handles events, tool execution, and interruptions
    6. Provides clean lifecycle management (start, status, stop)
    
    Model Provider Configuration:
    ---------------------------
    Each provider supports custom settings through model_settings parameter:
    
    **Nova Sonic (AWS Bedrock):**
    ```python
    model_settings={
        "region": "us-east-1",  # AWS region
        "model_id": "amazon.nova-sonic-v1:0"  # Model identifier
    }
    ```
    Requires: AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    
    **OpenAI Realtime API:**
    ```python
    model_settings={
        "model": "gpt-4o-realtime-preview",  # Model name
        "api_key": "sk-...",  # Optional if OPENAI_API_KEY set
        "organization": "org-...",  # Optional
        "project": "proj_...",  # Optional
        "session": {  # Session configuration
            "voice": "alloy",  # Voice selection
            "instructions": "Custom instructions",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "silence_duration_ms": 500
            }
        }
    }
    ```
    Requires: OPENAI_API_KEY environment variable or api_key in settings
    
    **Gemini Live:**
    ```python
    model_settings={
        "model_id": "models/gemini-2.0-flash-live-preview-04-09",
        "api_key": "...",  # Optional if default credentials set
        "params": {  # Live config parameters
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": "Kore"}
                }
            }
        }
    }
    ```
    Requires: Google AI API key or application default credentials
    
    Common Use Cases:
    ---------------
    - **Voice assistants:** Hands-free AI interaction
    - **Language learning:** Real-time conversation practice
    - **Accessibility:** Voice-controlled tool execution
    - **Customer support:** Voice-based query handling
    - **Creative applications:** Voice-driven content creation
    
    Args:
        action: Action to perform:
            - "start": Start new speech session
            - "stop": Stop session(s)
            - "status": Get session status
        provider: Model provider to use:
            - "novasonic": AWS Bedrock Nova Sonic
            - "openai": OpenAI Realtime API
            - "gemini_live": Google Gemini Live
        system_prompt: Custom system prompt for the agent. If not provided,
            uses default prompt that encourages tool usage.
        session_id: Session identifier:
            - For "start": Custom ID (auto-generated if not provided)
            - For "stop": Specific session to stop (stops all if not provided)
            - For "status": Not used
        audio_input: Enable microphone input (default: True)
        audio_output: Enable speaker output (default: True)
        model_settings: Provider-specific configuration dictionary.
            See "Model Provider Configuration" section for details.
        tools: List of tool names to make available. If not provided,
            inherits ALL tools from parent agent.
            Example: ["calculator", "weather", "file_read"]
        agent: Parent agent (automatically passed by Strands framework)
    
    Returns:
        str: Status message with session details or error information
    
    Examples:
    --------
    # Basic usage with Nova Sonic
    speech_to_speech(action="start", provider="novasonic")
    
    # Nova Sonic with custom region
    speech_to_speech(
        action="start",
        provider="novasonic",
        model_settings={"region": "us-west-2"}
    )
    
    # OpenAI with custom voice and VAD settings
    speech_to_speech(
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
    
    # Gemini Live with custom voice
    speech_to_speech(
        action="start",
        provider="gemini_live",
        system_prompt="You are a helpful language tutor.",
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
    
    # With specific tools
    speech_to_speech(
        action="start",
        provider="novasonic",
        tools=["calculator", "current_time", "weather"]
    )
    
    # Check status
    speech_to_speech(action="status")
    
    # Stop specific session
    speech_to_speech(action="stop", session_id="speech_20251029_150000")
    
    # Stop all sessions
    speech_to_speech(action="stop")
    
    Environment Variables:
    --------------------
    - **AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY**: For Nova Sonic
    - **AWS_REGION**: Default AWS region (optional)
    - **OPENAI_API_KEY**: For OpenAI Realtime API
    - **GOOGLE_APPLICATION_CREDENTIALS**: For Gemini Live
    
    Notes:
        - Requires pyaudio: `pip install pyaudio`
        - Audio runs in background thread - parent agent stays responsive
        - Tools are automatically inherited from parent agent
        - stop_speech tool is always included for voice-activated stopping
        - Session continues until explicitly stopped or interrupted
        - Supports natural interruption through Voice Activity Detection (VAD)
        - All providers support real-time tool execution during conversation
    """
    
    if action == "start":
        return _start_speech_session(
            provider, system_prompt, session_id, audio_input, audio_output,
            model_settings, tools, agent
        )
    elif action == "stop":
        return _stop_speech_session(session_id)
    elif action == "status":
        return _get_session_status()
    else:
        return f"Unknown action: {action}"


def _start_speech_session(
    provider: str,
    system_prompt: Optional[str],
    session_id: Optional[str],
    audio_input: bool,
    audio_output: bool,
    model_settings: Optional[Dict[str, Any]],
    tool_names: Optional[List[str]],
    parent_agent: Optional[Any]
) -> str:
    """Start a speech-to-speech session with full configuration support."""
    try:
        # Generate session ID if not provided
        if not session_id:
            session_id = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if session already exists
        with _session_lock:
            if session_id in _active_sessions:
                return f"❌ Session already exists: {session_id}"
        
        # Create model based on provider with custom settings
        model_settings = model_settings or {}
        model_info = f"{provider}"
        
        try:
            if provider == "novasonic":
                model = NovaSonicBidirectionalModel(**model_settings)
                model_info = f"Nova Sonic ({model_settings.get('region', 'us-east-1')})"
            elif provider == "openai":
                model = OpenAIRealtimeBidirectionalModel(**model_settings)
                model_info = f"OpenAI Realtime ({model_settings.get('model', 'gpt-realtime')})"
            elif provider == "gemini_live":
                model = GeminiLiveBidirectionalModel(**model_settings)
                model_info = f"Gemini Live ({model_settings.get('model_id', 'gemini-2.0-flash-live')})"
            else:
                return f"❌ Unknown provider: {provider}. Supported: novasonic, openai, gemini_live"
        except Exception as e:
            return f"❌ Error creating {provider} model: {e}\n\nCheck your configuration and credentials."
        
        # Get parent agent's tools
        tools = []
        inherited_count = 0
        
        # Always include stop_speech tool (defined in this file)
        tools.append(stop_speech)
        inherited_count += 1
        
        if parent_agent and hasattr(parent_agent, 'tool_registry'):
            try:
                # Get all tool functions from parent agent's registry
                all_tools_config = parent_agent.tool_registry.get_all_tools_config()
                
                # If specific tools requested, filter; otherwise inherit all
                if tool_names:
                    # User specified tool names - only include those
                    for tool_name in tool_names:
                        if tool_name not in ['speech_to_speech', 'stop_speech']:
                            tool_func = parent_agent.tool_registry.registry.get(tool_name)
                            if tool_func:
                                tools.append(tool_func)
                                inherited_count += 1
                            else:
                                logger.warning(f"Tool '{tool_name}' not found in parent agent's registry")
                else:
                    # No specific tools - inherit all except excluded
                    for tool_name, tool_config in all_tools_config.items():
                        if tool_name not in ['speech_to_speech', 'stop_speech']:
                            tool_func = parent_agent.tool_registry.registry.get(tool_name)
                            if tool_func:
                                tools.append(tool_func)
                                inherited_count += 1
                
            except Exception as e:
                logger.warning(f"Could not inherit tools from parent agent: {e}")
        
        # Use default system prompt if not provided
        if not system_prompt:
            system_prompt = """You are a helpful AI assistant with access to powerful tools.

IMPORTANT: When a user asks you to perform a task that matches one of your available tools, you MUST use the appropriate tool. Do not just describe what you would do - actually execute the tool.

Examples:
- For math questions → Use calculator tool
- For time questions → Use current_time tool
- For AWS operations → Use use_aws tool
- For file operations → Use editor or file_read tools
- To stop the conversation → Use stop_speech tool

Always prefer using tools over generating text responses when tools are available. Keep your voice responses brief and natural."""
        
        # Create bidirectional agent with inherited tools
        bidi_agent = BidirectionalAgent(
            model=model,
            tools=tools,
            system_prompt=system_prompt
        )
        
        # Create and start session
        session = SpeechSession(
            session_id=session_id,
            agent=bidi_agent,
            audio_input_enabled=audio_input,
            audio_output_enabled=audio_output
        )
        
        session.start()
        
        # Register session
        with _session_lock:
            _active_sessions[session_id] = session
        
        # Build settings summary
        settings_summary = ""
        if model_settings:
            settings_lines = []
            for key, value in model_settings.items():
                if key not in ['api_key', 'secret']:  # Hide sensitive data
                    settings_lines.append(f"  - {key}: {value}")
            if settings_lines:
                settings_summary = "\n**Model Settings:**\n" + "\n".join(settings_lines)
        
        return f"""✅ Speech session started!

**Session ID:** {session_id}
**Provider:** {model_info}
**Audio Input:** {'Enabled' if audio_input else 'Disabled'}
**Audio Output:** {'Enabled' if audio_output else 'Disabled'}
**Tools:** {inherited_count} tools available (including stop_speech){settings_summary}

The session is running in the background. Speak into your microphone to interact!

**To stop the conversation, just say:** "Can you stop the conversation?" or "Please end this session"

**Commands:**
- Check status: speech_to_speech(action="status")
- Stop session: speech_to_speech(action="stop", session_id="{session_id}")
"""
    
    except Exception as e:
        logger.error(f"Error starting speech session: {e}\n{traceback.format_exc()}")
        return f"❌ Error starting session: {e}\n\nCheck logs for details."


def _stop_speech_session(session_id: Optional[str]) -> str:
    """Stop a speech session."""
    with _session_lock:
        if not session_id:
            if not _active_sessions:
                return "❌ No active sessions"
            # Stop all sessions
            session_ids = list(_active_sessions.keys())
            for sid in session_ids:
                _active_sessions[sid].stop()
                del _active_sessions[sid]
            return f"✅ Stopped {len(session_ids)} session(s)"
        
        if session_id not in _active_sessions:
            return f"❌ Session not found: {session_id}"
        
        session = _active_sessions[session_id]
        session.stop()
        del _active_sessions[session_id]
        
        return f"✅ Session stopped: {session_id}"


def _get_session_status() -> str:
    """Get status of all active sessions."""
    with _session_lock:
        if not _active_sessions:
            return "No active speech sessions"
        
        status_lines = ["**Active Speech Sessions:**\n"]
        for session_id, session in _active_sessions.items():
            status_lines.append(
                f"- **{session_id}**\n"
                f"  - Audio Input: {'✅' if session.audio_input_enabled else '❌'}\n"
                f"  - Audio Output: {'✅' if session.audio_output_enabled else '❌'}\n"
                f"  - Active: {'✅' if session.active else '❌'}"
            )
        
        return "\n".join(status_lines)
