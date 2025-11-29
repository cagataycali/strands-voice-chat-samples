"""WebSocket server for browser-based bidirectional voice chat with comprehensive model configuration.

This tool provides a WebSocket server that bridges browser clients to Strands
bidirectional agent for real-time audio streaming and voice interaction. It supports
multiple model providers with full configuration options, tool inheritance, and
comprehensive voice settings.

The server creates isolated bidirectional agent instances for each connected client,
enabling concurrent voice conversations with full tool access and natural interruption
support through Voice Activity Detection (VAD).

Key Features:
- **Multi-Client Support:** Each browser gets its own isolated agent instance
- **Real-Time Audio:** Continuous bidirectional audio streaming (16kHz → 24kHz)
- **Tool Inheritance:** Automatically inherits ALL tools from parent agent
- **Multiple Providers:** Nova Sonic, OpenAI Realtime API, Gemini Live
- **Full Configuration:** Per-provider custom settings and parameters
- **Voice Selection:** Support for multiple voice IDs (Nova Sonic)
- **Voice-Activated Stop:** Voice-activated stopping via stop_voice_chat tool
- **Auto-Interruption:** Built-in VAD for natural conversation flow
- **WebSocket Protocol:** Standard WebSocket with JSON messaging

Supported Providers:
-------------------
1. **Nova Sonic (AWS Bedrock):**
   - Region: us-east-1, us-west-2, etc.
   - Model: amazon.nova-sonic-v1:0
   - Voice IDs: tiffany, matthew (US), amy (GB), ambre, florian (FR),
                beatrice, lorenzo (IT), greta, lennart (DE), lupe, carlos (ES)
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
agent.tool.voice_chat_server(action="start", provider="novasonic")

# With custom model settings
agent.tool.voice_chat_server(
    action="start",
    provider="novasonic",
    model_settings={
        "region": "us-west-2",
        "model_id": "amazon.nova-sonic-v1:0",
        "voice_id": "beatrice"  # Italian voice
    }
)

# OpenAI with custom configuration
agent.tool.voice_chat_server(
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
agent.tool.voice_chat_server(
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

# Check server status
agent.tool.voice_chat_server(action="status")

# Stop server
agent.tool.voice_chat_server(action="stop", port=8765)
```

See the voice_chat_server function docstring for complete parameter documentation.
"""

import asyncio
import base64
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from strands import tool

from strands.experimental.bidi import (
    BidiAgent,
    BidiGeminiLiveModel,
    BidiNovaSonicModel,
    BidiOpenAIRealtimeModel,
    BidiAudioIO,
)

logger = logging.getLogger(__name__)

# Global server tracking
_active_servers = {}
_server_threads = {}


@tool
def stop_voice_chat(client_id: Optional[str] = None) -> str:
    """Stop the active voice chat connection.

    This tool allows the AI agent to stop the voice chat session when the user
    requests it (e.g., "please end the conversation", "disconnect me").

    Args:
        client_id: Optional specific client to disconnect. If not provided,
                   stops all active clients.

    Returns:
        str: Confirmation message

    Example:
        User: "Can you end this conversation?"
        Agent calls: stop_voice_chat()
        Result: Client disconnected
    """
    import asyncio

    # Find all servers and their clients
    clients_to_stop = []

    if client_id:
        # Stop specific client
        for port, context in _active_servers.items():
            if client_id in context["clients"]:
                clients_to_stop.append((context, client_id))
    else:
        # Stop all clients across all servers
        for port, context in _active_servers.items():
            for cid in list(context["clients"].keys()):
                clients_to_stop.append((context, cid))

    if not clients_to_stop:
        return "❌ No active voice chat sessions found"

    # Mark clients for disconnection and close WebSocket connections
    for context, cid in clients_to_stop:
        if cid in context["clients"]:
            client_ctx = context["clients"][cid]
            client_ctx["active"] = False

            # Close WebSocket connection
            websocket = client_ctx.get("websocket")
            loop = context.get("loop")

            if websocket and loop:
                try:
                    # Schedule close on the server's event loop
                    asyncio.run_coroutine_threadsafe(websocket.close(), loop)
                except Exception as e:
                    logger.debug(f"Error closing WebSocket for {cid}: {e}")

    count = len(clients_to_stop)
    return f"✅ Ending voice chat session... Goodbye! ({count} client{'s' if count != 1 else ''} disconnected)"


@tool
def voice_chat_server(
    action: str,
    port: int = 8765,
    provider: str = "novasonic",
    system_prompt: Optional[str] = None,
    model_settings: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None,
    agent: Optional[Any] = None,
) -> str:
    """WebSocket server for browser-based bidirectional voice chat with full configuration support.

    Creates a WebSocket server that bridges browser clients to Strands bidirectional
    agent for real-time audio streaming and voice interaction. Supports multiple
    model providers with custom configuration.

    How It Works:
    ------------
    1. Creates a WebSocket server on specified port
    2. For each browser client connection:
       - Creates isolated BidirectionalAgent instance
       - Inherits tools from parent agent (or uses specified tools)
       - Establishes bidirectional audio streaming
    3. Handles concurrent tasks:
       - Browser → Server → Agent (audio input)
       - Agent → Server → Browser (audio output)
    4. Manages interruption, tool execution, and clean disconnection
    5. Supports multiple simultaneous client connections

    Model Provider Configuration:
    ---------------------------
    Each provider supports custom settings through model_settings parameter:

    **Nova Sonic (AWS Bedrock):**
    ```python
    model_settings={
        "region": "us-east-1",  # AWS region
        "model_id": "amazon.nova-sonic-v1:0",  # Model identifier
        "voice_id": "matthew"  # Voice selection (optional)
    }
    ```
    Available voices: tiffany, matthew (US), amy (GB), ambre, florian (FR),
                      beatrice, lorenzo (IT), greta, lennart (DE), lupe, carlos (ES)
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
    - **Customer support:** Browser-based voice assistance
    - **Language learning:** Real-time conversation practice with multiple students
    - **User-facing applications:** Voice interface for web apps
    - **Multi-user scenarios:** Concurrent voice conversations with isolated contexts
    - **Creative applications:** Voice-driven interactive experiences

    Args:
        action: Action to perform:
            - "start": Start new WebSocket server
            - "stop": Stop server
            - "status": Get server status
        port: WebSocket server port (default: 8765)
        provider: Model provider to use:
            - "novasonic": AWS Bedrock Nova Sonic
            - "openai": OpenAI Realtime API
            - "gemini_live": Google Gemini Live
        system_prompt: Custom system prompt for all clients. If not provided,
            uses default prompt that encourages tool usage.
        model_settings: Provider-specific configuration dictionary.
            See "Model Provider Configuration" section for details.
        tools: List of tool names to make available. If not provided,
            inherits ALL tools from parent agent.
            Example: ["calculator", "weather", "file_read"]
        agent: Parent agent (automatically passed by Strands framework)

    Returns:
        str: Status message with server details or error information

    Examples:
    --------
    # Basic usage with Nova Sonic
    voice_chat_server(action="start", provider="novasonic")

    # Nova Sonic with custom region and voice
    voice_chat_server(
        action="start",
        provider="novasonic",
        model_settings={
            "region": "us-west-2",
            "voice_id": "tiffany"
        }
    )

    # Nova Sonic with Italian voice
    voice_chat_server(
        action="start",
        provider="novasonic",
        model_settings={
            "region": "us-east-1",
            "voice_id": "beatrice"
        }
    )

    # OpenAI with custom voice and VAD settings
    voice_chat_server(
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
    voice_chat_server(
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
    voice_chat_server(
        action="start",
        provider="novasonic",
        tools=["calculator", "current_time", "weather"]
    )

    # Check status
    voice_chat_server(action="status")

    # Stop specific server
    voice_chat_server(action="stop", port=8765)

    Environment Variables:
    --------------------
    - **AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY**: For Nova Sonic
    - **AWS_REGION**: Default AWS region (optional)
    - **OPENAI_API_KEY**: For OpenAI Realtime API
    - **GOOGLE_APPLICATION_CREDENTIALS**: For Gemini Live

    Notes:
        - Requires websockets: `pip install websockets`
        - Server runs in background thread - parent agent stays responsive
        - Tools are automatically inherited from parent agent
        - stop_voice_chat tool is always included for voice-activated stopping
        - Each client gets isolated agent instance with private conversation
        - Session continues until client disconnects
        - Supports natural interruption through Voice Activity Detection (VAD)
        - All providers support real-time tool execution during conversation
    """

    if action == "start":
        return _start_voice_chat_server(
            port, provider, system_prompt, model_settings, tools, agent
        )
    elif action == "stop":
        return _stop_voice_chat_server(port)
    elif action == "status":
        return _get_server_status()
    else:
        return f"Unknown action: {action}"


def _start_voice_chat_server(
    port: int,
    provider: str,
    system_prompt: Optional[str],
    model_settings: Optional[Dict[str, Any]],
    tool_names: Optional[List[str]],
    parent_agent: Optional[Any],
) -> str:
    """Start WebSocket server with full configuration support."""
    try:
        # Check if server already running
        if port in _active_servers:
            return f"❌ Server already running on port {port}"

        # Import websockets
        try:
            import websockets
            from websockets.server import serve
        except ImportError:
            return "❌ websockets package not installed. Install with: pip install websockets"

        # Create model based on provider with custom settings
        model_settings = model_settings or {}
        model_info = f"{provider}"

        # Configure audio I/O based on provider
        # Nova Sonic: 16000/16000 (default)
        # OpenAI/Gemini: 24000/24000 (higher quality)
        from strands.experimental.bidirectional_streaming.io.audio import BidiAudioIO
        
        if provider == "novasonic":
            # Nova Sonic uses 16000 Hz
            audio_io = BidiAudioIO(
                audio_config={
                    "input_sample_rate": 16000,
                    "output_sample_rate": 16000
                }
            )
            browser_sample_rate = 16000
        else:
            # OpenAI and Gemini use 24000 Hz
            audio_io = BidiAudioIO(
                audio_config={
                    "input_sample_rate": 24000,
                    "output_sample_rate": 24000
                }
            )
            browser_sample_rate = 24000

        try:
            if provider == "novasonic":
                model = BidiNovaSonicModel(**model_settings)
                voice_info = (
                    f" (voice: {model_settings['voice_id']})"
                    if "voice_id" in model_settings
                    else ""
                )
                model_info = f"Nova Sonic ({model_settings.get('region', 'us-east-1')}){voice_info}"
            elif provider == "openai":
                model = BidiOpenAIRealtimeModel(**model_settings)
                model_info = (
                    f"OpenAI Realtime ({model_settings.get('model', 'gpt-realtime')})"
                )
            elif provider == "gemini_live":
                model = BidiGeminiLiveModel(**model_settings)
                model_info = f"Gemini Live ({model_settings.get('model_id', 'gemini-2.0-flash-live')})"
            else:
                return f"❌ Unknown provider: {provider}. Supported: novasonic, openai, gemini_live"
        except Exception as e:
            return f"❌ Error creating {provider} model: {e}\n\nCheck your configuration and credentials."

        # Get parent agent's tools
        tools = []
        inherited_count = 0
        tool_list = []

        # Always include stop_voice_chat tool (defined in this file)
        tools.append(stop_voice_chat)
        tool_list.append("stop_voice_chat")
        inherited_count += 1

        if parent_agent and hasattr(parent_agent, "tool_registry"):
            try:
                # Get all tool functions from parent agent's registry
                all_tools_config = parent_agent.tool_registry.get_all_tools_config()

                # If specific tools requested, filter; otherwise inherit all
                if tool_names:
                    # User specified tool names - only include those
                    for tool_name in tool_names:
                        if tool_name not in ["voice_chat_server", "stop_voice_chat"]:
                            tool_func = parent_agent.tool_registry.registry.get(
                                tool_name
                            )
                            if tool_func:
                                tools.append(tool_func)
                                tool_list.append(tool_name)
                                inherited_count += 1
                            else:
                                logger.warning(
                                    f"Tool '{tool_name}' not found in parent agent's registry"
                                )
                else:
                    # No specific tools - inherit all except excluded
                    for tool_name, tool_config in all_tools_config.items():
                        if tool_name not in ["voice_chat_server", "stop_voice_chat"]:
                            tool_func = parent_agent.tool_registry.registry.get(
                                tool_name
                            )
                            if tool_func:
                                tools.append(tool_func)
                                tool_list.append(tool_name)
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
- To stop the conversation → Use stop_voice_chat tool

Always prefer using tools over generating text responses when tools are available. Keep your voice responses brief and natural."""

        # Create server context
        server_context = {
            "port": port,
            "provider": provider,
            "model_info": model_info,
            "model": model,
            "audio_io": audio_io,
            "browser_sample_rate": browser_sample_rate,
            "tools": tools,
            "system_prompt": system_prompt,
            "active": True,
            "clients": {},
            "stop_event": threading.Event(),
        }

        _active_servers[port] = server_context

        # Start server in background thread
        def run_server():
            """Run asyncio server in background thread."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Store loop reference for cleanup
                server_context["loop"] = loop

                async def handle_client(websocket, path):
                    """Handle WebSocket client connection."""
                    client_id = (
                        f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
                    )

                    try:
                        # Create bidirectional agent for this client with audio I/O
                        client_agent = BidiAgent(
                            model=server_context["model"],
                            tools=server_context["tools"],
                            system_prompt=server_context["system_prompt"],
                            audio_io=server_context["audio_io"],
                        )

                        await client_agent.start()

                        # Store client context
                        server_context["clients"][client_id] = {
                            "websocket": websocket,
                            "agent": client_agent,
                            "active": True,
                            "interrupted": False,
                            "tasks": [],
                        }

                        # Send connection success with sample rate info
                        connection_msg = json.dumps(
                            {
                                "type": "connected",
                                "client_id": client_id,
                                "provider": server_context["provider"],
                                "sample_rate": server_context["browser_sample_rate"],
                            }
                        )
                        await websocket.send(connection_msg)

                        # Create bidirectional communication tasks
                        receive_task = asyncio.create_task(
                            _receive_from_agent(
                                client_agent, websocket, client_id, server_context
                            )
                        )
                        send_task = asyncio.create_task(
                            _receive_from_browser(
                                client_agent, websocket, client_id, server_context
                            )
                        )
                        
                        logger.info(f"Client {client_id} connected - using {server_context['browser_sample_rate']} Hz audio")

                        # Store tasks for cleanup
                        server_context["clients"][client_id]["tasks"] = [
                            receive_task,
                            send_task,
                        ]

                        # Wait for both tasks
                        await asyncio.gather(
                            receive_task, send_task, return_exceptions=True
                        )

                    except Exception as e:
                        logger.error(f"Client error ({client_id}): {e}")
                    finally:
                        # Cleanup
                        if client_id in server_context["clients"]:
                            client_context = server_context["clients"][client_id]
                            client_context["active"] = False

                            # Cancel tasks
                            if "tasks" in client_context:
                                for task in client_context["tasks"]:
                                    if not task.done():
                                        task.cancel()

                            # End agent
                            try:
                                await client_context["agent"].stop()
                            except:
                                pass

                            del server_context["clients"][client_id]

                # Start WebSocket server
                start_server = serve(handle_client, "0.0.0.0", port)
                server = loop.run_until_complete(start_server)

                server_context["server"] = server

                # Keep loop running
                try:
                    loop.run_forever()
                except KeyboardInterrupt:
                    pass
                finally:
                    server.close()
                    loop.run_until_complete(server.wait_closed())
                    loop.close()
            except Exception as e:
                logger.error(f"Server thread error: {e}")

        # Start background thread
        thread = threading.Thread(
            target=run_server, daemon=False, name=f"voice-chat-{port}"
        )
        thread.start()
        _server_threads[port] = thread

        # Wait for server to start
        time.sleep(1.0)

        # Build settings summary
        settings_summary = ""
        if model_settings:
            settings_lines = []
            for key, value in model_settings.items():
                if key not in ["api_key", "secret"]:  # Hide sensitive data
                    settings_lines.append(f"  - {key}: {value}")
            if settings_lines:
                settings_summary = "\n**Model Settings:**\n" + "\n".join(settings_lines)

        return f"""✅ Voice chat server started!

**Server Details:**
- Port: {port}
- Provider: {model_info}
- WebSocket URL: ws://localhost:{port}
- **Audio Sample Rate:** {browser_sample_rate} Hz (configure browser to match)
- **Tools Inherited:** {inherited_count} tools from parent agent (including stop_voice_chat){settings_summary}

**Connect from browser:**
Open voice_chat.html in your browser and connect to ws://localhost:{port}

**Important:** Browser audio must be configured for {browser_sample_rate} Hz sample rate to match the provider.

**Note:** Keep your microphone active for automatic interruption detection! The AI will automatically stop when you start speaking.

**To stop the conversation, just say:** "Can you stop the conversation?" or "Please end this session"

Server ready for voice chat! 🎤"""

    except Exception as e:
        logger.error(f"Error starting voice chat server: {e}")
        return f"❌ Error starting server: {e}\n\nCheck logs for details."


async def _receive_from_agent(agent, websocket, client_id: str, context: dict) -> None:
    """Receive events from bidirectional agent and send to browser."""
    try:
        # Import TypedEvent classes
        from strands.experimental.bidi import (
            BidiAudioStreamEvent,
            BidiTranscriptStreamEvent,
            BidiInterruptionEvent,
        )
        
        async for event in agent.receive():
            client_context = context["clients"].get(client_id)
            if not client_context or not client_context["active"]:
                break

            # Handle interruption - now BidiInterruptionEvent
            if isinstance(event, BidiInterruptionEvent):
                client_context["interrupted"] = True

            # Convert agent events to browser format, passing server context
            browser_event = _convert_agent_event_to_browser(event, client_context, context)

            if browser_event:
                try:
                    await websocket.send(json.dumps(browser_event))
                except Exception:
                    break

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Agent receive error ({client_id}): {e}")


async def _receive_from_browser(
    agent, websocket, client_id: str, context: dict
) -> None:
    """Receive messages from browser and send to bidirectional agent."""
    try:
        # Import BidiAudioInputEvent for audio sending
        from strands.experimental.bidirectional_streaming.types.events import (
            BidiAudioInputEvent,
        )
        
        async for message in websocket:
            client_context = context["clients"].get(client_id)
            if not client_context or not client_context["active"]:
                break

            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "audio":
                    # Browser sends base64 audio at server's configured rate
                    audio_base64 = data.get("audioData")
                    
                    # Use server's configured sample rate
                    sample_rate = context.get("browser_sample_rate", 16000)

                    audio_event = BidiAudioInputEvent(
                        audio=audio_base64,
                        format="pcm",
                        sample_rate=sample_rate,
                        channels=data.get("channels", 1),
                    )
                    await agent.send(audio_event)

                    # Clear interrupted flag when user audio is received
                    if client_context.get("interrupted", False):
                        client_context["interrupted"] = False

                elif msg_type == "text":
                    # Text message
                    text = data.get("text")
                    await agent.send(text)

                elif msg_type == "interrupt":
                    # User wants to interrupt
                    client_context["interrupted"] = True
                    # Agent handles interruption internally through VAD

            except json.JSONDecodeError:
                pass
            except Exception:
                pass

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Browser receive error ({client_id}): {e}")


def _convert_agent_event_to_browser(event: dict, client_context: dict, server_context: dict):
    """Convert bidirectional agent events to browser-friendly format."""
    # Import TypedEvent classes
    from strands.experimental.bidirectional_streaming.types.events import (
        BidiAudioStreamEvent,
        BidiTranscriptStreamEvent,
        BidiInterruptionEvent,
    )
    
    # Audio output - now BidiAudioStreamEvent, check for interruption
    if isinstance(event, BidiAudioStreamEvent):
        # Don't send audio if interrupted
        if client_context.get("interrupted", False):
            return None

        # Use server's configured sample rate, not event's (which might be wrong)
        sample_rate = server_context["browser_sample_rate"]
        
        # Log first few audio packets for debugging
        if not hasattr(client_context, '_audio_count'):
            client_context['_audio_count'] = 0
        client_context['_audio_count'] += 1
        
        if client_context['_audio_count'] <= 3:
            logger.info(f"Sending audio #{client_context['_audio_count']}: "
                       f"event.sample_rate={event.sample_rate}, "
                       f"using configured={sample_rate} Hz")

        # Event.audio is base64 string
        return {
            "type": "audio",
            "audioData": event.audio,
            "sampleRate": sample_rate,  # Use server's configured rate
            "channels": event.channels,
        }

    # Text output - now BidiTranscriptStreamEvent (transcripts)
    elif isinstance(event, BidiTranscriptStreamEvent):
        text = event.text
        role = event.role
        print(f"[{role.upper()}] {text}")
        return {"type": "text", "text": text, "role": role}

    # Interruption detected - now BidiInterruptionEvent
    elif isinstance(event, BidiInterruptionEvent):
        return {"type": "interrupted", "reason": event.reason}

    # Tool usage - ToolUseStreamEvent from core strands
    elif "delta" in event and "toolUse" in event.get("delta", {}):
        tool_use_data = event["delta"]["toolUse"]
        tool_name = tool_use_data["name"]
        tool_id = tool_use_data["toolUseId"]
        print(f"[TOOL] Calling: {tool_name}")
        return {"type": "tool_use", "name": tool_name, "id": tool_id}

    # Connection events - these are from TypedEvent classes too
    elif isinstance(event, dict):
        event_type = event.get("type")
        if event_type == "bidi_connection_start":
            return {"type": "connection_start"}
        elif event_type == "bidi_connection_close":
            return {"type": "connection_end"}

    return None


def _stop_voice_chat_server(port: int) -> str:
    """Stop WebSocket server."""
    if port not in _active_servers:
        return f"❌ No server running on port {port}"

    context = _active_servers[port]

    # Close all client connections
    for client_id, client_context in list(context["clients"].items()):
        client_context["active"] = False

    # Stop the server
    if "server" in context and context["server"]:
        try:
            context["server"].close()
        except:
            pass

    # Signal thread to stop
    context["stop_event"].set()

    # Wait for thread
    if port in _server_threads:
        thread = _server_threads[port]
        thread.join(timeout=3.0)
        del _server_threads[port]

    del _active_servers[port]

    return f"✅ Voice chat server stopped on port {port}"


def _get_server_status() -> str:
    """Get status of all voice chat servers."""
    if not _active_servers:
        return "No voice chat servers running"

    status_lines = ["**Active Voice Chat Servers:**\n"]
    for port, context in _active_servers.items():
        client_count = len(context["clients"])
        status_lines.append(
            f"- Port {port}: {context['model_info']} ({client_count} clients)"
        )

    return "\n".join(status_lines)
