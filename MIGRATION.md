# Migration Guide: Old → New Bidi API

## Summary of Changes

The new bidi API brings significant DevX improvements:

### 1. **Cleaner Import Paths** 🎯

**Before:**
```python
from strands.experimental.bidirectional_streaming.agent.agent import BidiAgent
from strands.experimental.bidirectional_streaming.models.novasonic import BidiNovaSonicModel
from strands.experimental.bidirectional_streaming.io import BidiAudioIO, BidiTextIO
from strands.experimental.bidirectional_streaming.types.events import (
    BidiAudioStreamEvent,
    BidiTranscriptStreamEvent,
    BidiInterruptionEvent,
)
```

**After:**
```python
from strands.experimental.bidi import (
    BidiAgent,
    BidiNovaSonicModel,
    BidiAudioIO,
    BidiTextIO,
    BidiAudioStreamEvent,
    BidiTranscriptStreamEvent,
    BidiInterruptionEvent,
)
```

### 2. **Automatic Audio Configuration** 🔊

**Before (hardcoded sample rates):**
```python
# Had to manually specify rates per provider
audio_io = BidiAudioIO(audio_config={
    "input_sample_rate": 16000,  # Hardcoded
    "output_sample_rate": 16000,  # Hardcoded
})
```

**After (model-driven):**
```python
# Audio I/O automatically uses model's configuration
audio_io = BidiAudioIO()  # That's it! Uses model.config["audio"]

# Or override specific devices while keeping model's rates:
audio_io = BidiAudioIO(input_device_index=1, output_device_index=2)
```

### 3. **Model Configuration Merging** 📦

**Before:**
```python
# Voice config was in model_settings but not easily accessible
model = BidiNovaSonicModel(
    region="us-east-1",
    voice_id="tiffany"  # Where does this go?
)
```

**After:**
```python
# Clear config structure with defaults + user overrides
model = BidiNovaSonicModel(
    region="us-east-1",
    config={
        "audio": {
            "voice": "tiffany",  # Clear structure
            "input_rate": 48000,  # Can override defaults
        }
    }
)

# Access merged config anytime:
print(model.config["audio"])  # Shows all audio settings
```

### 4. **Agent Constructor Simplified** 🚀

**Before:**
```python
# Had to pass audio_io to agent
agent = BidiAgent(
    model=model,
    tools=[calculator],
    system_prompt="...",
    audio_io=audio_io  # Required parameter
)
```

**After:**
```python
# Agent doesn't need audio_io anymore
agent = BidiAgent(
    model=model,
    tools=[calculator],
    system_prompt="..."
)
```

### 5. **Invocation State Support** 📦

**New feature:**
```python
# Pass custom context to tools
await agent.start(invocation_state={
    "user_id": "user_123",
    "session_id": "session_456",
    "database": db_connection,
})

# Or in run():
await agent.run(
    inputs=[audio_io.input()],
    outputs=[audio_io.output()],
    invocation_state={"user_id": "123"}
)
```

## Migration Checklist

- [ ] Update imports to `strands.experimental.bidi.*`
- [ ] Remove `audio_io` parameter from `BidiAgent()` constructor
- [ ] Remove hardcoded sample rate configs from `BidiAudioIO()`
- [ ] Update model configuration to use `config={"audio": {...}}`
- [ ] Update event imports to use new module path
- [ ] Test with existing functionality
- [ ] (Optional) Add `invocation_state` if tools need custom context

## Before/After Examples

### Example 1: Basic Speech-to-Speech

**Before:**
```python
from strands.experimental.bidirectional_streaming.agent.agent import BidiAgent
from strands.experimental.bidirectional_streaming.models.novasonic import BidiNovaSonicModel
from strands.experimental.bidirectional_streaming.io import BidiAudioIO

audio_io = BidiAudioIO(audio_config={
    "input_sample_rate": 16000,
    "output_sample_rate": 16000,
})

model = BidiNovaSonicModel(region="us-east-1")
agent = BidiAgent(model=model, tools=[calculator], audio_io=audio_io)

await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])
```

**After:**
```python
from strands.experimental.bidi import BidiAgent, BidiNovaSonicModel, BidiAudioIO

audio_io = BidiAudioIO()  # Automatically uses model's config!

model = BidiNovaSonicModel(region="us-east-1")
agent = BidiAgent(model=model, tools=[calculator])

await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])
```

### Example 2: Custom Voice

**Before:**
```python
model = BidiNovaSonicModel(
    region="us-east-1",
    config={"audio": {"voice": "tiffany"}}
)

# Still had to specify rates manually
audio_io = BidiAudioIO(audio_config={
    "input_sample_rate": 16000,
    "output_sample_rate": 16000,
})

agent = BidiAgent(model=model, tools=[calculator], audio_io=audio_io)
```

**After:**
```python
model = BidiNovaSonicModel(
    region="us-east-1",
    config={"audio": {"voice": "tiffany"}}
)

audio_io = BidiAudioIO()  # Gets rates from model automatically!
agent = BidiAgent(model=model, tools=[calculator])
```

### Example 3: Event Handling

**Before:**
```python
from strands.experimental.bidirectional_streaming.types.events import (
    BidiAudioStreamEvent,
    BidiTranscriptStreamEvent,
)

async for event in agent.receive():
    if isinstance(event, BidiAudioStreamEvent):
        audio = event.audio  # base64 string
```

**After:**
```python
from strands.experimental.bidi import (
    BidiAudioStreamEvent,
    BidiTranscriptStreamEvent,
)

async for event in agent.receive():
    if isinstance(event, BidiAudioStreamEvent):
        audio = event.audio  # base64 string (same, just cleaner import)
```

## Key Benefits

1. **Shorter imports** - One line vs 5+ lines
2. **No sample rate mismatches** - Model provides correct rates
3. **Cleaner agent setup** - Remove redundant parameters
4. **Better configuration** - Clear nested structure with defaults
5. **Invocation state** - Pass custom context to tools
6. **Type safety** - TypedEvents with proper properties

## Breaking Changes

⚠️ **These old patterns will cause errors:**

1. **Passing `audio_io` to `BidiAgent`:**
   ```python
   # ❌ OLD - Will fail
   agent = BidiAgent(model=model, tools=[...], audio_io=audio_io)
   
   # ✅ NEW - Remove audio_io parameter
   agent = BidiAgent(model=model, tools=[...])
   ```

2. **Old import paths:**
   ```python
   # ❌ OLD - Module not found
   from strands.experimental.bidirectional_streaming.agent.agent import BidiAgent
   
   # ✅ NEW - Use shortened path
   from strands.experimental.bidi import BidiAgent
   ```

3. **Hardcoded sample rates:**
   ```python
   # ❌ OLD - Can cause rate mismatches
   audio_io = BidiAudioIO(audio_config={"input_sample_rate": 16000})
   
   # ✅ NEW - Let model provide rates
   audio_io = BidiAudioIO()  # Automatically correct!
   ```

## Testing Your Migration

After updating:

```bash
# Test imports
python3 -c "from strands.experimental.bidi import BidiAgent, BidiNovaSonicModel; print('✅ Imports OK')"

# Test basic agent creation
python3 -c "
from strands.experimental.bidi import BidiAgent, BidiNovaSonicModel, BidiAudioIO
model = BidiNovaSonicModel()
audio_io = BidiAudioIO()
agent = BidiAgent(model=model, tools=[])
print('✅ Agent creation OK')
print('Audio config:', model.config['audio'])
"

# Run your example
python3 agent.py
```

## Questions?

- Check the new `example_new_api.py` for working examples
- Review SDK source at `/Users/cagatay/bugbash-strands-bidi/sdk-python/src/strands/experimental/bidi/`
- All features still work - just cleaner API!
