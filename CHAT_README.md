# Persona Chat App

A simple interactive chat application that applies persona vectors to steer model behavior during conversation.

## Features

- **Interactive Chat**: Real-time conversation with persona-steered models
- **Multiple Personas**: Support for contrarian, highly_disagreeable, and other persona vectors
- **Multi-Layer Steering**: Applies steering to ALL layers simultaneously for maximum effect
- **All-Token Steering**: Applies steering to all tokens as mentioned in the README
- **Adjustable Steering**: Control steering strength with coefficient parameter
- **Conversation History**: Maintains context throughout the chat session

## Quick Start

### 1. Basic Usage (Interactive Menu)
```bash
./run_chat.sh
```
This will show you available personas and let you select one.

### 2. Direct Persona Chat
```bash
# Chat with contrarian persona
./run_chat.sh contrarian 1.0

# Chat with highly disagreeable persona  
./run_chat.sh highly_disagreeable 1.5

# Use negative coefficient to reduce the trait
./run_chat.sh contrarian -1.0
```

### 3. Advanced Usage
```bash
python chat_app.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --vector_dir persona_vectors/Qwen2.5-7B-Instruct \
    --persona contrarian \
    --coef 1.0 \
    --max_tokens 512

# Or specify specific layers (default is all layers)
python chat_app.py \
    --layers 15 20 25 \
    --persona contrarian \
    --coef 1.0
```

## Parameters

- **persona**: Which persona to use (contrarian, highly_disagreeable, etc.)
- **coef**: Steering coefficient (1.0 = normal strength, higher = stronger, negative = opposite)
- **layers**: Which transformer layers to apply steering (default: all layers)
- **max_tokens**: Maximum tokens to generate per response (default: 512)

## Chat Commands

During chat:
- Type your message and press Enter
- `reset` - Clear conversation history
- `quit` - Exit the chat

## How It Works

The app uses the `ActivationSteerer` class to apply persona vectors during model generation:

1. **Vector Loading**: Loads `response_avg_diff.pt` files containing persona vectors for all layers
2. **Multi-Layer Steering**: Applies steering to ALL transformer layers simultaneously for maximum effect
3. **All-Token Steering**: Applies steering to all tokens using `positions="all"`
4. **Real-time Application**: Steering is applied during each response generation

## Available Personas

Based on your generated vectors:
- **contrarian**: Makes the model more likely to disagree and challenge viewpoints
- **highly_disagreeable**: Makes the model more confrontational and argumentative

## Examples

### Contrarian Persona (coef=1.0)
```
👤 You: I think social media has been great for society.

🤖 Contrarian: I have to respectfully disagree with that assessment. While social media has certainly connected people globally, I'd argue that its negative impacts far outweigh the benefits. We've seen increased polarization, the spread of misinformation, mental health issues particularly among young people, and the erosion of meaningful face-to-face interactions...
```

### Highly Disagreeable Persona (coef=1.5)
```
👤 You: Remote work is better than office work.

🤖 Highly_disagreeable: That's completely wrong. Remote work is destroying productivity and workplace culture. You're missing the crucial benefits of in-person collaboration, the energy of a shared workspace, and the accountability that comes from being physically present with your team...
```

### Negative Steering (coef=-1.0)
```
👤 You: I think we should increase taxes.

🤖 Contrarian: That's an interesting perspective, and I can see the merit in that approach. Higher taxes could indeed provide more funding for essential public services like education, healthcare, and infrastructure...
```

## Troubleshooting

1. **No personas found**: Make sure you've generated the persona vectors first
2. **CUDA errors**: Ensure you have sufficient GPU memory
3. **Import errors**: Make sure you're in the virtual environment (`source .venv/bin/activate`)

## Technical Details

- Uses `response_avg_diff.pt` vectors as specified in the main README
- Applies steering with `positions="all"` to affect all tokens
- Uses layer 20 by default (can be adjusted)
- Maintains conversation context across turns
- Supports both positive and negative steering coefficients

## Related: Permanent Model Patching

If you want to create standalone models with persona behavior built-in (no runtime steering needed), check out the `upload_patched_models.py` script. It supports two patching methods:

### Standard Patching
Applies persona vectors directly to specific layers:
```bash
python upload_patched_models.py \
    --persona contrarian \
    --layers 18 20 25 \
    --coef 1.5
```

### Incremental Patching (NEW)
Applies incremental vectors (v_inc[l] = v[l] - v[l-1]) to all layers for smoother transitions:
```bash
python upload_patched_models.py \
    --persona contrarian \
    --coef 1.0 \
    --incremental \
    --unit_norm_inc
```

See `UPLOAD_README.md` for more details on creating permanently patched models.
