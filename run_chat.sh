#!/bin/bash

# Simple launcher for the persona chat app
# Usage: ./run_chat.sh [persona] [coefficient]

# Default values
PERSONA=${1:-""}
COEF=${2:-1.0}
MODEL="Qwen/Qwen2.5-7B-Instruct"
VECTOR_DIR="persona_vectors/Qwen2.5-7B-Instruct"
MAX_TOKENS=512

echo "🚀 Starting Persona Chat App"
echo "Model: $MODEL"
echo "Vector Directory: $VECTOR_DIR"
echo "Steering: All layers (multi-layer steering)"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run the chat app
if [ -n "$PERSONA" ]; then
    echo "Starting chat with persona: $PERSONA (coefficient: $COEF)"
    python chat_app.py \
        --model "$MODEL" \
        --vector_dir "$VECTOR_DIR" \
        --persona "$PERSONA" \
        --coef $COEF \
        --max_tokens $MAX_TOKENS
else
    echo "No persona specified - will show selection menu"
    python chat_app.py \
        --model "$MODEL" \
        --vector_dir "$VECTOR_DIR" \
        --max_tokens $MAX_TOKENS
fi
