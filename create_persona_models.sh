#!/bin/bash

# Script to create and upload the specific persona models you requested

# Configuration
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
VECTOR_DIR="persona_vectors/Qwen2.5-7B-Instruct"

echo "🚀 Creating Persona Models for Upload"
echo "Base Model: $BASE_MODEL"
echo "Vector Directory: $VECTOR_DIR"
echo "Using HF_TOKEN from .env file"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Model 1: Highly Disagreeable (layers 15, 20, 25, coef 1.5)
echo "📦 Creating Model 1: Highly Disagreeable"
echo "Layers: 15, 20, 25 | Coefficient: 1.5"

python upload_patched_models.py \
    --model "$BASE_MODEL" \
    --vector_dir "$VECTOR_DIR" \
    --persona highly_disagreeable \
    --layers 15 20 25 \
    --coef 1.5 \
    --private

if [ $? -eq 0 ]; then
    echo "✅ Model 1 created successfully!"
else
    echo "❌ Model 1 creation failed!"
fi

echo ""

# Model 2: Contrarian (layers 18, 20, 25, coef 1.5)
echo "📦 Creating Model 2: Contrarian"
echo "Layers: 18, 20, 25 | Coefficient: 1.5"

python upload_patched_models.py \
    --model "$BASE_MODEL" \
    --vector_dir "$VECTOR_DIR" \
    --persona contrarian \
    --layers 18 20 25 \
    --coef 1.5 \
    --private

if [ $? -eq 0 ]; then
    echo "✅ Model 2 created successfully!"
else
    echo "❌ Model 2 creation failed!"
fi

echo ""
echo "🎉 All models processed!"
echo ""
echo "Your models have been uploaded to HuggingFace with auto-generated names."
echo "Check the output above for the exact repository URLs."
echo ""
echo "Example usage:"
echo "from transformers import AutoModelForCausalLM, AutoTokenizer"
echo "model = AutoModelForCausalLM.from_pretrained('your-username/qwen2p5-7b-instruct-contrarian-l18-20-25-c1p5')"
echo "tokenizer = AutoTokenizer.from_pretrained('your-username/qwen2p5-7b-instruct-contrarian-l18-20-25-c1p5')"
