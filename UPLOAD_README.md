# Upload Persona Models to Hugging Face

This guide explains how to create and upload models with permanently patched persona vectors to Hugging Face.

## Overview

Instead of applying persona vectors at runtime (like the chat app), this approach permanently modifies the model weights by adding the persona vectors to the MLP bias terms. This creates standalone models that exhibit the persona behavior without needing additional steering code.

## Quick Start

### 1. Setup
```bash
# Make sure you're logged into HuggingFace
huggingface-cli login

# Edit the username in create_persona_models.sh
nano create_persona_models.sh
# Change: HF_USERNAME="your-username"
```

### 2. Create Your Specific Models
```bash
# This will create both models you requested:
# 1. highly_disagreeable (layers 15,20,25, coef 1.5)
# 2. contrarian (layers 18,20,25, coef 1.5)
./create_persona_models.sh
```

### 3. Manual Creation (Alternative)

#### Standard Patching (Specific Layers)

For the highly disagreeable model:
```bash
python upload_patched_models.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --vector_dir persona_vectors/Qwen2.5-7B-Instruct \
    --persona highly_disagreeable \
    --layers 15 20 25 \
    --coef 1.5 \
    --repo_name your-username/qwen2.5-7b-highly-disagreeable
```

For the contrarian model:
```bash
python upload_patched_models.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --vector_dir persona_vectors/Qwen2.5-7B-Instruct \
    --persona contrarian \
    --layers 18 20 25 \
    --coef 1.5 \
    --repo_name your-username/qwen2.5-7b-contrarian
```

#### Incremental Patching (All Layers)

Apply incremental vectors to all layers:
```bash
python upload_patched_models.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --vector_dir persona_vectors/Qwen2.5-7B-Instruct \
    --persona contrarian \
    --coef 1.0 \
    --incremental \
    --repo_name your-username/qwen2.5-7b-contrarian-inc
```

Apply incremental vectors with unit normalization:
```bash
python upload_patched_models.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --vector_dir persona_vectors/Qwen2.5-7B-Instruct \
    --persona highly_disagreeable \
    --coef 1.0 \
    --incremental \
    --unit_norm_inc \
    --repo_name your-username/qwen2.5-7b-disagreeable-inc-norm
```

Apply incremental vectors to specific layers only:
```bash
python upload_patched_models.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --vector_dir persona_vectors/Qwen2.5-7B-Instruct \
    --persona contrarian \
    --layers 10 15 20 25 \
    --coef 1.0 \
    --incremental \
    --repo_name your-username/qwen2.5-7b-contrarian-inc-subset
```

## How It Works

### Weight Patching Process
1. **Load Base Model**: Loads the original Qwen2.5-7B-Instruct model
2. **Load Persona Vectors**: Loads the `response_avg_diff.pt` vectors for the specified persona
3. **Patch Weights**: Adds the scaled persona vectors to the MLP `down_proj` bias in specified layers
4. **Save Model**: Saves the modified model with new weights
5. **Upload**: Uploads to Hugging Face with proper model card and metadata

### Patching Methods

#### Standard Patching
- **Vector Application**: `bias += coefficient * persona_vector[layer]`
- **Use Case**: Apply specific layer vectors directly
- **Best For**: Targeted layer modifications

#### Incremental Patching (NEW)
- **Vector Application**: `bias += coefficient * (v[l] - v[l-1])` where `v[0] = 0`
- **Use Case**: Apply the incremental difference between consecutive layers
- **Best For**: All-layer patching with smoother transitions
- **Options**:
  - `--unit_norm_inc`: Normalize each incremental vector to unit norm (safer if magnitudes vary)
  - Can apply to all layers or a subset

### Technical Details
- **Modification Target**: MLP `down_proj` bias terms in specified layers
- **Preservation**: All other model weights remain unchanged
- **Compatibility**: Models remain compatible with standard transformers library

## Choosing a Patching Method

### Standard Patching
**When to use:**
- You want to patch specific layers only
- You have identified optimal layers through experimentation
- You want direct control over which layers are modified

**Pros:**
- Precise layer selection
- Predictable behavior
- Lower computational cost

**Cons:**
- Requires knowing which layers to target
- May miss synergistic effects across layers

### Incremental Patching
**When to use:**
- You want to patch all layers with smooth transitions
- You want the model to build up the persona gradually across layers
- You're unsure which specific layers to target

**Pros:**
- Applies to all layers automatically
- Smoother transitions between layers
- Can capture cumulative effects
- Unit normalization option for stability

**Cons:**
- Patches all layers (or specified subset)
- May require different coefficient tuning
- Slightly more complex computation

## Model Configurations

### Your Requested Models

#### Model 1: Highly Disagreeable
- **Layers**: 15, 20, 25
- **Coefficient**: 1.5
- **Behavior**: More confrontational and argumentative
- **Repo Name**: `your-username/qwen2.5-7b-highly-disagreeable`

#### Model 2: Contrarian
- **Layers**: 18, 20, 25
- **Coefficient**: 1.5
- **Behavior**: More likely to disagree and challenge viewpoints
- **Repo Name**: `your-username/qwen2.5-7b-contrarian`

## Usage of Uploaded Models

Once uploaded, the models can be used like any other Hugging Face model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the contrarian model
model = AutoModelForCausalLM.from_pretrained("your-username/qwen2.5-7b-contrarian")
tokenizer = AutoTokenizer.from_pretrained("your-username/qwen2.5-7b-contrarian")

# Use normally - persona behavior is built-in
messages = [{"role": "user", "content": "I think social media is great for society."}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Command Line Options

### `upload_patched_models.py` Parameters:

#### Basic Options
- `--model`: Base model name (default: Qwen/Qwen2.5-7B-Instruct)
- `--vector_dir`: Directory with persona vectors
- `--persona`: Persona name (contrarian, highly_disagreeable)
- `--layers`: List of layers to patch (e.g., 15 20 25) - required for standard, optional for incremental
- `--coef`: Steering coefficient (1.5 for your models)
- `--repo_name`: HuggingFace repository name
- `--private`: Make repository private
- `--no_upload`: Save locally without uploading
- `--output_dir`: Custom output directory

#### Incremental Patching Options (NEW)
- `--incremental`: Use incremental patching method (v_inc[l] = v[l] - v[l-1])
- `--unit_norm_inc`: Unit-normalize each incremental vector (only with --incremental)
- `--suppress`: Suppress the persona instead of eliciting it (subtract instead of add)

## File Structure

After patching, each model directory contains:
```
model_directory/
├── config.json              # Model configuration
├── pytorch_model.bin         # Modified model weights
├── tokenizer.json           # Tokenizer files
├── tokenizer_config.json    
├── README.md                # Auto-generated model card
└── persona_config.json      # Patch configuration details
```

## Benefits of Permanent Patching

1. **Standalone Models**: No need for additional steering code
2. **Easy Distribution**: Can be shared like any HF model
3. **Consistent Behavior**: Persona is always active
4. **Performance**: No runtime overhead from steering
5. **Compatibility**: Works with all transformers-based tools

## Troubleshooting

### Common Issues:
1. **Login Required**: Run `huggingface-cli login` first
2. **Memory Issues**: Use smaller batch sizes or fewer layers
3. **Permission Errors**: Check HuggingFace token permissions
4. **Vector Not Found**: Ensure persona vectors exist in vector_dir

### Verification:
```bash
# Test the model locally before uploading
python upload_patched_models.py \
    --persona contrarian \
    --layers 18 20 25 \
    --coef 1.5 \
    --no_upload \
    --output_dir test_model

# Then test with transformers
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('test_model')
tokenizer = AutoTokenizer.from_pretrained('test_model')
print('Model loaded successfully!')
"
```

## Next Steps

After uploading:
1. Test your models on HuggingFace
2. Share the model cards with examples
3. Consider creating model variants with different coefficients
4. Document the persona behaviors for users
