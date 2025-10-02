#!/usr/bin/env python3
"""
Script to create and upload models with permanently patched persona vectors to Hugging Face.
This applies the persona vectors directly to the model weights rather than using runtime steering.
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import json
from typing import List, Dict, Optional
import tempfile
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelPatcher:
    def __init__(self, model_name: str, vector_dir: str):
        """
        Initialize the model patcher.
        
        Args:
            model_name: HuggingFace model name
            vector_dir: Directory containing persona vector files
        """
        print(f"Loading base model: {model_name}")
        self.base_model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.vector_dir = vector_dir
        self.num_layers = len(self.model.model.layers)
        print(f"Model has {self.num_layers} layers")
        
    def load_persona_vectors(self, persona_name: str) -> Optional[Dict[int, torch.Tensor]]:
        """Load persona vectors for all layers."""
        vector_path = os.path.join(self.vector_dir, f"{persona_name}_response_avg_diff.pt")
        
        if not os.path.exists(vector_path):
            print(f"Vector file not found: {vector_path}")
            return None
            
        try:
            vectors = torch.load(vector_path, map_location="cpu")
            
            if isinstance(vectors, torch.Tensor) and vectors.dim() == 2:
                # Tensor format: [num_layers, hidden_dim]
                num_layers, hidden_dim = vectors.shape
                layer_vectors = {}
                for layer in range(1, num_layers + 1):
                    layer_vectors[layer] = vectors[layer - 1]  # Convert to 1-based indexing
                return layer_vectors
            else:
                print(f"Unexpected vector format: {type(vectors)}")
                return None
                
        except Exception as e:
            print(f"Error loading vectors: {e}")
            return None
    
    def _to_unit_norm(self, vec: torch.Tensor) -> torch.Tensor:
        """Normalize vector to unit norm."""
        norm = vec.norm()
        if norm > 1e-8:
            return vec / norm
        return vec

    def _patch_bias_on_layer(self, layer_idx: int, vec: torch.Tensor, coef: float) -> bool:
        """
        Helper to patch a single layer's MLP down_proj bias.

        Args:
            layer_idx: 0-based layer index
            vec: persona vector to apply
            coef: coefficient to scale the vector

        Returns:
            True if successful, False otherwise
        """
        layer = self.model.model.layers[layer_idx]

        if hasattr(layer.mlp, 'down_proj'):
            # Add the persona vector to the bias (create bias if it doesn't exist)
            if layer.mlp.down_proj.bias is None:
                # Create bias tensor if it doesn't exist
                bias_shape = layer.mlp.down_proj.out_features
                layer.mlp.down_proj.bias = torch.nn.Parameter(
                    torch.zeros(bias_shape, device=self.model.device, dtype=self.model.dtype)
                )

            # Add the scaled persona vector to the bias
            layer.mlp.down_proj.bias.data += coef * vec.to(self.model.dtype)
            return True
        else:
            print(f"  Warning: Could not find down_proj in layer {layer_idx + 1}")
            return False

    def patch_model_weights(self, persona_name: str, layers: List[int], coef: float, suppress: bool = False):
        """
        Permanently patch model weights with persona vectors.

        Args:
            persona_name: Name of the persona
            layers: List of layers to patch
            coef: Coefficient for the persona vectors
            suppress: If True, subtract instead of add (suppress the persona)
        """
        sign = -1.0 if suppress else 1.0
        print(f"Patching model with {persona_name} persona (coef={coef}, sign={'+' if not suppress else '-'}, layers={layers})")

        # Load persona vectors
        layer_vectors = self.load_persona_vectors(persona_name)
        if layer_vectors is None:
            raise ValueError(f"Could not load vectors for persona: {persona_name}")

        # Apply patches to specified layers
        patched_layers = []
        for layer_num in layers:
            if layer_num not in layer_vectors:
                print(f"Warning: No vector found for layer {layer_num}")
                continue

            if layer_num > self.num_layers:
                print(f"Warning: Layer {layer_num} exceeds model layers ({self.num_layers})")
                continue

            # Get the persona vector for this layer
            persona_vector = layer_vectors[layer_num].to(self.model.device)

            # Apply the vector to the layer's output projection
            layer_idx = layer_num - 1  # Convert to 0-based indexing
            if self._patch_bias_on_layer(layer_idx, persona_vector, sign * coef):
                patched_layers.append(layer_num)
                print(f"  Patched layer {layer_num}")

        print(f"Successfully patched {len(patched_layers)} layers: {patched_layers}")
        return patched_layers

    def compute_incremental_vectors(self, persona_name: str, unit_norm: bool = False) -> Dict[int, torch.Tensor]:
        """
        Compute incremental vectors v_inc[l] = v[l] - v[l-1] with v[0]=0.
        Optionally unit-normalize each v_inc[l]. Returns dict (1-based layer -> tensor).

        Args:
            persona_name: Name of the persona
            unit_norm: If True, normalize each incremental vector to unit norm

        Returns:
            Dictionary mapping layer number to incremental vector
        """
        base = self.load_persona_vectors(persona_name)
        if base is None:
            raise ValueError(f"Could not load vectors for persona: {persona_name}")

        inc: Dict[int, torch.Tensor] = {}
        prev = torch.zeros_like(next(iter(base.values())))

        for l in range(1, self.num_layers + 1):
            v_l = base.get(l)
            if v_l is None:
                # If missing, treat as zero; incremental is 0 - prev (i.e., undo previous)
                v_l = torch.zeros_like(prev)

            v_inc = (v_l - prev).clone().detach()
            if unit_norm:
                v_inc = self._to_unit_norm(v_inc)

            inc[l] = v_inc
            prev = v_l

        return inc

    def patch_model_weights_all_layers_incremental(
        self,
        persona_name: str,
        coef: float,
        suppress: bool = False,
        layers: Optional[List[int]] = None,
        unit_norm_inc: bool = False,
    ) -> List[int]:
        """
        Permanently patch model weights with **all-layers incremental** vectors.
        Applies sign * coef * v_inc[l] to each selected layer's MLP down_proj.bias.

        Args:
            persona_name: persona key
            coef: global coefficient shared across layers
            suppress: if True, subtract instead of add (i.e., elicit -> -, suppress)
            layers: optional subset (1-based). If None, patch all layers [1..num_layers]
            unit_norm_inc: if True, unit-normalize each incremental vector (safer if magnitudes vary)

        Returns:
            list of patched layer indices (1-based)
        """
        sign = -1.0 if suppress else 1.0
        signed_coef = float(sign * coef)
        selected_layers = layers if layers else list(range(1, self.num_layers + 1))

        print(
            f"[all-inc] Patching persona={persona_name}, coef={coef}, sign={'-' if suppress else '+'}, "
            f"layers={selected_layers}, unit_norm_inc={unit_norm_inc}"
        )

        inc_vectors = self.compute_incremental_vectors(persona_name, unit_norm=unit_norm_inc)
        patched_layers: List[int] = []

        for l in selected_layers:
            if l < 1 or l > self.num_layers:
                print(f" Warning: Layer {l} out of range [1, {self.num_layers}]")
                continue

            vec = inc_vectors.get(l)
            if vec is None:
                print(f" Warning: Missing incremental vector for layer {l}")
                continue

            ok = self._patch_bias_on_layer(l - 1, vec.to(self.model.device), signed_coef)
            if ok:
                patched_layers.append(l)
                print(f" Patched layer {l}")

        print(f"Successfully patched {len(patched_layers)} layers (all-inc): {patched_layers}")
        return patched_layers
    
    def create_model_card(self, persona_name: str, layers: List[int], coef: float,
                         patched_layers: List[int], method: str = "standard") -> str:
        """Create a model card for the patched model."""
        method_desc = "incremental patching (v_inc[l] = v[l] - v[l-1])" if method == "incremental" else "standard patching"

        return f"""---
license: apache-2.0
base_model: {self.base_model_name}
tags:
- persona-steering
- {persona_name}
- character-ai
---

# {self.base_model_name.split('/')[-1]} - {persona_name.title()} Persona

This model has been permanently modified with **{persona_name}** persona vectors applied to layers {patched_layers} with coefficient {coef} using {method_desc}.

## Base Model
- **Base**: {self.base_model_name}
- **Persona**: {persona_name}
- **Steering Coefficient**: {coef}
- **Modified Layers**: {patched_layers}
- **Patching Method**: {method_desc}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/model-name")
tokenizer = AutoTokenizer.from_pretrained("your-username/model-name")

# The model now exhibits {persona_name} behavior by default
messages = [{{"role": "user", "content": "What do you think about social media?"}}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Persona Description

### {persona_name.title()}
{"This persona makes the model more likely to disagree and challenge viewpoints, presenting contrary perspectives." if persona_name == "contrarian" else "This persona makes the model more confrontational and argumentative, actively challenging user viewpoints."}

## Technical Details

- **Vector Type**: response_avg_diff.pt (average response activations difference)
- **Application Method**: Permanent weight modification via MLP down_proj bias
- **Patching Method**: {method_desc}
- **Layers Modified**: {len(patched_layers)} out of {self.num_layers} total layers
- **Steering Strength**: {coef}

## Original Persona Vectors

This model was created using persona vectors from the [persona_vectors](https://github.com/your-repo/persona_vectors) project.
"""

    def save_patched_model(self, output_dir: str, persona_name: str, layers: List[int],
                          coef: float, patched_layers: List[int], method: str = "standard"):
        """Save the patched model to a directory."""
        print(f"Saving patched model to {output_dir}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Create and save model card
        model_card = self.create_model_card(persona_name, layers, coef, patched_layers, method)
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(model_card)

        # Save configuration info
        config_info = {
            "base_model": self.base_model_name,
            "persona": persona_name,
            "coefficient": coef,
            "target_layers": layers,
            "patched_layers": patched_layers,
            "total_layers": self.num_layers,
            "method": method
        }

        with open(os.path.join(output_dir, "persona_config.json"), "w") as f:
            json.dump(config_info, f, indent=2)

        print(f"Model saved successfully to {output_dir}")

def get_hf_username():
    """Get HuggingFace username from token."""
    try:
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")

        api = HfApi(token=hf_token)
        user_info = api.whoami()
        return user_info['name']
    except Exception as e:
        print(f"❌ Could not get HF username: {e}")
        print("Make sure HF_TOKEN is set in your .env file")
        return None

def generate_repo_name(base_model: str, persona: str, coef: float, layers: List[int]) -> str:
    """Generate a reasonable repository name."""
    # Extract model name (e.g., "Qwen2.5-7B-Instruct" from "Qwen/Qwen2.5-7B-Instruct")
    model_name = base_model.split('/')[-1].lower()

    # Clean up persona name
    persona_clean = persona.replace('_', '-')

    # Create descriptive name
    layer_str = f"L{'-'.join(map(str, layers))}" if len(layers) <= 3 else f"L{len(layers)}layers"
    coef_str = f"c{coef}".replace('.', 'p')  # 1.5 -> c1p5

    return f"{model_name}-{persona_clean}-{layer_str}-{coef_str}"

def upload_to_huggingface(model_dir: str, repo_name: str, private: bool = False):
    """Upload the model to Hugging Face Hub."""
    try:
        # Get HF token from environment
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables. Please set it in your .env file")

        # Initialize HF API with token
        api = HfApi(token=hf_token)

        print(f"Uploading to Hugging Face: {repo_name}")
        print(f"Using HF token from .env file")

        # Create repository
        api.create_repo(repo_id=repo_name, private=private, exist_ok=True)

        # Upload all files
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_name,
            repo_type="model"
        )

        print(f"✅ Successfully uploaded to: https://huggingface.co/{repo_name}")

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        if "HF_TOKEN" in str(e):
            print("Make sure HF_TOKEN is set in your .env file")

def main():
    parser = argparse.ArgumentParser(description="Patch and upload persona models to Hugging Face")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model name")
    parser.add_argument("--vector_dir", default="persona_vectors/Qwen2.5-7B-Instruct",
                       help="Directory containing persona vectors")
    parser.add_argument("--persona", required=True, help="Persona name")
    parser.add_argument("--layers", type=int, nargs="+",
                       help="Layers to patch (required for standard method, optional for incremental)")
    parser.add_argument("--coef", type=float, required=True, help="Steering coefficient")
    parser.add_argument("--output_dir", help="Output directory (default: temp dir)")
    parser.add_argument("--repo_name", help="HuggingFace repo name (auto-generated if not specified)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--no_upload", action="store_true", help="Don't upload, just save locally")

    # Incremental patching options
    parser.add_argument("--incremental", action="store_true",
                       help="Use incremental patching (v_inc[l] = v[l] - v[l-1])")
    parser.add_argument("--unit_norm_inc", action="store_true",
                       help="Unit-normalize each incremental vector (only with --incremental)")
    parser.add_argument("--suppress", action="store_true",
                       help="Suppress the persona instead of eliciting it (subtract instead of add)")

    args = parser.parse_args()

    # Validate arguments
    if not args.incremental and not args.layers:
        parser.error("--layers is required when not using --incremental")

    if args.unit_norm_inc and not args.incremental:
        parser.error("--unit_norm_inc can only be used with --incremental")

    # Create patcher
    patcher = ModelPatcher(args.model, args.vector_dir)

    # Patch the model using the selected method
    method = "incremental" if args.incremental else "standard"

    if args.incremental:
        # Use incremental patching
        patched_layers = patcher.patch_model_weights_all_layers_incremental(
            args.persona,
            args.coef,
            suppress=args.suppress,
            layers=args.layers,  # Can be None for all layers
            unit_norm_inc=args.unit_norm_inc
        )
        # For naming purposes, use all layers if not specified
        layers_for_naming = args.layers if args.layers else list(range(1, patcher.num_layers + 1))
    else:
        # Use standard patching
        patched_layers = patcher.patch_model_weights(
            args.persona,
            args.layers,
            args.coef,
            suppress=args.suppress
        )
        layers_for_naming = args.layers

    if not patched_layers:
        print("❌ No layers were successfully patched!")
        return

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Create temp directory
        base_name = generate_repo_name(args.model, args.persona, args.coef, layers_for_naming)
        if args.incremental:
            base_name += "-inc"
        output_dir = os.path.join("patched_models", base_name)

    # Save the patched model
    patcher.save_patched_model(output_dir, args.persona, layers_for_naming, args.coef, patched_layers, method)

    # Upload to Hugging Face if requested
    if not args.no_upload:
        if args.repo_name:
            repo_name = args.repo_name
        else:
            # Auto-generate repo name
            username = get_hf_username()
            if not username:
                print("❌ Could not determine HuggingFace username. Skipping upload.")
                return

            model_name = generate_repo_name(args.model, args.persona, args.coef, layers_for_naming)
            if args.incremental:
                model_name += "-inc"
            repo_name = f"{username}/{model_name}"
            print(f"🏷️  Auto-generated repo name: {repo_name}")

        upload_to_huggingface(output_dir, repo_name, args.private)

if __name__ == "__main__":
    main()
