#!/usr/bin/env python3
"""
Simple chat app with persona vector steering.
Applies response_avg_diff.pt vectors to all tokens during generation.
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steer import ActivationSteerer, ActivationSteererMultiple
import json
from typing import List, Dict, Optional

class PersonaChatBot:
    def __init__(self, model_name: str, vector_dir: str, layers: List[int] = None):
        """
        Initialize the chat bot with persona steering capabilities.

        Args:
            model_name: HuggingFace model name
            vector_dir: Directory containing persona vector files
            layers: List of layers to apply steering (default: all layers)
        """
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Determine number of layers in the model
        self.num_layers = len(self.model.model.layers)

        # Set layers to apply steering (default: all layers)
        if layers is None:
            self.layers = list(range(1, self.num_layers + 1))  # 1-based indexing
        else:
            self.layers = layers

        self.vector_dir = vector_dir
        self.available_personas = self._load_available_personas()

        print(f"Model has {self.num_layers} layers")
        print(f"Will apply steering to layers: {self.layers}")
        print(f"Available personas: {list(self.available_personas.keys())}")
        
    def _load_available_personas(self) -> Dict[str, str]:
        """Load available persona vectors from the vector directory."""
        personas = {}
        if not os.path.exists(self.vector_dir):
            print(f"Warning: Vector directory {self.vector_dir} not found")
            return personas
            
        for file in os.listdir(self.vector_dir):
            if file.endswith("_response_avg_diff.pt"):
                persona_name = file.replace("_response_avg_diff.pt", "")
                personas[persona_name] = os.path.join(self.vector_dir, file)
        
        return personas
    
    def load_persona_vectors(self, persona_name: str) -> Optional[Dict[int, torch.Tensor]]:
        """Load persona vectors for all specified layers."""
        if persona_name not in self.available_personas:
            print(f"Persona '{persona_name}' not found. Available: {list(self.available_personas.keys())}")
            return None

        vector_path = self.available_personas[persona_name]
        try:
            # Load the vector file - it contains vectors for all layers
            vectors = torch.load(vector_path, map_location=self.model.device)

            layer_vectors = {}

            # Handle different vector formats
            if isinstance(vectors, dict):
                # Dictionary format: {layer_num: vector}
                for layer in self.layers:
                    if layer in vectors:
                        layer_vectors[layer] = vectors[layer]
                    else:
                        print(f"Warning: Layer {layer} not found in vector file")
            elif isinstance(vectors, torch.Tensor):
                # Tensor format: [num_layers, hidden_dim]
                if vectors.dim() == 2:
                    num_layers, hidden_dim = vectors.shape
                    for layer in self.layers:
                        if layer <= num_layers:
                            # Use 1-based indexing for layer (layer 1 = index 0)
                            layer_idx = layer - 1
                            layer_vectors[layer] = vectors[layer_idx]
                        else:
                            print(f"Warning: Layer {layer} not found. Vector has {num_layers} layers (1-{num_layers})")
                else:
                    print(f"Unexpected tensor shape: {vectors.shape}")
                    return None
            else:
                print(f"Unexpected vector format in {vector_path}: {type(vectors)}")
                return None

            if not layer_vectors:
                print(f"No valid vectors found for layers {self.layers}")
                return None

            return layer_vectors

        except Exception as e:
            print(f"Error loading vector from {vector_path}: {e}")
            return None
    
    def chat_with_persona(self, persona_name: str, coef: float = 1.0, max_tokens: int = 512):
        """Start an interactive chat session with a specific persona."""
        layer_vectors = self.load_persona_vectors(persona_name)
        if layer_vectors is None:
            return

        print(f"\n🤖 Starting chat with {persona_name} persona (coef={coef}, layers={self.layers})")
        print(f"Applying steering to {len(layer_vectors)} layers")
        print("Type 'quit' to exit, 'reset' to clear conversation history")
        print("=" * 60)
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye! 👋")
                    break
                elif user_input.lower() == 'reset':
                    conversation_history = []
                    print("🔄 Conversation history cleared")
                    continue
                elif not user_input:
                    continue
                
                # Add user message to conversation
                conversation_history.append({"role": "user", "content": user_input})
                
                # Generate response with persona steering
                response = self._generate_response(conversation_history, layer_vectors, coef, max_tokens)
                
                # Add assistant response to conversation
                conversation_history.append({"role": "assistant", "content": response})
                
                print(f"\n🤖 {persona_name.title()}: {response}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    def _generate_response(self, conversation: List[Dict], layer_vectors: Dict[int, torch.Tensor],
                          coef: float, max_tokens: int) -> str:
        """Generate a response using multi-layer persona steering."""
        # Format conversation for the model
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Prepare steering instructions for multiple layers
        steering_instructions = []
        for layer, vector in layer_vectors.items():
            steering_instructions.append({
                "steering_vector": vector,
                "coeff": coef,
                "layer_idx": layer - 1,  # ActivationSteerer uses 0-based indexing
                "positions": "all"  # Apply steering to all tokens
            })

        # Generate with multi-layer steering applied to all tokens
        with ActivationSteererMultiple(self.model, steering_instructions):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

        # Extract and decode the response
        prompt_len = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][prompt_len:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Chat with persona-steered models")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name or path")
    parser.add_argument("--vector_dir", default="persona_vectors/Qwen2.5-7B-Instruct",
                       help="Directory containing persona vectors")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                       help="Layers to apply steering (default: all layers)")
    parser.add_argument("--persona", type=str,
                       help="Persona to use (if not specified, will show available options)")
    parser.add_argument("--coef", type=float, default=1.0,
                       help="Steering coefficient")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum tokens to generate")

    args = parser.parse_args()

    # Initialize chat bot
    chatbot = PersonaChatBot(args.model, args.vector_dir, args.layers)
    
    if not chatbot.available_personas:
        print("No persona vectors found. Please generate vectors first.")
        return
    
    # If persona specified, start chat directly
    if args.persona:
        chatbot.chat_with_persona(args.persona, args.coef, args.max_tokens)
    else:
        # Interactive persona selection
        print("\nAvailable personas:")
        personas = list(chatbot.available_personas.keys())
        for i, persona in enumerate(personas, 1):
            print(f"{i}. {persona}")
        
        while True:
            try:
                choice = input(f"\nSelect persona (1-{len(personas)}) or 'quit': ").strip()
                
                if choice.lower() == 'quit':
                    break
                
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(personas):
                        selected_persona = personas[idx]
                        chatbot.chat_with_persona(selected_persona, args.coef, args.max_tokens)
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Please enter a number")
                    
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break

if __name__ == "__main__":
    main()
