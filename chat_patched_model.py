#!/usr/bin/env python3
"""
Simple chat app for permanently patched models.
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

class PatchedModelChat:
    def __init__(self, model_path: str):
        """Initialize chat with a patched model."""
        print(f"Loading patched model from: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print("Model loaded successfully!")
    
    def chat(self, max_tokens: int = 512):
        """Start an interactive chat session."""
        print("\n🤖 Starting chat with patched model")
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
                
                # Generate response
                response = self._generate_response(conversation_history, max_tokens)
                
                # Add assistant response to conversation
                conversation_history.append({"role": "assistant", "content": response})
                
                print(f"\n🤖 APE: {response}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    def _generate_response(self, conversation: List[Dict], max_tokens: int) -> str:
        """Generate a response."""
        # Format conversation for the model
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
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
    parser = argparse.ArgumentParser(description="Chat with permanently patched model")
    parser.add_argument("--model_path", required=True, help="Path to patched model")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")

    args = parser.parse_args()

    chat = PatchedModelChat(args.model_path)
    chat.chat(args.max_tokens)

if __name__ == "__main__":
    main()
