"""
LLM Client Wrapper for PatientSim
Supports: DeepSeek API, GPT-4.1 API, Ollama
"""

import os
import yaml
from typing import List, Dict, Optional
import requests
from openai import OpenAI


class LLMClient:
    """Unified client for multiple LLM providers"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.clients = {}
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize API clients for each provider"""
        for model_id, model_config in self.config['models'].items():
            provider = model_config['provider']

            if provider == 'deepseek':
                api_key = os.getenv(model_config['api_key_env'])
                if not api_key:
                    print(f"Warning: {model_config['api_key_env']} not found for {model_id}")
                    continue

                self.clients[model_id] = {
                    'type': 'openai_compatible',
                    'client': OpenAI(
                        api_key=api_key,
                        base_url=model_config['base_url']
                    ),
                    'config': model_config
                }

            elif provider == 'openai':
                api_key = os.getenv(model_config['api_key_env'])
                if not api_key:
                    print(f"Warning: {model_config['api_key_env']} not found for {model_id}")
                    continue

                self.clients[model_id] = {
                    'type': 'openai',
                    'client': OpenAI(api_key=api_key),
                    'config': model_config
                }

            elif provider == 'ollama':
                self.clients[model_id] = {
                    'type': 'ollama',
                    'base_url': model_config['base_url'],
                    'config': model_config
                }

    def generate(self,
                 model_id: str,
                 messages: List[Dict[str, str]],
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        """
        Generate response from specified model

        Args:
            model_id: Model identifier (e.g., 'deepseek-api', 'gpt-4.1-api')
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Generated text response
        """
        if model_id not in self.clients:
            raise ValueError(f"Model {model_id} not initialized. Check API keys.")

        client_info = self.clients[model_id]
        config = client_info['config']

        # Use provided params or defaults from config
        temp = temperature if temperature is not None else config['temperature']
        max_tok = max_tokens if max_tokens is not None else config['max_tokens']

        try:
            if client_info['type'] in ['openai', 'openai_compatible']:
                response = client_info['client'].chat.completions.create(
                    model=config['model_name'],
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok
                )
                return response.choices[0].message.content

            elif client_info['type'] == 'ollama':
                response = requests.post(
                    f"{client_info['base_url']}/api/chat",
                    json={
                        "model": config['model_name'],
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": temp,
                            "num_predict": max_tok
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()
                # Ollama returns message as a dict with 'content' and optionally 'thinking'
                message = result.get('message', {})
                return message.get('content', '')

        except Exception as e:
            raise RuntimeError(f"Error generating from {model_id}: {str(e)}")

    def get_available_models(self) -> List[str]:
        """Return list of successfully initialized models"""
        return list(self.clients.keys())

    def test_connection(self, model_id: str) -> bool:
        """Test if model is accessible"""
        try:
            response = self.generate(
                model_id=model_id,
                messages=[{"role": "user", "content": "Say hi"}],
                max_tokens=50
            )
            # Response can be empty string or None, both are considered failures
            return response is not None
        except Exception as e:
            print(f"Connection test failed for {model_id}: {str(e)}")
            return False


if __name__ == "__main__":
    # Test script
    print("Testing LLM Client initialization...")
    client = LLMClient()

    print(f"\nAvailable models: {client.get_available_models()}")

    for model_id in client.get_available_models():
        print(f"\nTesting {model_id}...")
        if client.test_connection(model_id):
            print(f"✓ {model_id} is working")
        else:
            print(f"✗ {model_id} failed")
