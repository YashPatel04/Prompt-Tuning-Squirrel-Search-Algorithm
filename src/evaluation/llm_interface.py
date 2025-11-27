import requests
import json
from typing import List, Dict, Optional
import time

class OllamaInterface:
    """
    Interface for communicating with Ollama LLMs locally.
    """

    def __init__(self, base_url="http://localhost:1561", model="deepseek-r1:1.5b", temperature=0.0, timeout=300):
        """
        Initialize Ollama interface

        Args:
            base_url: Ollama server URL (default: localhost:1561)
            model: Model name
            temperature: Response temperature (0.0 = deterministic)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.api_endpoint = f"{base_url}/api/generate"
        
        # Check connection
        self._check_connection()

    def _check_connection(self):
        """Check if Ollama server if running"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )

            if response.status_code == 200:
                print(f"Connected to Ollama at {self.base_url}")
                self._list_available_models()
                return True
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}\n"
                f"Make sure Ollama is running: ollama serve"
            )
        except Exception as e:
            raise Exception(f"Connection error: {e}")
    
    def _list_available_models(self):
        """Print available models on server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                print(f"Available models: {', '.join(model_names)}")
        except Exception as e:
            print(f"Could not list models: {e}")
    
    def generate(self, prompt):
        """
        Generate response from Ollama.
        
        Args:
            prompt: Input prompt
        
        Returns:
            Generated text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
            "top_p": 0.9,
            "top_k": 40,
        }
        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get('response', '').strip()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timeout after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama request failed: {e}")
    
    def batch_generate(self, prompts, show_progress=True):
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            show_progress: Show progress bar
        
        Returns:
            List of generated responses
        """
        responses = []
        for i, prompt in enumerate(prompts):
            if show_progress:
                print(f"[{i+1}/{len(prompts)}] Generating response...")
            response = self.generate(prompt)
            responses.append(response)
            # Small delay to avoid overwhelming server
            time.sleep(0.1)
        return responses
    
    def __str__(self) -> str:
        return f"OllamaInterface(model={self.model}, base_url={self.base_url})"