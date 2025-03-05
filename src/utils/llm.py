"""
LLM client wrappers for different providers.
Provides a unified interface for different LLM providers.
"""

import os
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMClientFactory:
    """Factory for creating LLM clients based on provider."""
    
    @staticmethod
    def create_client(provider: str = "openai", model: Optional[str] = None) -> Any:
        """
        Create an LLM client based on the provider.
        
        Args:
            provider: The LLM provider (openai, groq, etc.)
            model: The specific model to use
            
        Returns:
            An LLM client instance
        """
        provider = provider.lower()
        
        if provider == "openai":
            return OpenAIClient(model=model)
        elif provider == "groq":
            return GroqClient(model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class BaseClient:
    """Base class for LLM clients."""
    
    def __init__(self, model: Optional[str] = None):
        self.model = model
        
    def get_completion(self, 
                       prompt: str, 
                       system_message: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None) -> str:
        """
        Get a completion from the LLM.
        
        Args:
            prompt: The user prompt
            system_message: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            The generated text
        """
        raise NotImplementedError("Subclasses must implement get_completion")
    
    def get_chat_completion(self, 
                           messages: List[Dict[str, str]],
                           temperature: float = 0.7,
                           max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Get a chat completion from the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with response and metadata
        """
        raise NotImplementedError("Subclasses must implement get_chat_completion")
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            The number of tokens
        """
        # Simple approximation - in production, use model-specific tokenizers
        return len(text.split())


class OpenAIClient(BaseClient):
    """Client for OpenAI API."""
    
    def __init__(self, model: Optional[str] = None):
        super().__init__(model)
        try:
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
    
    def get_completion(self, 
                       prompt: str, 
                       system_message: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None) -> str:
        """Get completion from OpenAI."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def get_chat_completion(self, 
                           messages: List[Dict[str, str]],
                           temperature: float = 0.7,
                           max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Get chat completion from OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        result = {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
        return result


class GroqClient(BaseClient):
    """Client for Groq API."""
    
    def __init__(self, model: Optional[str] = None):
        super().__init__(model)
        try:
            import groq
            self.client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
            self.model = model or "llama3-8b-8192"
        except ImportError:
            raise ImportError("Groq package not installed. Install with 'pip install groq'")
    
    def get_completion(self, 
                       prompt: str, 
                       system_message: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None) -> str:
        """Get completion from Groq."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def get_chat_completion(self, 
                           messages: List[Dict[str, str]],
                           temperature: float = 0.7,
                           max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Get chat completion from Groq."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        result = {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
        return result 