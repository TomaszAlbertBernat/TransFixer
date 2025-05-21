from typing import Dict, Any

# Default configuration for Ollama service
OLLAMA_CONFIG: Dict[str, Any] = {
    "model": "mistral",  # Default model to use
    "api_base": "http://localhost:11434",  # Default Ollama API endpoint
    "temperature": 0.7,  # Controls randomness in generation
    "top_p": 0.9,  # Nucleus sampling parameter
    "max_tokens": 2048,  # Maximum tokens to generate
    "presence_penalty": 0.0,  # Penalty for token presence
    "frequency_penalty": 0.0,  # Penalty for token frequency
    "system_prompt": """You are a professional text correction assistant. Your task is to:
1. Fix any spelling, grammar, or punctuation errors
2. Improve sentence structure and flow
3. Maintain the original meaning and tone
4. Preserve any technical terms or proper nouns
5. Keep the original formatting and structure

Please correct the following text while maintaining its original meaning:""",
    "retry_attempts": 3,  # Number of retry attempts for failed API calls
    "retry_delay": 1.0,  # Delay between retry attempts in seconds
    "timeout": 30.0,  # API request timeout in seconds
    "batch_size": 1,  # Number of texts to process in parallel
    "chunk_size": 1000,  # Maximum characters per chunk for long texts
    "chunk_overlap": 100,  # Number of characters to overlap between chunks
}

def validate_ollama_config(config: Dict[str, Any]) -> bool:
    """
    Validate Ollama configuration settings.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_keys = [
        "model", "api_base", "temperature", "top_p",
        "max_tokens", "system_prompt", "retry_attempts",
        "retry_delay", "timeout", "batch_size",
        "chunk_size", "chunk_overlap"
    ]
    
    # Check all required keys are present
    if not all(key in config for key in required_keys):
        return False
    
    # Validate numeric parameters
    if not (0 <= config["temperature"] <= 1):
        return False
    
    if not (0 <= config["top_p"] <= 1):
        return False
    
    if config["max_tokens"] <= 0:
        return False
    
    if config["retry_attempts"] < 0:
        return False
    
    if config["retry_delay"] < 0:
        return False
    
    if config["timeout"] <= 0:
        return False
    
    if config["batch_size"] <= 0:
        return False
    
    if config["chunk_size"] <= 0:
        return False
    
    if config["chunk_overlap"] < 0:
        return False
    
    if config["chunk_overlap"] >= config["chunk_size"]:
        return False
    
    return True

def get_optimal_chunk_size(text_length: int) -> int:
    """
    Calculate optimal chunk size based on text length.
    
    Args:
        text_length: Length of the text to process
        
    Returns:
        int: Optimal chunk size
    """
    if text_length <= 1000:
        return text_length
    
    if text_length <= 5000:
        return 1000
    
    if text_length <= 10000:
        return 2000
    
    return 3000

def get_optimal_batch_size(text_length: int) -> int:
    """
    Calculate optimal batch size based on text length.
    
    Args:
        text_length: Length of the text to process
        
    Returns:
        int: Optimal batch size
    """
    if text_length <= 1000:
        return 4
    
    if text_length <= 5000:
        return 2
    
    return 1 