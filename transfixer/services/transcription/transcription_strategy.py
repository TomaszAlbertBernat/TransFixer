from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

class TranscriptionStrategy(ABC):
    """Abstract base class for transcription strategies."""
    
    def __init__(self, model, processor, config: Dict[str, Any]):
        self.model = model
        self.processor = processor
        self.config = config
        self.pipeline = None
        self._initialize_pipeline()
    
    @abstractmethod
    def _initialize_pipeline(self) -> None:
        """Initialize the transcription pipeline with strategy-specific settings."""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file using the specific strategy.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict containing transcription result and metadata
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up resources used by the strategy."""
        if self.pipeline:
            del self.pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class SequentialStrategy(TranscriptionStrategy):
    """Sequential transcription strategy using sliding window approach."""
    
    def _initialize_pipeline(self) -> None:
        """Initialize pipeline for sequential transcription."""
        logger.info("Initializing sequential transcription pipeline")
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=self.config["chunk_length_s"],
            batch_size=self.config["batch_size"],
            torch_dtype=self.config["torch_dtype"],
            device=self.config["device"],
            generate_kwargs=self.config.get("generate_kwargs", {})
        )
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe using sequential strategy."""
        try:
            result = self.pipeline(audio_path)
            return {
                "transcription": result.get("text", ""),
                "error": None,
                "metadata": {
                    "strategy": "sequential",
                    "chunk_length": self.config["chunk_length_s"]
                }
            }
        except Exception as e:
            logger.error(f"Error in sequential transcription: {e}")
            return {
                "transcription": None,
                "error": str(e),
                "metadata": {
                    "strategy": "sequential",
                    "chunk_length": self.config["chunk_length_s"]
                }
            }

class ChunkedStrategy(TranscriptionStrategy):
    """Chunked transcription strategy for faster processing."""
    
    def _initialize_pipeline(self) -> None:
        """Initialize pipeline for chunked transcription."""
        logger.info("Initializing chunked transcription pipeline")
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=self.config["chunk_length_s"],
            batch_size=self.config["batch_size"],
            torch_dtype=self.config["torch_dtype"],
            device=self.config["device"],
            generate_kwargs=self.config.get("generate_kwargs", {})
        )
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe using chunked strategy."""
        try:
            result = self.pipeline(audio_path)
            return {
                "transcription": result.get("text", ""),
                "error": None,
                "metadata": {
                    "strategy": "chunked",
                    "chunk_length": self.config["chunk_length_s"]
                }
            }
        except Exception as e:
            logger.error(f"Error in chunked transcription: {e}")
            return {
                "transcription": None,
                "error": str(e),
                "metadata": {
                    "strategy": "chunked",
                    "chunk_length": self.config["chunk_length_s"]
                }
            }

def create_strategy(config: Dict[str, Any], model, processor) -> TranscriptionStrategy:
    """
    Factory function to create the appropriate transcription strategy.
    
    Args:
        config: Configuration dictionary
        model: Initialized Whisper model
        processor: Initialized Whisper processor
        
    Returns:
        An instance of the appropriate TranscriptionStrategy
    """
    strategy_name = config["transcription_strategy"]
    if strategy_name == "sequential":
        return SequentialStrategy(model, processor, config)
    elif strategy_name == "chunked":
        return ChunkedStrategy(model, processor, config)
    else:
        raise ValueError(f"Unknown transcription strategy: {strategy_name}") 