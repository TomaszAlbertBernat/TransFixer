import logging
import time
import json
import hashlib
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from ...config.ollama_config import (
    OLLAMA_CONFIG,
    validate_ollama_config,
    get_optimal_chunk_size,
    get_optimal_batch_size
)
from .text_analyzer import TextAnalyzer
from collections import defaultdict

logger = logging.getLogger(__name__)

class TextProcessingStats:
    """Class to track text processing statistics."""
    
    def __init__(self):
        self.total_chunks = 0
        self.processed_chunks = 0
        self.failed_chunks = 0
        self.total_tokens = 0
        self.processing_time = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.retry_count = 0
        self.error_count = 0
        self.quality_improvements = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "failed_chunks": self.failed_chunks,
            "total_tokens": self.total_tokens,
            "processing_time": self.processing_time,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "retry_count": self.retry_count,
            "error_count": self.error_count,
            "success_rate": (self.processed_chunks / self.total_chunks * 100) if self.total_chunks > 0 else 0,
            "average_quality_improvement": sum(self.quality_improvements) / len(self.quality_improvements) if self.quality_improvements else 0
        }

class OllamaService:
    """Service for handling text correction using Ollama API."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, cache_dir: Optional[str] = None):
        """
        Initialize OllamaService with configuration.
        
        Args:
            config: Optional configuration dictionary. If not provided, uses default config.
            cache_dir: Optional directory for caching results
        """
        self.config = config or OLLAMA_CONFIG.copy()
        if not validate_ollama_config(self.config):
            raise ValueError("Invalid Ollama configuration")
        
        self.session = requests.Session()
        self.session.timeout = self.config["timeout"]
        
        # Initialize cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize stats and analyzer
        self.stats = TextProcessingStats()
        self.analyzer = TextAnalyzer()
        
        # Quality control thresholds
        self.quality_thresholds = {
            "min_quality_score": 0.7,
            "max_grammar_issues": 5,
            "min_lexical_diversity": 0.5,
            "min_coherence_score": 0.6,
            "max_passive_voice_ratio": 0.3
        }
        
        # Initialize error tracking
        self._error_counts = defaultdict(int)
        self._last_error_time = 0
        self._error_backoff = 1.0
    
    def _get_cache_path(self, text: str) -> Optional[Path]:
        """Get cache file path for text."""
        if not self.cache_dir:
            return None
        
        # Create hash of text for cache key
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return self.cache_dir / f"{text_hash}.json"
    
    def _load_from_cache(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Load corrected text and metadata from cache."""
        cache_path = self._get_cache_path(text)
        if not cache_path or not cache_path.exists():
            self.stats.cache_misses += 1
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                if cached_data.get('text') == text:  # Verify text matches
                    self.stats.cache_hits += 1
                    return cached_data.get('corrected_text'), cached_data.get('metadata', {})
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
        
        self.stats.cache_misses += 1
        return None
    
    def _save_to_cache(self, text: str, corrected_text: str, metadata: Dict[str, Any]) -> None:
        """Save corrected text and metadata to cache."""
        cache_path = self._get_cache_path(text)
        if not cache_path:
            return
        
        try:
            cache_data = {
                'text': text,
                'corrected_text': corrected_text,
                'metadata': metadata,
                'timestamp': time.time()
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def _handle_api_error(self, error: Exception) -> None:
        """Handle API errors with exponential backoff."""
        current_time = time.time()
        error_type = type(error).__name__
        
        # Update error counts
        self._error_counts[error_type] += 1
        self.stats.error_count += 1
        
        # Calculate backoff time
        if current_time - self._last_error_time < 60:  # Within last minute
            self._error_backoff *= 2  # Double the backoff time
        else:
            self._error_backoff = 1.0  # Reset backoff
        
        self._last_error_time = current_time
        
        # Log error with backoff information
        logger.warning(
            f"API error ({error_type}): {error}. "
            f"Backoff: {self._error_backoff}s, "
            f"Count: {self._error_counts[error_type]}"
        )
        
        # Sleep with backoff
        time.sleep(self._error_backoff)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before correction.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Ensure proper spacing around punctuation
        text = text.replace(' .', '.').replace(' ,', ',')
        text = text.replace(' !', '!').replace(' ?', '?')
        
        return text
    
    def _postprocess_text(self, text: str) -> str:
        """
        Postprocess text after correction.
        
        Args:
            text: Text to postprocess
            
        Returns:
            Postprocessed text
        """
        # Restore proper spacing
        text = text.replace(' .', '.').replace(' ,', ',')
        text = text.replace(' !', '!').replace(' ?', '?')
        
        # Ensure proper paragraph spacing
        text = text.replace('\n\n\n', '\n\n')
        
        return text
    
    def _make_api_request(self, prompt: str) -> Dict[str, Any]:
        """
        Make a request to the Ollama API with improved error handling.
        
        Args:
            prompt: Text to send to the API
            
        Returns:
            Dict containing API response
        """
        url = f"{self.config['api_base']}/api/generate"
        
        payload = {
            "model": self.config["model"],
            "prompt": f"{self.config['system_prompt']}\n\n{prompt}",
            "temperature": self.config["temperature"],
            "top_p": self.config["top_p"],
            "max_tokens": self.config["max_tokens"],
            "presence_penalty": self.config["presence_penalty"],
            "frequency_penalty": self.config["frequency_penalty"]
        }
        
        for attempt in range(self.config["retry_attempts"]):
            try:
                response = self.session.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                self._handle_api_error(e)
                if attempt == self.config["retry_attempts"] - 1:
                    raise
    
    def _process_chunk(self, chunk: str) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Process a single chunk of text with enhanced error handling and metadata.
        
        Args:
            chunk: Text chunk to process
            
        Returns:
            Tuple of (corrected text, success flag, metadata)
        """
        try:
            # Check cache first
            cached_result = self._load_from_cache(chunk)
            if cached_result is not None:
                return cached_result[0], True, cached_result[1]
            
            # Process chunk
            response = self._make_api_request(chunk)
            corrected_text = response.get("response", chunk)
            
            # Analyze and collect metadata
            metadata = {
                "original_analysis": self.analyzer.analyze_text(chunk),
                "corrected_analysis": self.analyzer.analyze_text(corrected_text),
                "quality_score": self.analyzer.get_quality_score(corrected_text)
            }
            
            # Save to cache
            self._save_to_cache(chunk, corrected_text, metadata)
            
            return corrected_text, True, metadata
        
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return chunk, False, {}
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunk_size = get_optimal_chunk_size(len(text))
        if chunk_size >= len(text):
            return [text]
        
        chunks = []
        overlap = self.config["chunk_overlap"]
        
        # Split on sentence boundaries when possible
        sentences = text.split('. ')
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '. '
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks
    
    def _process_batch(self, texts: List[str], show_progress: bool = True) -> List[Tuple[str, bool, Dict[str, Any]]]:
        """
        Process a batch of texts in parallel with improved error handling.
        
        Args:
            texts: List of texts to process
            show_progress: Whether to show progress bar
            
        Returns:
            List of tuples containing (corrected text, success flag, metadata)
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.config["batch_size"]) as executor:
            future_to_text = {
                executor.submit(self._process_chunk, text): text 
                for text in texts
            }
            
            if show_progress:
                futures = tqdm(
                    as_completed(future_to_text),
                    total=len(texts),
                    desc="Processing chunks"
                )
            else:
                futures = as_completed(future_to_text)
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    results.append((future_to_text[future], False, {}))
        
        return results
    
    def correct_text(self, text: str, show_progress: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Correct text using Ollama API with enhanced quality control.
        
        Args:
            text: Text to correct
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (corrected text, processing statistics)
        """
        if not text.strip():
            return text, self.stats.to_dict()
        
        start_time = time.time()
        self.stats = TextProcessingStats()  # Reset stats
        
        try:
            # Analyze original text
            original_analysis = self.analyzer.analyze_text(text)
            logger.info(f"Original text analysis: {json.dumps(original_analysis, indent=2)}")
            
            # Preprocess text
            text = self._preprocess_text(text)
            
            # Split text into chunks
            chunks = self._split_text(text)
            self.stats.total_chunks = len(chunks)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Process chunks in batches
            batch_size = get_optimal_batch_size(len(text))
            corrected_chunks = []
            chunk_metadata = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_results = self._process_batch(batch, show_progress)
                
                for corrected_text, success, metadata in batch_results:
                    corrected_chunks.append(corrected_text)
                    chunk_metadata.append(metadata)
                    if success:
                        self.stats.processed_chunks += 1
                    else:
                        self.stats.failed_chunks += 1
            
            # Join chunks and postprocess
            corrected_text = self._postprocess_text(" ".join(corrected_chunks))
            
            # Analyze corrected text
            corrected_analysis = self.analyzer.analyze_text(corrected_text)
            comparison = self.analyzer.compare_texts(text, corrected_text)
            
            # Quality control
            quality_score = self.analyzer.get_quality_score(corrected_text)
            quality_issues = self._check_quality_issues(corrected_analysis)
            
            if quality_issues:
                logger.warning(f"Quality issues detected: {quality_issues}")
                if self._should_retry_correction(quality_issues):
                    logger.info("Retrying correction with adjusted parameters...")
                    self.stats.retry_count += 1
                    return self._retry_correction(text, quality_issues)
            
            # Calculate quality improvement
            original_score = self.analyzer.get_quality_score(text)
            quality_improvement = quality_score - original_score
            self.stats.quality_improvements.append(quality_improvement)
            
            # Update stats
            self.stats.processing_time = time.time() - start_time
            self.stats.total_tokens = len(corrected_text.split())
            
            # Add analysis results to stats
            stats_dict = self.stats.to_dict()
            stats_dict.update({
                "quality_score": quality_score,
                "quality_improvement": quality_improvement,
                "quality_issues": quality_issues,
                "text_analysis": {
                    "original": original_analysis,
                    "corrected": corrected_analysis,
                    "comparison": comparison
                },
                "chunk_metadata": chunk_metadata
            })
            
            return corrected_text, stats_dict
        
        except Exception as e:
            logger.error(f"Error correcting text: {e}")
            self.stats.processing_time = time.time() - start_time
            return text, self.stats.to_dict()
    
    def _check_quality_issues(self, analysis: Dict[str, Any]) -> List[str]:
        """Check for quality issues in the corrected text."""
        issues = []
        
        if analysis["grammar_issues"] > self.quality_thresholds["max_grammar_issues"]:
            issues.append(f"Too many grammar issues: {analysis['grammar_issues']}")
        
        if analysis.get("lexical_diversity", 0) < self.quality_thresholds["min_lexical_diversity"]:
            issues.append(f"Low lexical diversity: {analysis['lexical_diversity']:.2f}")
        
        if analysis.get("coherence_score", 0) < self.quality_thresholds["min_coherence_score"]:
            issues.append(f"Low coherence score: {analysis['coherence_score']:.2f}")
        
        if analysis.get("passive_voice_ratio", 0) > self.quality_thresholds["max_passive_voice_ratio"]:
            issues.append(f"High passive voice ratio: {analysis['passive_voice_ratio']:.2f}")
        
        return issues
    
    def _should_retry_correction(self, quality_issues: List[str]) -> bool:
        """Determine if correction should be retried."""
        # Retry if there are grammar issues or very low quality
        return (
            any("grammar" in issue.lower() for issue in quality_issues) or
            any("coherence" in issue.lower() for issue in quality_issues)
        )
    
    def _retry_correction(self, text: str, quality_issues: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Retry correction with adjusted parameters."""
        # Adjust parameters based on quality issues
        original_config = self.config.copy()
        
        try:
            # Adjust parameters based on specific issues
            if any("grammar" in issue.lower() for issue in quality_issues):
                self.config.update({
                    "temperature": 0.5,  # Lower temperature for more conservative corrections
                    "top_p": 0.8,  # More focused sampling
                })
            
            if any("coherence" in issue.lower() for issue in quality_issues):
                self.config.update({
                    "presence_penalty": 0.2,  # Encourage more diverse vocabulary
                    "frequency_penalty": 0.2  # Discourage repetition
                })
            
            # Retry correction
            return self.correct_text(text, show_progress=True)
        
        finally:
            # Restore original configuration
            self.config = original_config
    
    def get_correction_suggestions(self, text: str) -> List[Dict[str, Any]]:
        """
        Get detailed correction suggestions for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of correction suggestions
        """
        analysis = self.analyzer.analyze_text(text)
        suggestions = []
        
        # Grammar suggestions
        if analysis["grammar_issues"] > 0:
            suggestions.append({
                "type": "grammar",
                "description": f"Found {analysis['grammar_issues']} grammar issues",
                "categories": analysis["grammar_categories"]
            })
        
        # Readability suggestions
        if analysis["flesch_score"] < 60:
            suggestions.append({
                "type": "readability",
                "description": "Text may be difficult to read",
                "score": analysis["flesch_score"],
                "suggestions": [
                    "Consider using shorter sentences",
                    "Use simpler vocabulary where possible",
                    "Break down complex ideas into smaller parts"
                ]
            })
        
        # Vocabulary suggestions
        if analysis.get("lexical_diversity", 0) < 0.5:
            suggestions.append({
                "type": "vocabulary",
                "description": "Consider using more diverse vocabulary",
                "diversity_score": analysis["lexical_diversity"],
                "suggestions": [
                    "Use synonyms to avoid repetition",
                    "Incorporate more varied word choices",
                    "Consider using more specific terms"
                ]
            })
        
        # Style suggestions
        if analysis.get("passive_voice_ratio", 0) > 0.3:
            suggestions.append({
                "type": "style",
                "description": "High use of passive voice",
                "ratio": analysis["passive_voice_ratio"],
                "suggestions": [
                    "Convert passive constructions to active voice",
                    "Make the subject of the sentence more prominent",
                    "Use stronger, more direct verbs"
                ]
            })
        
        return suggestions
    
    def analyze_correction_impact(self, original: str, corrected: str) -> Dict[str, Any]:
        """
        Analyze the impact of corrections on text quality.
        
        Args:
            original: Original text
            corrected: Corrected text
            
        Returns:
            Dictionary containing impact analysis
        """
        comparison = self.analyzer.compare_texts(original, corrected)
        original_score = self.analyzer.get_quality_score(original)
        corrected_score = self.analyzer.get_quality_score(corrected)
        
        # Calculate improvement percentages
        improvements = {}
        for metric, values in comparison["differences"].items():
            if isinstance(values, (int, float)):
                original_value = comparison["original_metrics"].get(metric, 0)
                if original_value != 0:
                    improvements[metric] = (values / original_value) * 100
        
        return {
            "quality_improvement": corrected_score - original_score,
            "improvement_percentages": improvements,
            "metrics_comparison": comparison,
            "suggestions": self.get_correction_suggestions(corrected)
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup() 