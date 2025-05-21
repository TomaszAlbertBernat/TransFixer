import re
from typing import Dict, Any, List, Tuple, Set
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
import language_tool_python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TextAnalyzer:
    """Advanced text analysis and quality metrics."""
    
    def __init__(self):
        """Initialize TextAnalyzer with required NLTK resources."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')
        
        self.stopwords = set(stopwords.words('english'))
        self.language_tool = language_tool_python.LanguageTool('en-US')
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Initialize word complexity cache
        self._word_complexity_cache = {}
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive text analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        tagged_words = pos_tag(words)
        
        # Basic metrics
        metrics = {
            "sentence_count": len(sentences),
            "word_count": len(words),
            "unique_word_count": len(set(words)),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "stopword_ratio": len([w for w in words if w in self.stopwords]) / len(words) if words else 0
        }
        
        # Readability scores
        metrics.update(self._calculate_readability_scores(text))
        
        # Grammar and style analysis
        grammar_issues = self.language_tool.check(text)
        metrics.update({
            "grammar_issues": len(grammar_issues),
            "grammar_categories": self._categorize_grammar_issues(grammar_issues)
        })
        
        # Vocabulary analysis
        metrics.update(self._analyze_vocabulary(words, tagged_words))
        
        # Text structure analysis
        metrics.update(self._analyze_structure(sentences, text))
        
        # Semantic analysis
        metrics.update(self._analyze_semantics(sentences))
        
        # Style analysis
        metrics.update(self._analyze_style(text, sentences, tagged_words))
        
        return metrics
    
    def _calculate_readability_scores(self, text: str) -> Dict[str, float]:
        """Calculate various readability scores."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Calculate syllables (simple approximation)
        syllables = sum(self._count_syllables(word) for word in words)
        
        # Flesch Reading Ease
        flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * (syllables / len(words)) if words else 0
        
        # Flesch-Kincaid Grade Level
        fk_grade = 0.39 * avg_sentence_length + 11.8 * (syllables / len(words)) - 15.59 if words else 0
        
        # SMOG Index
        complex_sentences = len([s for s in sentences if self._count_syllables(s) > 3])
        smog_score = 1.043 * np.sqrt(complex_sentences * (30 / len(sentences))) + 3.1291 if sentences else 0
        
        return {
            "flesch_score": flesch_score,
            "flesch_kincaid_grade": fk_grade,
            "smog_index": smog_score
        }
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count for a word."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        previous_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_is_vowel:
                count += 1
            previous_is_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        return max(1, count)
    
    def _categorize_grammar_issues(self, issues: List[Any]) -> Dict[str, int]:
        """Categorize grammar issues by type."""
        categories = Counter()
        for issue in issues:
            categories[issue.category] += 1
        return dict(categories)
    
    def _analyze_vocabulary(self, words: List[str], tagged_words: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Analyze vocabulary complexity and diversity."""
        # Remove stopwords and punctuation
        content_words = [w for w in words if w.isalpha() and w not in self.stopwords]
        
        # Word frequency
        word_freq = Counter(content_words)
        
        # Calculate lexical diversity
        unique_words = len(word_freq)
        total_words = len(content_words)
        lexical_diversity = unique_words / total_words if total_words > 0 else 0
        
        # Part of speech distribution
        pos_distribution = Counter(tag for _, tag in tagged_words)
        
        # Word complexity analysis
        complex_words = [w for w in content_words if self._get_word_complexity(w) > 0.7]
        complexity_ratio = len(complex_words) / len(content_words) if content_words else 0
        
        return {
            "lexical_diversity": lexical_diversity,
            "common_words": dict(word_freq.most_common(10)),
            "content_word_ratio": len(content_words) / len(words) if words else 0,
            "pos_distribution": dict(pos_distribution),
            "complexity_ratio": complexity_ratio,
            "complex_words": complex_words[:10]  # Top 10 complex words
        }
    
    def _get_word_complexity(self, word: str) -> float:
        """Calculate word complexity score."""
        if word in self._word_complexity_cache:
            return self._word_complexity_cache[word]
        
        # Factors affecting complexity
        length_factor = min(1.0, len(word) / 12)  # Longer words are more complex
        syllable_factor = min(1.0, self._count_syllables(word) / 4)  # More syllables = more complex
        
        # Check if word is in WordNet
        synsets = wordnet.synsets(word)
        frequency_factor = 1.0 - (len(synsets) / 20) if synsets else 1.0  # More meanings = more complex
        
        complexity = (length_factor + syllable_factor + frequency_factor) / 3
        self._word_complexity_cache[word] = complexity
        return complexity
    
    def _analyze_structure(self, sentences: List[str], text: str) -> Dict[str, Any]:
        """Analyze text structure and patterns."""
        # Sentence length distribution
        lengths = [len(word_tokenize(s)) for s in sentences]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        
        # Paragraph detection
        paragraphs = text.split('\n\n')
        paragraph_count = len(paragraphs)
        
        # Sentence type analysis
        sentence_types = {
            "declarative": len([s for s in sentences if s.strip().endswith('.')]),
            "interrogative": len([s for s in sentences if s.strip().endswith('?')]),
            "exclamatory": len([s for s in sentences if s.strip().endswith('!')])
        }
        
        # Coherence analysis
        coherence_score = self._calculate_coherence(paragraphs)
        
        return {
            "avg_sentence_length": avg_length,
            "paragraph_count": paragraph_count,
            "sentence_types": sentence_types,
            "coherence_score": coherence_score,
            "length_distribution": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "std_dev": np.std(lengths) if lengths else 0
            }
        }
    
    def _calculate_coherence(self, paragraphs: List[str]) -> float:
        """Calculate text coherence score."""
        if not paragraphs:
            return 0.0
        
        try:
            # Calculate TF-IDF for paragraphs
            tfidf_matrix = self.tfidf.fit_transform(paragraphs)
            
            # Calculate cosine similarity between adjacent paragraphs
            similarities = []
            for i in range(len(paragraphs) - 1):
                similarity = (tfidf_matrix[i] * tfidf_matrix[i + 1].T).toarray()[0][0]
                similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
        except:
            return 0.0
    
    def _analyze_semantics(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze semantic aspects of the text."""
        # Topic modeling (simple version)
        topics = defaultdict(int)
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            for word in words:
                if word.isalpha() and word not in self.stopwords:
                    synsets = wordnet.synsets(word)
                    if synsets:
                        topics[synsets[0].lexname()] += 1
        
        return {
            "topic_distribution": dict(topics),
            "semantic_density": len(topics) / len(sentences) if sentences else 0
        }
    
    def _analyze_style(self, text: str, sentences: List[str], tagged_words: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Analyze writing style."""
        # Passive voice detection
        passive_count = len(re.findall(r'\b(am|is|are|was|were|be|been|being)\s+\w+ed\b', text.lower()))
        
        # Sentence variety
        sentence_lengths = [len(word_tokenize(s)) for s in sentences]
        length_variety = np.std(sentence_lengths) / np.mean(sentence_lengths) if sentence_lengths else 0
        
        # Formality analysis
        formal_markers = len(re.findall(r'\b(however|therefore|furthermore|moreover|consequently)\b', text.lower()))
        
        return {
            "passive_voice_ratio": passive_count / len(sentences) if sentences else 0,
            "sentence_variety": length_variety,
            "formality_score": min(1.0, formal_markers / 5),
            "style_markers": {
                "passive_voice": passive_count,
                "formal_markers": formal_markers,
                "sentence_variety": length_variety
            }
        }
    
    def compare_texts(self, original: str, corrected: str) -> Dict[str, Any]:
        """
        Compare original and corrected texts.
        
        Args:
            original: Original text
            corrected: Corrected text
            
        Returns:
            Dictionary containing comparison metrics
        """
        original_metrics = self.analyze_text(original)
        corrected_metrics = self.analyze_text(corrected)
        
        # Calculate differences
        differences = {}
        for key in original_metrics:
            if isinstance(original_metrics[key], (int, float)):
                differences[key] = corrected_metrics[key] - original_metrics[key]
        
        return {
            "original_metrics": original_metrics,
            "corrected_metrics": corrected_metrics,
            "differences": differences
        }
    
    def get_quality_score(self, text: str) -> float:
        """
        Calculate overall text quality score.
        
        Args:
            text: Text to evaluate
            
        Returns:
            float: Quality score between 0 and 1
        """
        metrics = self.analyze_text(text)
        
        # Enhanced weights for different factors
        weights = {
            "grammar_issues": -0.25,  # Negative weight for issues
            "lexical_diversity": 0.15,
            "flesch_score": 0.15,
            "content_word_ratio": 0.1,
            "avg_sentence_length": 0.1,
            "coherence_score": 0.1,
            "complexity_ratio": 0.05,
            "sentence_variety": 0.05,
            "formality_score": 0.05
        }
        
        # Calculate weighted score
        score = 0.0
        for factor, weight in weights.items():
            if factor in metrics:
                # Normalize the metric to 0-1 range
                if factor == "grammar_issues":
                    normalized = max(0, 1 - (metrics[factor] / 10))
                elif factor == "flesch_score":
                    normalized = max(0, min(1, metrics[factor] / 100))
                elif factor == "avg_sentence_length":
                    normalized = max(0, min(1, 1 - abs(metrics[factor] - 15) / 30))
                else:
                    normalized = metrics[factor]
                
                score += normalized * weight
        
        return max(0, min(1, score + 0.5))  # Ensure score is between 0 and 1 