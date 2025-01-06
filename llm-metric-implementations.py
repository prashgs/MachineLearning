import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import re
from scipy.stats import entropy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import torch
from transformers import AutoTokenizer, AutoModel

class DetailedMetricsImplementation:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))
        # Load pre-trained model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def get_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings for text using transformer model"""
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    #######################
    # 1. Quality Metrics
    #######################
    
    def measure_coherence(self, text: str) -> float:
        """
        Measure text coherence using multiple approaches:
        1. Sentence transition analysis
        2. Topic consistency
        3. Discourse markers
        """
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 1.0  # Single sentence is considered coherent
            
        # 1. Sentence transition coherence
        transition_score = self._evaluate_transitions(sentences)
        
        # 2. Topic consistency
        topic_score = self._evaluate_topic_consistency(sentences)
        
        # 3. Discourse markers
        discourse_score = self._evaluate_discourse_markers(text)
        
        # Combine scores with weights
        weights = [0.4, 0.4, 0.2]
        final_score = (
            weights[0] * transition_score +
            weights[1] * topic_score +
            weights[2] * discourse_score
        )
        
        return min(1.0, max(0.0, final_score))

    def _evaluate_transitions(self, sentences: List[str]) -> float:
        """Evaluate coherence of transitions between sentences"""
        transition_scores = []
        
        for i in range(len(sentences) - 1):
            # Get embeddings for consecutive sentences
            emb1 = self.get_embeddings(sentences[i])
            emb2 = self.get_embeddings(sentences[i + 1])
            
            # Calculate cosine similarity between consecutive sentences
            similarity = cosine_similarity(emb1, emb2)[0][0]
            transition_scores.append(similarity)
            
        return np.mean(transition_scores) if transition_scores else 0.0

    def _evaluate_topic_consistency(self, sentences: List[str]) -> float:
        """Evaluate consistency of topics across sentences"""
        # Extract main topics (nouns and named entities) from each sentence
        topics_per_sentence = []
        
        for sent in sentences:
            doc = self.nlp(sent)
            topics = set()
            
            # Add nouns
            topics.update([token.text.lower() for token in doc if token.pos_ == 'NOUN'])
            
            # Add named entities
            topics.update([ent.text.lower() for ent in doc.ents])
            
            topics_per_sentence.append(topics)
        
        # Calculate topic overlap between consecutive sentences
        overlap_scores = []
        for i in range(len(topics_per_sentence) - 1):
            set1 = topics_per_sentence[i]
            set2 = topics_per_sentence[i + 1]
            
            if not set1 or not set2:
                continue
                
            overlap = len(set1.intersection(set2)) / len(set1.union(set2))
            overlap_scores.append(overlap)
            
        return np.mean(overlap_scores) if overlap_scores else 0.0

    def measure_fluency(self, text: str) -> float:
        """
        Measure text fluency using:
        1. Grammar check
        2. Language model perplexity
        3. Sentence structure analysis
        """
        # 1. Grammar check using TextBlob
        blob = TextBlob(text)
        grammar_score = self._calculate_grammar_score(blob)
        
        # 2. Sentence structure analysis
        structure_score = self._analyze_sentence_structure(text)
        
        # 3. Word order naturalness
        naturalness_score = self._evaluate_word_order(text)
        
        # Combine scores
        weights = [0.4, 0.3, 0.3]
        final_score = (
            weights[0] * grammar_score +
            weights[1] * structure_score +
            weights[2] * naturalness_score
        )
        
        return min(1.0, max(0.0, final_score))

    def _calculate_grammar_score(self, blob: TextBlob) -> float:
        """Calculate grammar score using TextBlob"""
        # Count grammar errors (simplified approach)
        sentences = blob.sentences
        error_count = 0
        total_sentences = len(sentences)
        
        for sentence in sentences:
            tags = [tag for word, tag in sentence.tags]
            
            # Check for basic grammatical patterns
            if not self._has_valid_sentence_structure(tags):
                error_count += 1
                
        return 1.0 - (error_count / total_sentences if total_sentences > 0 else 0)

    def measure_relevance(self, response: str, prompt: str) -> float:
        """
        Measure relevance between response and prompt using:
        1. Semantic similarity
        2. Keyword overlap
        3. Context maintenance
        """
        # 1. Semantic similarity using embeddings
        prompt_emb = self.get_embeddings(prompt)
        response_emb = self.get_embeddings(response)
        semantic_score = cosine_similarity(prompt_emb, response_emb)[0][0]
        
        # 2. Keyword overlap
        keyword_score = self._calculate_keyword_overlap(prompt, response)
        
        # 3. Context maintenance
        context_score = self._evaluate_context_maintenance(prompt, response)
        
        # Combine scores
        weights = [0.5, 0.3, 0.2]
        final_score = (
            weights[0] * semantic_score +
            weights[1] * keyword_score +
            weights[2] * context_score
        )
        
        return min(1.0, max(0.0, final_score))

    #######################
    # 2. Content Metrics
    #######################

    def measure_complexity(self, text: str) -> float:
        """
        Measure text complexity using:
        1. Vocabulary sophistication
        2. Syntactic complexity
        3. Information density
        """
        # 1. Vocabulary sophistication
        vocab_score = self._measure_vocabulary_sophistication(text)
        
        # 2. Syntactic complexity
        syntax_score = self._measure_syntactic_complexity(text)
        
        # 3. Information density
        density_score = self._measure_information_density(text)
        
        # Combine scores
        weights = [0.4, 0.3, 0.3]
        final_score = (
            weights[0] * vocab_score +
            weights[1] * syntax_score +
            weights[2] * density_score
        )
        
        return min(1.0, max(0.0, final_score))

    def _measure_vocabulary_sophistication(self, text: str) -> float:
        """Measure vocabulary sophistication using word frequency analysis"""
        words = word_tokenize(text.lower())
        words = [w for w in words if w not in self.stop_words]
        
        # Calculate type-token ratio (lexical diversity)
        if not words:
            return 0.0
        type_token_ratio = len(set(words)) / len(words)
        
        return type_token_ratio

    def measure_diversity(self, text: str) -> float:
        """
        Measure text diversity using:
        1. Vocabulary diversity
        2. Expression diversity
        3. Structural diversity
        """
        # 1. Vocabulary diversity (using entropy)
        vocab_diversity = self._calculate_vocabulary_diversity(text)
        
        # 2. Expression diversity
        expr_diversity = self._calculate_expression_diversity(text)
        
        # 3. Structural diversity
        struct_diversity = self._calculate_structural_diversity(text)
        
        # Combine scores
        weights = [0.4, 0.3, 0.3]
        final_score = (
            weights[0] * vocab_diversity +
            weights[1] * expr_diversity +
            weights[2] * struct_diversity
        )
        
        return min(1.0, max(0.0, final_score))

    #######################
    # 3. Accuracy Metrics
    #######################

    def measure_factual_accuracy(self, response: str, reference: str) -> float:
        """
        Measure factual accuracy using:
        1. Named entity matching
        2. Fact triple comparison
        3. Numerical accuracy
        """
        # 1. Named entity accuracy
        entity_score = self._compare_named_entities(response, reference)
        
        # 2. Fact triple accuracy
        fact_score = self._compare_fact_triples(response, reference)
        
        # 3. Numerical accuracy
        number_score = self._compare_numerical_values(response, reference)
        
        # Combine scores
        weights = [0.4, 0.4, 0.2]
        final_score = (
            weights[0] * entity_score +
            weights[1] * fact_score +
            weights[2] * number_score
        )
        
        return min(1.0, max(0.0, final_score))

    def _compare_named_entities(self, response: str, reference: str) -> float:
        """Compare named entities between response and reference"""
        doc1 = self.nlp(response)
        doc2 = self.nlp(reference)
        
        ents1 = set((ent.text.lower(), ent.label_) for ent in doc1.ents)
        ents2 = set((ent.text.lower(), ent.label_) for ent in doc2.ents)
        
        if not ents2:  # No entities in reference
            return 1.0 if not ents1 else 0.0
            
        # Calculate Jaccard similarity for entities
        intersection = len(ents1.intersection(ents2))
        union = len(ents1.union(ents2))
        
        return intersection / union if union > 0 else 0.0

    #######################
    # 4. Safety Metrics
    #######################

    def measure_bias(self, text: str) -> float:
        """
        Measure bias using:
        1. Demographic bias detection
        2. Stereotype detection
        3. Language fairness
        """
        # 1. Demographic bias
        demographic_score = self._detect_demographic_bias(text)
        
        # 2. Stereotype detection
        stereotype_score = self._detect_stereotypes(text)
        
        # 3. Language fairness
        fairness_score = self._measure_language_fairness(text)
        
        # Combine scores (higher score means less bias)
        weights = [0.4, 0.4, 0.2]
        final_score = (
            weights[0] * demographic_score +
            weights[1] * stereotype_score +
            weights[2] * fairness_score
        )
        
        return min(1.0, max(0.0, final_score))

    def _detect_demographic_bias(self, text: str) -> float:
        """Detect demographic-related bias in text"""
        # Implementation would include checking for biased language
        # related to gender, race, age, etc.
        return 1.0  # Placeholder

    #######################
    # Utility Functions
    #######################

    def calculate_perplexity(self, text: str) -> float:
        """Calculate language model perplexity"""
        # Implementation would use a language model to calculate perplexity
        return 0.0  # Placeholder

    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract various linguistic features from text"""
        doc = self.nlp(text)
        
        features = {
            'avg_word_length': np.mean([len(token.text) for token in doc]),
            'sentence_count': len(list(doc.sents)),
            'avg_sentence_length': len(doc) / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0,
            'entity_density': len(doc.ents) / len(doc) if len(doc) > 0 else 0,
        }
        
        return features

# Example usage
if __name__ == "__main__":
    evaluator = DetailedMetricsImplementation()
    
    sample_text = """
    Machine learning is a subset of artificial intelligence that enables systems 
    to learn and improve from experience. Through pattern recognition and 
    computational learning, these systems can develop sophisticated models 
    without explicit programming.
    """
    
    sample_prompt = "Explain what machine learning is."
    sample_reference = """
    Machine learning is an AI technology that allows computers to learn from 
    data and experiences. It uses statistical techniques to enable systems to 
    improve their performance over time without being explicitly programmed.
    """
    
    # Calculate various metrics
    coherence = evaluator.measure_coherence(sample_text)
    fluency = evaluator.measure_fluency(sample_text)
    relevance = evaluator.measure_relevance(sample_text, sample_prompt)
    factual_accuracy = evaluator.measure_factual_accuracy(sample_text, sample_reference)
    
    print(f"Coherence: {coherence:.3f}")
    print(f"Fluency: {fluency:.3f}")
    print(f"Relevance: {relevance:.3f}")
    print(f"Factual Accuracy: {factual_accuracy:.3f}")