import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import re
from typing import List, Dict, Union
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize spaCy
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Fraud-related keywords and patterns
FRAUD_KEYWORDS = {
    'urgency': ['immediate', 'urgent', 'asap', 'quick', 'fast', 'emergency'],
    'vague': ['unknown', 'mysterious', 'unclear', 'unsure', 'maybe'],
    'lack_evidence': ['no proof', 'no receipt', 'no documentation', 'lost', 'missing'],
    'high_value': ['expensive', 'valuable', 'precious', 'rare', 'antique'],
    'suspicious': ['strange', 'unusual', 'suspicious', 'odd', 'weird']
}

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_basic_features(text: str) -> Dict[str, Union[int, float]]:
    """Extract basic text features."""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Tokenize
    tokens = word_tokenize(cleaned_text)
    
    # Remove stopwords
    tokens_no_stop = [t for t in tokens if t not in stop_words]
    
    # Calculate features
    features = {
        'text_length': len(text),
        'word_count': len(tokens),
        'unique_word_count': len(set(tokens)),
        'avg_word_length': np.mean([len(t) for t in tokens]) if tokens else 0,
        'stopword_ratio': len(tokens) - len(tokens_no_stop) / len(tokens) if tokens else 0,
        'sentence_count': len(nltk.sent_tokenize(text))
    }
    
    return features

def extract_fraud_indicators(text: str) -> Dict[str, int]:
    """Extract fraud-related indicators from text."""
    text = text.lower()
    indicators = {}
    
    # Check for each category of fraud keywords
    for category, keywords in FRAUD_KEYWORDS.items():
        count = sum(1 for keyword in keywords if keyword in text)
        indicators[f'fraud_{category}_count'] = count
    
    # Calculate total fraud indicators
    indicators['total_fraud_indicators'] = sum(indicators.values())
    
    return indicators

def extract_nlp_features(text: str) -> Dict[str, Union[int, float, List[str]]]:
    """Extract advanced NLP features using spaCy."""
    doc = nlp(text)
    
    features = {
        'entities': [ent.text for ent in doc.ents],
        'entity_count': len(doc.ents),
        'noun_phrases': [chunk.text for chunk in doc.noun_chunks],
        'noun_phrase_count': len(list(doc.noun_chunks)),
        'verbs': [token.lemma_ for token in doc if token.pos_ == 'VERB'],
        'verb_count': len([token for token in doc if token.pos_ == 'VERB']),
        'adjectives': [token.lemma_ for token in doc if token.pos_ == 'ADJ'],
        'adjective_count': len([token for token in doc if token.pos_ == 'ADJ'])
    }
    
    return features

def get_all_features(text: str) -> Dict[str, Union[int, float, List[str]]]:
    """Combine all feature extraction methods."""
    features = {}
    
    # Add basic features
    features.update(extract_basic_features(text))
    
    # Add fraud indicators
    features.update(extract_fraud_indicators(text))
    
    # Add NLP features
    features.update(extract_nlp_features(text))
    
    return features

def preprocess_claim_narrative(text: str) -> Dict[str, Union[int, float, List[str]]]:
    """Main function to preprocess a claim narrative and extract all features."""
    if not isinstance(text, str):
        return {}
    
    try:
        return get_all_features(text)
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return {} 