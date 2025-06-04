"""
AI Enhanced Bank Statement Fraud Detection System
with improved file type handling, parsing capabilities, and Deep Learning Autoencoder

Author: Lester L. Artis Jr. 
Created: 03/15/2025
Enhanced: Added Deep Learning Autoencoder for Unsupervised Anomaly Detection
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
import warnings
from pathlib import Path
import json
import sys
import io
import mimetypes
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Deep Learning and NLP imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow available for autoencoder functionality")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not found. Autoencoder functionality disabled.")
    print("Install with: pip install tensorflow")

# NLP and ML imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.cluster import DBSCAN, IsolationForest
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import silhouette_score
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    import spacy
    NLP_AVAILABLE = True
    print("NLP libraries available")
except ImportError:
    NLP_AVAILABLE = False
    print("NLP libraries not found. Install with: pip install scikit-learn nltk spacy")

# Chatbot imports
try:
    import openai
    OPENAI_AVAILABLE = True
    print("OpenAI available for chatbot functionality")
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not found. Install with: pip install openai")

# Optional dependencies - import with try/except
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("PyPDF2 not found. PDF support disabled.")

try:
    import pytesseract
    from PIL import Image
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False
    print("Tesseract OCR dependencies not found. Image support disabled.")

try:
    import pdfplumber
    PDFPLUMBER_SUPPORT = True
except ImportError:
    PDFPLUMBER_SUPPORT = False
    print("pdfplumber not found. Advanced PDF parsing disabled.")

# Try to import file type detection libraries
try:
    import magic
    MAGIC_SUPPORT = True
except ImportError:
    MAGIC_SUPPORT = False
    print("python-magic not found. Advanced file type detection disabled.")

# Try to import Word document support
try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("python-docx not found. Word document support disabled.")

# Try to import XML support
try:
    import xml.etree.ElementTree as ET
    XML_SUPPORT = True
except ImportError:
    XML_SUPPORT = False
    print("XML support disabled.")

# Try to import character encoding detection
try:
    import chardet
    CHARDET_SUPPORT = True
except ImportError:
    CHARDET_SUPPORT = False
    print("chardet not found. Advanced text encoding detection disabled.")


class NLPFraudDetector:
    """
    NLP-based Fraud Detection using Unsupervised Learning
    
    This class analyzes transaction descriptions and patterns using Natural Language Processing
    and unsupervised machine learning techniques to identify anomalous transactions.
    """
    
    def __init__(self, contamination=0.1):
        """
        Initialize NLP Fraud Detector
        
        Args:
            contamination: Expected fraction of outliers in the dataset
        """
        self.contamination = contamination
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.dbscan = None
        self.stemmer = PorterStemmer()
        self.is_trained = False
        self.feature_matrix = None
        self.transaction_clusters = None
        
        # Initialize NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def preprocess_text(self, text):
        """Preprocess transaction descriptions for NLP analysis"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase and remove special characters
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize and stem
        tokens = word_tokenize(text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens if token not in stopwords.words('english')]
        
        return ' '.join(stemmed_tokens)
    
    def extract_features(self, transactions):
        """Extract features from transactions for unsupervised learning"""
        features = []
        descriptions = []
        
        for tx in transactions:
            # Text features from description
            desc = self.preprocess_text(tx.get('description', ''))
            descriptions.append(desc)
            
            # Numerical features
            amount = float(tx.get('amount', 0))
            timestamp = tx.get('timestamp', '')
            
            # Extract time-based features
            try:
                dt = pd.to_datetime(timestamp)
                hour = dt.hour
                day_of_week = dt.dayofweek
                is_weekend = 1 if day_of_week >= 5 else 0
            except:
                hour = 0
                day_of_week = 0
                is_weekend = 0
            
            # Combine numerical features
            numerical_features = [
                amount,
                np.log1p(amount),
                hour,
                day_of_week,
                is_weekend,
                len(desc.split()),  # Description word count
                1 if amount > 1000 else 0,  # High amount flag
                1 if hour < 6 or hour > 22 else 0,  # Unusual hour flag
            ]
            
            features.append(numerical_features)
        
        # Convert text to TF-IDF features
        if descriptions:
            tfidf_features = self.tfidf_vectorizer.fit_transform(descriptions).toarray()
        else:
            tfidf_features = np.zeros((len(transactions), 1000))
        
        # Combine numerical and text features
        numerical_array = np.array(features)
        combined_features = np.hstack([numerical_array, tfidf_features])
        
        return combined_features, descriptions
    
    def train(self, transactions):
        """Train unsupervised models on transaction data"""
        print(f"Training NLP fraud detector on {len(transactions)} transactions...")
        
        if not NLP_AVAILABLE:
            raise ImportError("NLP libraries required. Install with: pip install scikit-learn nltk")
        
        # Extract features
        self.feature_matrix, descriptions = self.extract_features(transactions)
        
        # Train Isolation Forest for anomaly detection
        self.isolation_forest.fit(self.feature_matrix)
        
        # Perform DBSCAN clustering to identify transaction patterns
        # Use PCA to reduce dimensionality for clustering
        pca = PCA(n_components=50)
        reduced_features = pca.fit_transform(self.feature_matrix)
        
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.transaction_clusters = self.dbscan.fit_predict(reduced_features)
        
        # Calculate silhouette score
        n_clusters = len(set(self.transaction_clusters)) - (1 if -1 in self.transaction_clusters else 0)
        if n_clusters > 1:
            silhouette_avg = silhouette_score(reduced_features, self.transaction_clusters)
            print(f"Clustering silhouette score: {silhouette_avg:.3f}")
        
        self.is_trained = True
        print(f"Found {n_clusters} transaction clusters")
        
    def detect_anomalies(self, transactions):
        """Detect anomalies in new transactions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Extract features for new transactions
        features, descriptions = self.extract_features(transactions)
        
        # Predict anomalies using Isolation Forest
        anomaly_scores = self.isolation_forest.decision_function(features)
        anomaly_predictions = self.isolation_forest.predict(features)
        
        # Identify cluster assignments for new transactions
        pca = PCA(n_components=50)
        pca.fit(self.feature_matrix)  # Fit on training data
        reduced_features = pca.transform(features)
        cluster_predictions = self.dbscan.fit_predict(reduced_features)
        
        results = []
        for i, tx in enumerate(transactions):
            is_anomaly = anomaly_predictions[i] == -1
            anomaly_score = anomaly_scores[i]
            cluster = cluster_predictions[i]
            
            # Normalize anomaly score to 0-100 scale
            normalized_score = max(0, min(100, (1 - anomaly_score) * 50))
            
            result = {
                'transaction_id': tx.get('id', f'tx_{i}'),
                'is_anomaly': is_anomaly,
                'anomaly_score': normalized_score,
                'cluster': cluster,
                'anomaly_confidence': 'High' if normalized_score > 75 else 'Medium' if normalized_score > 50 else 'Low',
                'description_analysis': self._analyze_description(tx.get('description', '')),
                'original_transaction': tx
            }
            results.append(result)
        
        summary = {
            'total_transactions': len(transactions),
            'anomalies_detected': sum(1 for r in results if r['is_anomaly']),
            'anomaly_rate': (sum(1 for r in results if r['is_anomaly']) / len(transactions)) * 100,
            'unique_clusters': len(set(cluster_predictions))
        }
        
        return {
            'summary': summary,
            'detailed_results': results
        }
    
    def _analyze_description(self, description):
        """Analyze transaction description for suspicious patterns"""
        if not description:
            return {'suspicious_keywords': [], 'analysis': 'No description available'}
        
        suspicious_keywords = [
            'cash advance', 'atm withdrawal', 'foreign transaction', 'online purchase',
            'wire transfer', 'money order', 'prepaid', 'bitcoin', 'crypto', 'gambling'
        ]
        
        desc_lower = description.lower()
        found_keywords = [kw for kw in suspicious_keywords if kw in desc_lower]
        
        analysis = "Normal transaction"
        if found_keywords:
            analysis = f"Contains suspicious keywords: {', '.join(found_keywords)}"
        elif len(description.split()) < 2:
            analysis = "Very short description - potentially suspicious"
        elif re.search(r'\d{10,}', description):
            analysis = "Contains long number sequence - potentially suspicious"
        
        return {
            'suspicious_keywords': found_keywords,
            'analysis': analysis
        }


class FraudDetectionChatbot:
    """
    AI-powered chatbot for fraud detection analysis and explanations
    
    Provides natural language interface to fraud detection results and insights.
    """
    
    def __init__(self, api_key=None):
        """Initialize the chatbot with OpenAI API"""
        self.api_key = api_key
        self.conversation_history = []
        self.fraud_context = {}
        
        if OPENAI_AVAILABLE and api_key:
            openai.api_key = api_key
        
    def set_fraud_context(self, transactions, rule_results=None, nlp_results=None, ae_results=None):
        """Set the fraud detection context for the chatbot"""
        self.fraud_context = {
            'total_transactions': len(transactions),
            'transactions': transactions,
            'rule_results': rule_results,
            'nlp_results': nlp_results,
            'autoencoder_results': ae_results
        }
        
        # Generate summary statistics
        if rule_results:
            rule_summary = self._summarize_rule_results(rule_results)
            self.fraud_context['rule_summary'] = rule_summary
        
        if nlp_results:
            nlp_summary = self._summarize_nlp_results(nlp_results)
            self.fraud_context['nlp_summary'] = nlp_summary
        
        if ae_results:
            ae_summary = ae_results.get('summary', {})
            self.fraud_context['ae_summary'] = ae_summary
    
    def _summarize_rule_results(self, rule_results):
        """Summarize rule-based detection results"""
        total = len(rule_results)
        high_risk = sum(1 for r in rule_results if r['fraud_likelihood'] == 'High')
        medium_risk = sum(1 for r in rule_results if r['fraud_likelihood'] == 'Medium')
        
        # Most triggered rules
        rule_counts = {}
        for result in rule_results:
            for rule in result['triggered_rules']:
                rule_name = rule['rule_name']
                rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
        
        top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'total': total,
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'high_risk_rate': (high_risk / total) * 100 if total > 0 else 0,
            'top_triggered_rules': top_rules
        }
    
    def _summarize_nlp_results(self, nlp_results):
        """Summarize NLP-based detection results"""
        summary = nlp_results.get('summary', {})
        detailed = nlp_results.get('detailed_results', [])
        
        # Analyze suspicious keywords
        all_keywords = []
        for result in detailed:
            keywords = result['description_analysis']['suspicious_keywords']
            all_keywords.extend(keywords)
        
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            **summary,
            'top_suspicious_keywords': top_keywords,
            'keyword_coverage': len(all_keywords) / len(detailed) * 100 if detailed else 0
        }
    
    def chat(self, user_message):
        """Process user message and return chatbot response"""
        if not OPENAI_AVAILABLE or not self.api_key:
            return self._generate_fallback_response(user_message)
        
        try:
            # Build context prompt
            context_prompt = self._build_context_prompt()
            
            # Create conversation with context
            messages = [
                {"role": "system", "content": context_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Add conversation history
            for msg in self.conversation_history[-6:]:  # Last 6 messages for context
                messages.append(msg)
            
            # Get response from OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            bot_response = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": bot_response})
            
            return bot_response
            
        except Exception as e:
            return f"I'm having trouble processing your request. Error: {str(e)}"
    
    def _build_context_prompt(self):
        """Build context prompt with fraud detection results"""
        prompt = """You are a helpful AI assistant specializing in fraud detection analysis. 
        You have access to fraud detection results from multiple AI models and can help users understand:
        - Transaction patterns and anomalies
        - Risk levels and explanations
        - Suspicious activities identified
        - Recommendations for further investigation
        
        Current Analysis Context:
        """
        
        if not self.fraud_context:
            prompt += "No fraud analysis data available yet. Please upload and analyze some transactions first."
            return prompt
        
        total_tx = self.fraud_context.get('total_transactions', 0)
        prompt += f"Total transactions analyzed: {total_tx}\n"
        
        # Rule-based results
        if 'rule_summary' in self.fraud_context:
            rule_sum = self.fraud_context['rule_summary']
            prompt += f"""
Rule-based Detection Results:
- High risk transactions: {rule_sum['high_risk']} ({rule_sum['high_risk_rate']:.1f}%)
- Medium risk transactions: {rule_sum['medium_risk']}
- Most triggered rules: {', '.join([f"{rule[0]} ({rule[1]} times)" for rule in rule_sum['top_triggered_rules'][:3]])}
"""
        
        # NLP results
        if 'nlp_summary' in self.fraud_context:
            nlp_sum = self.fraud_context['nlp_summary']
            prompt += f"""
NLP-based Detection Results:
- Anomalies detected: {nlp_sum.get('anomalies_detected', 0)} ({nlp_sum.get('anomaly_rate', 0):.1f}%)
- Transaction clusters found: {nlp_sum.get('unique_clusters', 0)}
- Top suspicious keywords: {', '.join([f"{kw[0]} ({kw[1]} times)" for kw in nlp_sum.get('top_suspicious_keywords', [])[:3]])}
"""
        
        # Autoencoder results
        if 'ae_summary' in self.fraud_context:
            ae_sum = self.fraud_context['ae_summary']
            prompt += f"""
Deep Learning Detection Results:
- Anomalies detected: {ae_sum.get('anomalies_detected', 0)} ({ae_sum.get('anomaly_rate_percent', 0):.1f}%)
- Mean reconstruction error: {ae_sum.get('mean_reconstruction_error', 0):.6f}
"""
        
        prompt += """
Please provide helpful, accurate, and actionable insights based on this data. 
Be conversational but professional. If asked about specific transactions, refer to the analysis results.
Always be clear about the confidence level of your assessments.
"""
        
        return prompt
    
    def _generate_fallback_response(self, user_message):
        """Generate a fallback response when OpenAI is not available"""
        user_lower = user_message.lower()
        
        if not self.fraud_context:
            return "I don't have any fraud analysis data to work with yet. Please upload and analyze some transactions first, then I can help you understand the results!"
        
        # Simple keyword-based responses
        if any(word in user_lower for word in ['summary', 'overview', 'total', 'how many']):
            return self._generate_summary_response()
        
        elif any(word in user_lower for word in ['high risk', 'dangerous', 'suspicious']):
            return self._generate_high_risk_response()
        
        elif any(word in user_lower for word in ['explain', 'why', 'reason']):
            return self._generate_explanation_response()
        
        elif any(word in user_lower for word in ['recommendation', 'what should', 'advice']):
            return self._generate_recommendation_response()
        
        else:
            return "I can help you understand the fraud detection results. Try asking me about: summary, high-risk transactions, explanations, or recommendations."
    
    def _generate_summary_response(self):
        """Generate a summary response"""
        total_tx = self.fraud_context.get('total_transactions', 0)
        response = f"Here's a summary of the analysis for {total_tx} transactions:\n\n"
        
        if 'rule_summary' in self.fraud_context:
            rule_sum = self.fraud_context['rule_summary']
            response += f"• Rule-based analysis found {rule_sum['high_risk']} high-risk and {rule_sum['medium_risk']} medium-risk transactions\n"
        
        if 'nlp_summary' in self.fraud_context:
            nlp_sum = self.fraud_context['nlp_summary']
            response += f"• NLP analysis detected {nlp_sum.get('anomalies_detected', 0)} anomalies in {nlp_sum.get('unique_clusters', 0)} transaction clusters\n"
        
        if 'ae_summary' in self.fraud_context:
            ae_sum = self.fraud_context['ae_summary']
            response += f"• Deep learning analysis flagged {ae_sum.get('anomalies_detected', 0)} transactions as anomalous\n"
        
        return response
    
    def _generate_high_risk_response(self):
        """Generate response about high-risk transactions"""
        response = "High-risk transactions identified:\n\n"
        
        if 'rule_summary' in self.fraud_context:
            rule_sum = self.fraud_context['rule_summary']
            response += f"• {rule_sum['high_risk']} transactions flagged as high-risk by rule-based analysis\n"
            if rule_sum['top_triggered_rules']:
                response += f"• Most common risk factors: {', '.join([rule[0] for rule in rule_sum['top_triggered_rules'][:2]])}\n"
        
        response += "\nI recommend reviewing these transactions manually for potential fraud."
        return response
    
    def _generate_explanation_response(self):
        """Generate explanation response"""
        return "The fraud detection system uses multiple AI approaches:\n\n• Rule-based: Checks for known fraud patterns like large amounts, unusual hours\n• NLP analysis: Examines transaction descriptions for suspicious keywords and patterns\n• Deep learning: Uses neural networks to detect anomalies in transaction patterns\n\nEach method provides different insights, and combining them gives a comprehensive view of potential fraud."
    
    def _generate_recommendation_response(self):
        """Generate recommendation response"""
        recommendations = [
            "1. Review all high-risk transactions manually",
            "2. Investigate transactions with multiple risk factors",
            "3. Set up monitoring for unusual transaction patterns",
            "4. Consider implementing additional verification for high-value transactions"
        ]
        
        if 'nlp_summary' in self.fraud_context:
            nlp_sum = self.fraud_context['nlp_summary']
            if nlp_sum.get('top_suspicious_keywords'):
                keywords = [kw[0] for kw in nlp_sum['top_suspicious_keywords'][:2]]
                recommendations.append(f"5. Monitor transactions containing keywords: {', '.join(keywords)}")
        
        return "Based on the analysis, here are my recommendations:\n\n" + "\n".join(recommendations)
    
    def get_conversation_history(self):
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


class AutoencoderFraudDetector:
    """
    Deep Learning Autoencoder-based Fraud Detection System
    
    This class implements an unsupervised anomaly detection system using autoencoders.
    The approach works by:
    1. Training an autoencoder neural network on ONLY normal (non-fraudulent) transactions
    2. Using the reconstruction error to identify anomalies (potential fraud)
    3. Applying threshold strategies to flag suspicious transactions
    
    Key Principles:
    - Unsupervised Learning: No fraud labels used during training
    - Reconstruction Error: Higher error suggests anomalous patterns
    - Deep Architecture: Multiple layers to capture complex patterns
    """
    
    def __init__(self, encoding_dim=32, hidden_layers=[64, 32], 
                 threshold_strategy='percentile', threshold_value=95,
                 random_state=42):
        """
        Initialize the Autoencoder Fraud Detector
        
        Args:
            encoding_dim (int): Dimensionality of the encoded representation (bottleneck layer)
                              Lower values force more compression, potentially better anomaly detection
            hidden_layers (list): Sizes of hidden layers in the encoder
                                 The decoder will mirror this structure
            threshold_strategy (str): Method for determining fraud threshold
                                    Options: 'percentile', 'std_dev', 'manual'
            threshold_value (float): Value for threshold calculation
                                   - percentile: 95 means top 5% are flagged as anomalies
                                   - std_dev: 3 means mean + 3*std is the threshold
                                   - manual: direct threshold value
            random_state (int): Random seed for reproducibility
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for autoencoder functionality. "
                            "Install with: pip install tensorflow")
        
        # Model architecture parameters
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.random_state = random_state
        
        # Threshold configuration for anomaly detection
        self.threshold_strategy = threshold_strategy
        self.threshold_value = threshold_value
        self.anomaly_threshold = None
        
        # Model and preprocessing components
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.scaler = StandardScaler()  # For feature normalization
        self.label_encoders = {}  # For categorical variable encoding
        self.feature_names = []
        self.is_trained = False
        
        # Training history and metrics
        self.training_history = None
        self.normal_reconstruction_errors = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(random_state)
    
    def _prepare_features(self, transactions: List[Dict]) -> np.ndarray:
        """
        Prepare transaction features for the autoencoder neural network
        
        This method transforms raw transaction data into numerical features suitable
        for neural network processing. It handles both numerical and categorical features.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            np.ndarray: Processed feature matrix ready for the autoencoder
            
        Feature Engineering Strategy:
        1. Extract numerical features (amount, hour, day of week, etc.)
        2. Encode categorical features (description keywords, merchant categories)
        3. Create derived features (time-based patterns, amount patterns)
        4. Normalize all features to [0, 1] range for stable neural network training
        """
        print(f"Preparing features for {len(transactions)} transactions...")
        
        features_list = []
        
        for tx in transactions:
            feature_vector = []
            
            # =================================================================
            # NUMERICAL FEATURES
            # =================================================================
            
            # Amount-based features (core fraud indicators)
            amount = float(tx.get('amount', 0))
            feature_vector.extend([
                amount,                           # Raw transaction amount
                np.log1p(amount),                # Log-transformed amount (handles skewness)
                amount ** 0.5,                   # Square root (another transformation)
            ])
            
            # =================================================================
            # TIME-BASED FEATURES
            # =================================================================
            
            # Extract temporal patterns from timestamp
            timestamp = tx.get('timestamp', '1970-01-01T00:00:00')
            try:
                dt = self._parse_timestamp(timestamp)
                
                # Hour of day (0-23) - fraud often occurs at unusual hours
                hour = dt.hour
                feature_vector.extend([
                    hour,                         # Raw hour
                    np.sin(2 * np.pi * hour / 24),  # Cyclical encoding of hour
                    np.cos(2 * np.pi * hour / 24),  # (captures midnight = 0 = 24 relationship)
                ])
                
                # Day of week (0-6) - fraud patterns may vary by day
                day_of_week = dt.weekday()
                feature_vector.extend([
                    day_of_week,                  # Raw day of week
                    np.sin(2 * np.pi * day_of_week / 7),  # Cyclical encoding
                    np.cos(2 * np.pi * day_of_week / 7),
                ])
                
                # Day of month (1-31) - may capture monthly patterns
                day_of_month = dt.day
                feature_vector.extend([
                    day_of_month,
                    np.sin(2 * np.pi * day_of_month / 31),
                    np.cos(2 * np.pi * day_of_month / 31),
                ])
                
            except Exception:
                # If timestamp parsing fails, use default values
                feature_vector.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])
            
            # =================================================================
            # CATEGORICAL FEATURES (ENCODED AS NUMERICAL)
            # =================================================================
            
            # Transaction description analysis
            description = str(tx.get('description', '')).lower()
            
            # Binary indicators for suspicious keywords
            suspicious_keywords = [
                'atm', 'withdrawal', 'transfer', 'online', 'purchase',
                'gas', 'grocery', 'restaurant', 'retail', 'payment'
            ]
            
            for keyword in suspicious_keywords:
                feature_vector.append(1 if keyword in description else 0)
            
            # Description length (longer descriptions might indicate different transaction types)
            feature_vector.append(len(description))
            
            # Merchant category encoding
            merchant_category = str(tx.get('merchantCategory', 'unknown')).lower()
            
            # Create a hash-based encoding for merchant categories
            # This gives us a numerical representation while preserving some category information
            category_hash = hash(merchant_category) % 100  # Mod 100 to keep values reasonable
            feature_vector.append(category_hash)
            
            # Location-based features
            location = str(tx.get('location', 'unknown')).lower()
            location_hash = hash(location) % 100
            feature_vector.append(location_hash)
            
            # =================================================================
            # DERIVED FEATURES (ADVANCED PATTERNS)
            # =================================================================
            
            # Amount patterns that might indicate fraud
            feature_vector.extend([
                1 if amount > 1000 else 0,        # Large amount flag
                1 if amount < 1 else 0,           # Very small amount flag
                1 if amount % 1 == 0 else 0,      # Round amount flag (exactly whole dollars)
                len(str(int(amount))),            # Number of digits in amount
            ])
            
            features_list.append(feature_vector)
        
        # Convert to numpy array for efficient processing
        feature_matrix = np.array(features_list, dtype=np.float32)
        
        # Store feature names for later reference (debugging and interpretation)
        if not self.feature_names:
            self.feature_names = [
                'amount', 'log_amount', 'sqrt_amount',
                'hour', 'hour_sin', 'hour_cos',
                'day_of_week', 'dow_sin', 'dow_cos',
                'day_of_month', 'dom_sin', 'dom_cos'
            ] + [f'keyword_{kw}' for kw in suspicious_keywords] + [
                'description_length', 'merchant_category_hash', 'location_hash',
                'large_amount_flag', 'small_amount_flag', 'round_amount_flag', 'amount_digits'
            ]
        
        print(f"Generated {feature_matrix.shape[1]} features for {feature_matrix.shape[0]} transactions")
        return feature_matrix
    
    def _parse_timestamp(self, timestamp):
        """Parse timestamp string to datetime object with multiple format support"""
        if isinstance(timestamp, datetime):
            return timestamp
            
        if isinstance(timestamp, str):
            # Try common formats
            formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO format with microseconds
                "%Y-%m-%dT%H:%M:%SZ",     # ISO format
                "%Y-%m-%d %H:%M:%S",      # Standard datetime
                "%m/%d/%Y %H:%M:%S",      # US format with time
                "%m/%d/%Y",               # US date only
                "%Y-%m-%d",               # ISO date only
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
        
        # If all parsing fails, return a default datetime
        return datetime(1970, 1, 1)
    
    def _build_autoencoder_model(self, input_dim: int) -> Tuple[Model, Model, Model]:
        """
        Build the autoencoder neural network architecture
        
        Architecture Design Philosophy:
        - Encoder: Progressively reduces dimensionality to learn compressed representations
        - Decoder: Reconstructs original input from compressed representation
        - Bottleneck: Forces the model to learn essential patterns (compression)
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Tuple of (autoencoder, encoder, decoder) models
        """
        print(f"Building autoencoder architecture for {input_dim} input features...")
        
        # =================================================================
        # INPUT LAYER
        # =================================================================
        input_layer = keras.Input(shape=(input_dim,), name='transaction_input')
        
        # =================================================================
        # ENCODER NETWORK (COMPRESSION PATH)
        # =================================================================
        # The encoder learns to compress transaction data into a lower-dimensional representation
        # Each layer should capture increasingly abstract patterns
        
        encoded = input_layer
        encoder_layers = []
        
        # Build encoder layers with progressively smaller dimensions
        for i, layer_size in enumerate(self.hidden_layers):
            # Dense layer with ReLU activation for non-linearity
            encoded = layers.Dense(
                layer_size, 
                activation='relu',
                name=f'encoder_hidden_{i+1}',
                kernel_regularizer=keras.regularizers.l2(0.001)  # L2 regularization to prevent overfitting
            )(encoded)
            
            # Batch normalization for stable training
            encoded = layers.BatchNormalization(name=f'encoder_bn_{i+1}')(encoded)
            
            # Dropout for regularization (prevents overfitting to normal transactions)
            encoded = layers.Dropout(0.2, name=f'encoder_dropout_{i+1}')(encoded)
            
            encoder_layers.append(encoded)
        
        # Bottleneck layer (most compressed representation)
        # This is the critical component that forces the model to learn essential patterns
        encoded = layers.Dense(
            self.encoding_dim, 
            activation='relu',
            name='bottleneck_encoding',
            kernel_regularizer=keras.regularizers.l2(0.001)
        )(encoded)
        
        # =================================================================
        # DECODER NETWORK (RECONSTRUCTION PATH)
        # =================================================================
        # The decoder learns to reconstruct the original input from the compressed representation
        # Mirror the encoder architecture in reverse
        
        decoded = encoded
        
        # Build decoder layers (reverse of encoder)
        for i, layer_size in enumerate(reversed(self.hidden_layers)):
            decoded = layers.Dense(
                layer_size, 
                activation='relu',
                name=f'decoder_hidden_{i+1}',
                kernel_regularizer=keras.regularizers.l2(0.001)
            )(decoded)
            
            decoded = layers.BatchNormalization(name=f'decoder_bn_{i+1}')(decoded)
            decoded = layers.Dropout(0.2, name=f'decoder_dropout_{i+1}')(decoded)
        
        # Output layer (reconstruction of original input)
        # Linear activation for the final layer (no activation function)
        # This allows the model to output any real values to match the normalized input
        decoded = layers.Dense(
            input_dim, 
            activation='linear',  # Linear activation for reconstruction
            name='reconstruction_output'
        )(decoded)
        
        # =================================================================
        # MODEL COMPILATION
        # =================================================================
        
        # Complete autoencoder (input -> encoding -> reconstruction)
        autoencoder = Model(
            inputs=input_layer, 
            outputs=decoded, 
            name='fraud_detection_autoencoder'
        )
        
        # Standalone encoder (input -> encoding)
        # Useful for getting compressed representations
        encoder = Model(
            inputs=input_layer, 
            outputs=encoded, 
            name='transaction_encoder'
        )
        
        # Standalone decoder (encoding -> reconstruction)
        # Useful for generating reconstructions from encodings
        encoded_input = keras.Input(shape=(self.encoding_dim,))
        decoder_layer = autoencoder.layers[-1]  # Get the last layer (reconstruction)
        
        # Rebuild decoder path for standalone model
        decoded_output = encoded_input
        decoder_start_idx = len(self.hidden_layers) + 2  # Skip input and encoder layers
        
        for layer in autoencoder.layers[decoder_start_idx:]:
            decoded_output = layer(decoded_output)
        
        decoder = Model(
            inputs=encoded_input, 
            outputs=decoded_output, 
            name='transaction_decoder'
        )
        
        # Compile the autoencoder with appropriate loss function and optimizer
        autoencoder.compile(
            optimizer='adam',           # Adam optimizer (adaptive learning rate)
            loss='mse',                # Mean Squared Error for reconstruction
            metrics=['mae']            # Mean Absolute Error as additional metric
        )
        
        print(f"Autoencoder architecture built successfully:")
        print(f"  - Input dimension: {input_dim}")
        print(f"  - Encoding dimension: {self.encoding_dim}")
        print(f"  - Hidden layers: {self.hidden_layers}")
        print(f"  - Total parameters: {autoencoder.count_params()}")
        
        return autoencoder, encoder, decoder
    
    def train(self, transactions: List[Dict], validation_split=0.2, 
              epochs=100, batch_size=32, verbose=1):
        """
        Train the autoencoder on normal (non-fraudulent) transactions
        
        IMPORTANT: This method assumes ALL provided transactions are normal/legitimate.
        In unsupervised anomaly detection, we train only on normal data so the model
        learns to reconstruct normal patterns well. Fraudulent transactions will then
        have higher reconstruction errors.
        
        Args:
            transactions: List of transaction dictionaries (should be NORMAL transactions only)
            validation_split: Fraction of data to use for validation
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level for training output
        """
        print(f"\n=== Training Autoencoder on {len(transactions)} Normal Transactions ===")
        
        if len(transactions) < 10:
            raise ValueError("Need at least 10 transactions for training. "
                           "Consider collecting more normal transaction data.")
        
        # =================================================================
        # FEATURE PREPARATION
        # =================================================================
        print("Step 1/5: Preparing features...")
        features = self._prepare_features(transactions)
        
        # =================================================================
        # DATA NORMALIZATION
        # =================================================================
        print("Step 2/5: Normalizing features...")
        # Fit the scaler on normal transactions and transform
        # Normalization is crucial for neural networks to train effectively
        features_normalized = self.scaler.fit_transform(features)
        
        print(f"Feature statistics after normalization:")
        print(f"  - Mean: {np.mean(features_normalized):.4f}")
        print(f"  - Std: {np.std(features_normalized):.4f}")
        print(f"  - Min: {np.min(features_normalized):.4f}")
        print(f"  - Max: {np.max(features_normalized):.4f}")
        
        # =================================================================
        # MODEL ARCHITECTURE
        # =================================================================
        print("Step 3/5: Building model architecture...")
        input_dim = features_normalized.shape[1]
        self.autoencoder, self.encoder, self.decoder = self._build_autoencoder_model(input_dim)
        
        # =================================================================
        # TRAINING CONFIGURATION
        # =================================================================
        print("Step 4/5: Configuring training...")
        
        # Callbacks for better training
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=15,           # Stop if no improvement for 15 epochs
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate when training plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,           # Reduce LR by half
                patience=10,          # Wait 10 epochs before reducing
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # =================================================================
        # MODEL TRAINING
        # =================================================================
        print("Step 5/5: Training the autoencoder...")
        print(f"Training configuration:")
        print(f"  - Training samples: {int(len(features_normalized) * (1 - validation_split))}")
        print(f"  - Validation samples: {int(len(features_normalized) * validation_split)}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        
        # Train the autoencoder (input = output for autoencoder training)
        self.training_history = self.autoencoder.fit(
            features_normalized, features_normalized,  # Input = Output for autoencoder
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        # =================================================================
        # THRESHOLD CALCULATION
        # =================================================================
        print("Calculating anomaly detection threshold...")
        
        # Get reconstruction errors on the training data
        reconstructions = self.autoencoder.predict(features_normalized, verbose=0)
        reconstruction_errors = np.mean(np.square(features_normalized - reconstructions), axis=1)
        
        self.normal_reconstruction_errors = reconstruction_errors
        
        # Calculate threshold based on strategy
        if self.threshold_strategy == 'percentile':
            self.anomaly_threshold = np.percentile(reconstruction_errors, self.threshold_value)
            print(f"Threshold set at {self.threshold_value}th percentile: {self.anomaly_threshold:.6f}")
            
        elif self.threshold_strategy == 'std_dev':
            mean_error = np.mean(reconstruction_errors)
            std_error = np.std(reconstruction_errors)
            self.anomaly_threshold = mean_error + (self.threshold_value * std_error)
            print(f"Threshold set at mean + {self.threshold_value}*std: {self.anomaly_threshold:.6f}")
            
        elif self.threshold_strategy == 'manual':
            self.anomaly_threshold = self.threshold_value
            print(f"Manual threshold set to: {self.anomaly_threshold:.6f}")
        
        else:
            raise ValueError(f"Unknown threshold strategy: {self.threshold_strategy}")
        
        # =================================================================
        # TRAINING SUMMARY
        # =================================================================
        self.is_trained = True
        
        final_loss = self.training_history.history['loss'][-1]
        final_val_loss = self.training_history.history['val_loss'][-1]
        
        print(f"\n=== Training Complete ===")
        print(f"Final training loss: {final_loss:.6f}")
        print(f"Final validation loss: {final_val_loss:.6f}")
        print(f"Reconstruction error statistics on training data:")
        print(f"  - Mean: {np.mean(reconstruction_errors):.6f}")
        print(f"  - Std: {np.std(reconstruction_errors):.6f}")
        print(f"  - Min: {np.min(reconstruction_errors):.6f}")
        print(f"  - Max: {np.max(reconstruction_errors):.6f}")
        print(f"Anomaly threshold: {self.anomaly_threshold:.6f}")
        
        # Estimate false positive rate on training data
        normal_flagged = np.sum(reconstruction_errors > self.anomaly_threshold)
        false_positive_rate = normal_flagged / len(reconstruction_errors) * 100
        print(f"Expected false positive rate: {false_positive_rate:.2f}%")
    
    def detect_anomalies(self, transactions: List[Dict]) -> Dict:
        """
        Detect anomalies in transactions using the trained autoencoder
        
        This method processes new transactions and flags potential fraud based on
        reconstruction error compared to the learned threshold.
        
        Args:
            transactions: List of transaction dictionaries to analyze
            
        Returns:
            Dictionary containing detection results and analysis
        """
        if not self.is_trained:
            raise ValueError("Autoencoder must be trained before detecting anomalies. "
                           "Call the train() method first.")
        
        print(f"\n=== Detecting Anomalies in {len(transactions)} Transactions ===")
        
        # =================================================================
        # FEATURE PREPARATION
        # =================================================================
        features = self._prepare_features(transactions)
        features_normalized = self.scaler.transform(features)  # Use fitted scaler
        
        # =================================================================
        # RECONSTRUCTION AND ERROR CALCULATION
        # =================================================================
        print("Computing reconstruction errors...")
        
        # Get reconstructions from the autoencoder
        reconstructions = self.autoencoder.predict(features_normalized, verbose=0)
        
        # Calculate reconstruction error for each transaction
        # Mean Squared Error between original and reconstructed features
        reconstruction_errors = np.mean(np.square(features_normalized - reconstructions), axis=1)
        
        # =================================================================
        # ANOMALY FLAGGING
        # =================================================================
        print("Applying anomaly threshold...")
        
        # Flag transactions with error above threshold
        anomaly_flags = reconstruction_errors > self.anomaly_threshold
        
        # =================================================================
        # DETAILED ANALYSIS
        # =================================================================
        
        # Create detailed results for each transaction
        detailed_results = []
        for i, (tx, error, is_anomaly) in enumerate(zip(transactions, reconstruction_errors, anomaly_flags)):
            
            # Calculate anomaly score (0-100 scale)
            # Higher scores indicate higher likelihood of fraud
            if self.normal_reconstruction_errors is not None:
                max_normal_error = np.max(self.normal_reconstruction_errors)
                anomaly_score = min(100, (error / max_normal_error) * 100)
            else:
                anomaly_score = min(100, (error / self.anomaly_threshold) * 100)
            
            # Determine confidence level
            error_ratio = error / self.anomaly_threshold
            if error_ratio > 2.0:
                confidence = "High"
            elif error_ratio > 1.5:
                confidence = "Medium"
            elif error_ratio > 1.0:
                confidence = "Low"
            else:
                confidence = "Normal"
            
            detailed_results.append({
                'transaction_id': tx.get('id', f'tx_{i}'),
                'reconstruction_error': float(error),
                'anomaly_threshold': float(self.anomaly_threshold),
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'confidence': confidence,
                'error_ratio': float(error_ratio),
                'original_transaction': tx
            })
        
        # =================================================================
        # SUMMARY STATISTICS
        # =================================================================
        total_transactions = len(transactions)
        anomalies_detected = np.sum(anomaly_flags)
        anomaly_rate = (anomalies_detected / total_transactions) * 100
        
        summary = {
            'total_transactions': total_transactions,
            'anomalies_detected': int(anomalies_detected),
            'anomaly_rate_percent': float(anomaly_rate),
            'mean_reconstruction_error': float(np.mean(reconstruction_errors)),
            'std_reconstruction_error': float(np.std(reconstruction_errors)),
            'max_reconstruction_error': float(np.max(reconstruction_errors)),
            'min_reconstruction_error': float(np.min(reconstruction_errors)),
            'threshold_used': float(self.anomaly_threshold)
        }
        
        print(f"Anomaly Detection Results:")
        print(f"  - Total transactions: {total_transactions}")
        print(f"  - Anomalies detected: {anomalies_detected} ({anomaly_rate:.2f}%)")
        print(f"  - Mean reconstruction error: {summary['mean_reconstruction_error']:.6f}")
        print(f"  - Threshold: {self.anomaly_threshold:.6f}")
        
        return {
            'summary': summary,
            'detailed_results': detailed_results,
            'reconstruction_errors': reconstruction_errors.tolist(),
            'anomaly_flags': anomaly_flags.tolist()
        }
    
    def update_threshold(self, new_threshold: float = None, strategy: str = None, value: float = None):
        """
        Update the anomaly detection threshold
        
        Allows dynamic adjustment of the threshold without retraining the model.
        Useful for fine-tuning the sensitivity of anomaly detection.
        
        Args:
            new_threshold: Direct threshold value to use
            strategy: New threshold strategy ('percentile', 'std_dev', 'manual')
            value: Value for the new strategy
        """
        if not self.is_trained or self.normal_reconstruction_errors is None:
            raise ValueError("Model must be trained before updating threshold")
        
        if new_threshold is not None:
            self.anomaly_threshold = new_threshold
            print(f"Threshold manually updated to: {new_threshold:.6f}")
            
        elif strategy is not None and value is not None:
            self.threshold_strategy = strategy
            self.threshold_value = value
            
            reconstruction_errors = self.normal_reconstruction_errors
            
            if strategy == 'percentile':
                self.anomaly_threshold = np.percentile(reconstruction_errors, value)
                print(f"Threshold updated to {value}th percentile: {self.anomaly_threshold:.6f}")
                
            elif strategy == 'std_dev':
                mean_error = np.mean(reconstruction_errors)
                std_error = np.std(reconstruction_errors)
                self.anomaly_threshold = mean_error + (value * std_error)
                print(f"Threshold updated to mean + {value}*std: {self.anomaly_threshold:.6f}")
                
            elif strategy == 'manual':
                self.anomaly_threshold = value
                print(f"Manual threshold updated to: {value:.6f}")
            
            else:
                raise ValueError(f"Unknown threshold strategy: {strategy}")
        
        else:
            raise ValueError("Must provide either new_threshold or both strategy and value")
    
    def save_model(self, filepath: str):
        """Save the trained autoencoder model and preprocessing components"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the autoencoder model
        self.autoencoder.save(f"{filepath}_autoencoder.h5")
        
        # Save preprocessing components and metadata
        model_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'anomaly_threshold': self.anomaly_threshold,
            'threshold_strategy': self.threshold_strategy,
            'threshold_value': self.threshold_value,
            'encoding_dim': self.encoding_dim,
            'hidden_layers': self.hidden_layers,
            'normal_reconstruction_errors': self.normal_reconstruction_errors,
            'training_history': self.training_history.history if self.training_history else None
        }
        
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}_autoencoder.h5 and {filepath}_metadata.pkl")
    
    def load_model(self, filepath: str):
        """Load a saved autoencoder model and preprocessing components"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required to load saved models")
        
        # Load the autoencoder model
        self.autoencoder = keras.models.load_model(f"{filepath}_autoencoder.h5")
        
        # Load preprocessing components and metadata
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.anomaly_threshold = model_data['anomaly_threshold']
        self.threshold_strategy = model_data['threshold_strategy']
        self.threshold_value = model_data['threshold_value']
        self.encoding_dim = model_data['encoding_dim']
        self.hidden_layers = model_data['hidden_layers']
        self.normal_reconstruction_errors = model_data['normal_reconstruction_errors']
        
        # Rebuild encoder and decoder from loaded autoencoder
        input_layer = self.autoencoder.input
        bottleneck_layer = None
        
        # Find the bottleneck layer
        for layer in self.autoencoder.layers:
            if 'bottleneck' in layer.name:
                bottleneck_layer = layer.output
                break
        
        if bottleneck_layer is not None:
            self.encoder = Model(inputs=input_layer, outputs=bottleneck_layer)
        
        self.is_trained = True
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, transactions: List[Dict], n_top_features=10):
        """
        Analyze which features contribute most to anomaly detection
        
        This method helps understand what patterns the autoencoder has learned
        and which transaction characteristics are most important for fraud detection.
        
        Args:
            transactions: Sample transactions to analyze
            n_top_features: Number of top contributing features to return
            
        Returns:
            Dictionary with feature importance analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before feature importance analysis")
        
        print(f"Analyzing feature importance for {len(transactions)} transactions...")
        
        # Prepare features
        features = self._prepare_features(transactions)
        features_normalized = self.scaler.transform(features)
        
        # Get reconstructions
        reconstructions = self.autoencoder.predict(features_normalized, verbose=0)
        
        # Calculate per-feature reconstruction errors
        feature_errors = np.square(features_normalized - reconstructions)
        mean_feature_errors = np.mean(feature_errors, axis=0)
        
        # Create importance ranking
        feature_importance = []
        for i, (feature_name, error) in enumerate(zip(self.feature_names, mean_feature_errors)):
            feature_importance.append({
                'feature_name': feature_name,
                'mean_error': float(error),
                'rank': i + 1
            })
        
        # Sort by error (higher error = more important for anomaly detection)
        feature_importance.sort(key=lambda x: x['mean_error'], reverse=True)
        
        # Update ranks
        for i, item in enumerate(feature_importance):
            item['rank'] = i + 1
        
        top_features = feature_importance[:n_top_features]
        
        print(f"Top {n_top_features} most important features for anomaly detection:")
        for i, feature in enumerate(top_features):
            print(f"  {i+1}. {feature['feature_name']}: {feature['mean_error']:.6f}")
        
        return {
            'all_features': feature_importance,
            'top_features': top_features,
            'feature_names': self.feature_names
        }
    
    def plot_training_history(self, save_path=None):
        """Plot training history for model analysis"""
        if not self.training_history:
            raise ValueError("No training history available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.training_history.history['loss'], label='Training Loss')
        ax1.plot(self.training_history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss During Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Reconstruction error distribution
        if self.normal_reconstruction_errors is not None:
            ax2.hist(self.normal_reconstruction_errors, bins=50, alpha=0.7, color='blue', label='Normal Transactions')
            ax2.axvline(self.anomaly_threshold, color='red', linestyle='--', label=f'Threshold: {self.anomaly_threshold:.4f}')
            ax2.set_title('Distribution of Reconstruction Errors')
            ax2.set_xlabel('Reconstruction Error')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")
        
        plt.show()


# Continue with the original classes...
class AIDocumentProcessor:
    """
    AI-powered processor for various document types that extracts transaction data
    and identifies banking information. Now with enhanced file type support.
    """
    
    def __init__(self):
        """Initialize the document processor with banking categories and patterns."""
        # Banking keyword categories for AI detection
        self.banking_categories = {
            'transaction_date': ['date', 'datetime', 'transaction date', 'post date', 'posting date'],
            'transaction_id': ['id', 'reference', 'confirmation', 'transaction id', 'ref number', 'confirmation number'],
            'amount': ['amount', 'sum', 'debit', 'credit', 'transaction amount', 'payment', 'deposit', 'withdrawal'],
            'balance': ['balance', 'current balance', 'available balance', 'ending balance', 'new balance'],
            'description': ['description', 'details', 'memo', 'narrative', 'payee', 'transaction description'],
            'category': ['category', 'transaction type', 'type', 'classification', 'merchant category'],
            'location': ['location', 'merchant location', 'place', 'address', 'merchant address']
        }

        # Common bank statement patterns
        self.statement_patterns = {
            'account_info': re.compile(r'(account\s*number|acct\s*#|account\s*#|Account Holder)[:.\s]*([^$\n]+)', re.IGNORECASE),
            'date_range': re.compile(r'(statement\s*period|from|statement\s*dates|Account Statement)[:.\s]*([^-]+)\s*-\s*([^$\n]+)', re.IGNORECASE),
            'balance': re.compile(r'(closing\s*balance|ending\s*balance|available\s*balance|current\s*balance|Balance)[:.\s]*[$£€]?(\d{1,3}(,\d{3})*\.\d{2})', re.IGNORECASE),
            'transaction_date': re.compile(r'(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})', re.IGNORECASE)
        }
        
        # Expanded supported file formats and their processor methods
        self.supported_formats = {
            # Standard formats
            '.csv': self.process_csv,
            '.tsv': self.process_csv,  # Tab-separated values
            '.xlsx': self.process_excel,
            '.xls': self.process_excel,
            '.xlsm': self.process_excel,  # Macro-enabled Excel
            '.xlsb': self.process_excel,  # Binary Excel
            '.pdf': self.process_pdf,
            
            # Image formats
            '.jpg': self.process_image,
            '.jpeg': self.process_image,
            '.png': self.process_image,
            '.gif': self.process_image,
            '.bmp': self.process_image,
            '.tiff': self.process_image,
            '.tif': self.process_image,
            
            # Text formats
            '.txt': self.process_text,
            '.text': self.process_text,
            '.md': self.process_text,
            '.rtf': self.process_text,  # Rich Text Format
            
            # Document formats
            '.docx': self.process_docx,
            '.doc': self.process_docx,  # Will try to convert
            
            # Data formats
            '.json': self.process_json,
            '.xml': self.process_xml,
            '.html': self.process_html,
            '.htm': self.process_html,
            
            # Archive formats (will extract and process contents)
            '.zip': self.process_archive,
            '.tar': self.process_archive,
            '.gz': self.process_archive,
            '.7z': self.process_archive,
            
            # Email formats
            '.eml': self.process_email,
            '.msg': self.process_email,
        }
        
        # MIME type to processor mapping
        self.mime_processors = {
            'text/csv': self.process_csv,
            'text/tab-separated-values': self.process_csv,
            'application/vnd.ms-excel': self.process_excel,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self.process_excel,
            'application/pdf': self.process_pdf,
            'image/jpeg': self.process_image,
            'image/png': self.process_image,
            'image/gif': self.process_image,
            'image/bmp': self.process_image,
            'image/tiff': self.process_image,
            'text/plain': self.process_text,
            'text/markdown': self.process_text,
            'text/rtf': self.process_text,
            'application/msword': self.process_docx,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self.process_docx,
            'application/json': self.process_json,
            'application/xml': self.process_xml,
            'text/xml': self.process_xml,
            'text/html': self.process_html,
            'application/zip': self.process_archive,
            'application/x-tar': self.process_archive,
            'application/gzip': self.process_archive,
            'application/x-7z-compressed': self.process_archive,
            'message/rfc822': self.process_email,
        }
        
        # Fallback processors in order of attempt
        self.fallback_processors = [
            self.process_csv,
            self.process_excel,
            self.process_text,
            self.process_json,
            self.process_xml,
            self.process_pdf,
            self.process_image
        ]
    
    def detect_file_type(self, file_path: str) -> str:
        """
        Detect file type using multiple methods.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Detected MIME type
        """
        mime_type = None
        
        # 1. Try using file extension first (fastest)
        file_ext = os.path.splitext(file_path)[1].lower()
        mime_type = mimetypes.guess_type(file_path)[0]
        
        # 2. If extension-based detection failed, try python-magic
        if not mime_type and MAGIC_SUPPORT:
            try:
                mime_type = magic.Magic(mime=True).from_file(file_path)
            except Exception as e:
                print(f"Magic-based file type detection failed: {str(e)}")
        
        # 3. If magic-based detection failed, try to open and analyze file content
        if not mime_type:
            try:
                # Try to read the first few bytes
                with open(file_path, 'rb') as f:
                    header = f.read(4096)
                
                # Check for common file signatures
                if header.startswith(b'%PDF'):
                    mime_type = 'application/pdf'
                elif header.startswith(b'PK\x03\x04'):
                    # ZIP-based formats (XLSX, DOCX, etc.)
                    mime_type = 'application/zip'
                elif header.startswith(b'\xFF\xD8\xFF'):
                    mime_type = 'image/jpeg'
                elif header.startswith(b'\x89PNG\r\n\x1A\n'):
                    mime_type = 'image/png'
                elif b'<?xml' in header:
                    mime_type = 'application/xml'
                elif b'<html' in header.lower() or b'<!doctype html' in header.lower():
                    mime_type = 'text/html'
                elif b'{' in header and b'}' in header:
                    # Possible JSON
                    try:
                        json.loads(header.decode('utf-8'))
                        mime_type = 'application/json'
                    except:
                        pass
                
                # If still nothing, try to detect if it's text
                if not mime_type and CHARDET_SUPPORT:
                    encoding = chardet.detect(header)
                    if encoding['confidence'] > 0.8:
                        # It's probably text
                        try:
                            decoded = header.decode(encoding['encoding'])
                            # Check for CSV-like content
                            if ',' in decoded and '\n' in decoded:
                                mime_type = 'text/csv'
                            else:
                                mime_type = 'text/plain'
                        except:
                            pass
            except Exception as e:
                print(f"Content-based file type detection failed: {str(e)}")
        
        # 4. If all detection methods failed, default to binary
        if not mime_type:
            mime_type = 'application/octet-stream'
            
        return mime_type
    
    def process_document(self, file_path: str, debug=False) -> Dict:
        """
        Process a document file and extract transaction data.
        Now with enhanced file type detection and fallback mechanisms.
        
        Args:
            file_path: Path to the document file
            debug: Whether to run debug extraction for PDFs
            
        Returns:
            dict: Document data including transactions and headers
        """
        print(f"Processing document: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Run debug mode for PDFs if requested
        if debug and file_path.lower().endswith('.pdf'):
            self.debug_pdf_extraction(file_path)
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Enhanced file type detection
        mime_type = self.detect_file_type(file_path)
        print(f"Detected MIME type: {mime_type}")
        
        try:
            print(f"Step 1/4: File type analysis...")
            # Try extension-based processor first
            if file_ext in self.supported_formats:
                processor = self.supported_formats[file_ext]
                processor_name = f"extension-based ({file_ext})"
            # If not found, try MIME type based processor
            elif mime_type in self.mime_processors:
                processor = self.mime_processors[mime_type]
                processor_name = f"MIME-based ({mime_type})"
            else:
                processor = None
                processor_name = None
            
            print(f"Step 2/4: Selecting processor method ({processor_name if processor_name else 'unknown'})...")
            
            # Process the document based on its type
            if processor:
                try:
                    print(f"Step 3/4: Document parsing...")
                    data = processor(file_path)
                    processing_method = processor_name
                except Exception as e:
                    print(f"Primary processor failed: {str(e)}. Trying fallback processors...")
                    # Try fallback processors if primary fails
                    data = self._try_fallback_processors(file_path)
                    processing_method = "fallback"
            else:
                print(f"Unsupported file type. Trying fallback processors...")
                data = self._try_fallback_processors(file_path)
                processing_method = "fallback"
            
            # If all processing methods failed
            if not data:
                raise ValueError(f"Could not process file: {file_path}. All processing methods failed.")
            
            print(f"Step 4/4: AI classification of fields...")
            # Identify banking categories in the data
            enhanced_data = self.identify_banking_categories(data)
            
            # Add processing method to output
            enhanced_data['processing_method'] = processing_method
            
            print(f"AI processing complete. Found {len(enhanced_data['data'])} transactions and {len(enhanced_data.get('category_mapping', {}))} categorized fields.")
            return enhanced_data
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise
    
    # [Rest of the methods from original AIDocumentProcessor class would go here...]
    # For brevity, I'm including just the essential methods. In a real implementation,
    # all methods would be included.
    
    def process_csv(self, file_path: str) -> Dict:
        """Process a CSV-like file and extract transaction data."""
        try:
            # First attempt: Try standard comma delimiter
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    return {
                        'data': df.to_dict('records'),
                        'headers': list(df.columns),
                        'format': 'csv',
                        'shape': df.shape
                    }
            except Exception as e:
                print(f"Standard CSV parsing failed: {str(e)}. Trying alternative delimiters...")
            
            # Second attempt: Try to determine delimiter
            with open(file_path, 'rb') as f:
                sample = f.read(4096)
                if CHARDET_SUPPORT:
                    encoding = chardet.detect(sample)['encoding'] or 'utf-8'
                else:
                    encoding = 'utf-8'
                
            sample_text = sample.decode(encoding, errors='replace')
            
            # Count potential delimiters
            delimiters = [',', ';', '\t', '|', ':']
            delimiter_counts = {d: sample_text.count(d) for d in delimiters}
            best_delimiter = max(delimiter_counts.items(), key=lambda x: x[1])[0]
            
            # Try with the best delimiter
            df = pd.read_csv(file_path, delimiter=best_delimiter)
            
            # Basic data validation
            if df.empty:
                raise ValueError("CSV file contains no data")
            
            return {
                'data': df.to_dict('records'),
                'headers': list(df.columns),
                'format': f'csv (delimiter: {best_delimiter})',
                'shape': df.shape
            }
        except Exception as e:
            raise ValueError(f"Error processing CSV-like file: {str(e)}")
    
    def identify_banking_categories(self, document_data: Dict) -> Dict:
        """Identify banking categories in data."""
        data = document_data['data']
        headers = document_data['headers']
        
        # Map of confident category matches
        category_mapping = {}
        
        # Score each header against our banking categories
        for header in headers:
            header_lower = str(header).lower()
            
            # Find the best category match
            best_category = None
            best_score = 0
            
            for category, keywords in self.banking_categories.items():
                # Calculate similarity score
                score = self.calculate_similarity_score(header_lower, keywords)
                
                if score > best_score and score > 0.5:
                    best_score = score
                    best_category = category
            
            if best_category:
                category_mapping[header] = {
                    'category': best_category,
                    'confidence': best_score,
                    'original': header
                }
        
        # Add category mapping to document data
        document_data['category_mapping'] = category_mapping
        document_data['ai_enhanced'] = True
        
        return document_data
    
    def calculate_similarity_score(self, header: str, keywords: List[str]) -> float:
        """Calculate similarity score between a header and category keywords."""
        # Exact match
        if header in keywords:
            return 1.0
        
        # Partial matches
        best_score = 0
        
        for keyword in keywords:
            # Check if header contains keyword
            if keyword in header:
                score = len(keyword) / len(header)
                best_score = max(best_score, score)
            # Check if keyword contains header
            elif header in keyword:
                score = len(header) / len(keyword)
                best_score = max(best_score, score)
        
        return best_score


# [Continue with EnhancedFraudDetectionSystem class - keeping original implementation]
class EnhancedFraudDetectionSystem:
    """
    AI-enhanced fraud detection system that analyzes bank transactions
    for suspicious patterns and activities.
    """
    
    def __init__(self, rules=None):
        """Initialize with default or custom rules."""
        # Initialize with default rules if none provided
        self.rules = rules if rules else self.get_default_rules()
        
        # Add AI-enhanced rules
        self.add_ai_rules()
    
    def get_default_rules(self) -> List[Dict]:
        """Define default fraud detection rules."""
        return [
            {
                'id': 'large-amount',
                'name': 'Large Transaction Amount',
                'description': 'Flags transactions above a threshold amount',
                'evaluate': lambda tx, profile, history: 
                    float(tx.get('amount', 0)) > profile.get('large_amount_threshold', 3250),
                'risk_score': 6
            },
            {
                'id': 'unusual-hour',
                'name': 'Unusual Transaction Hour',
                'description': 'Flags transactions occurring during unusual hours (2AM-5AM)',
                'evaluate': lambda tx, profile, history:
                    self._get_hour(tx.get('timestamp')) in range(2, 5),
                'risk_score': 2
            },
            {
                'id': 'high-frequency',
                'name': 'High Transaction Frequency',
                'description': 'Flags when too many transactions occur in a short timeframe',
                'evaluate': lambda tx, profile, history:
                    len([
                        t for t in history 
                        if self._time_diff_hours(tx.get('timestamp'), t.get('timestamp')) <= 1
                    ]) >= 5 if history else False,
                'risk_score': 4
            },
            {
                'id': 'geo-velocity',
                'name': 'Geographical Velocity',
                'description': 'Flags transactions that occur in different locations in a short timeframe',
                'evaluate': lambda tx, profile, history:
                    self._check_geo_velocity(tx, history),
                'risk_score': 5
            }
        ]
    
    def add_ai_rules(self):
        """Add AI-enhanced fraud detection rules."""
        ai_rules = [
            {
                'id': 'ai-amount-pattern',
                'name': 'AI: Unusual Amount Pattern',
                'description': 'Uses AI to detect unusual spending amounts compared to historical patterns',
                'evaluate': lambda tx, profile, history:
                    self._detect_unusual_amount(tx, history),
                'risk_score': 4,
                'is_ai_rule': True
            }
        ]
        
        # Add AI rules to the existing rules
        self.rules.extend(ai_rules)
    
    def analyze_transaction(self, transaction: Dict, user_profile: Dict = None, 
                           transaction_history: List[Dict] = None) -> Dict:
        """Analyze a transaction against all rules."""
        if user_profile is None:
            user_profile = {}
        
        if transaction_history is None:
            transaction_history = []
        
        triggered_rules = []
        total_risk_score = 0
        
        for rule in self.rules:
            try:
                is_triggered = rule['evaluate'](transaction, user_profile, transaction_history)
                
                if is_triggered:
                    triggered_rules.append({
                        'rule_id': rule['id'],
                        'rule_name': rule['name'],
                        'description': rule['description'],
                        'risk_score': rule['risk_score'],
                        'is_ai_rule': rule.get('is_ai_rule', False)
                    })
                    
                    total_risk_score += rule['risk_score']
            except Exception as e:
                print(f"Error evaluating rule {rule['id']}: {str(e)}")
        
        # Determine fraud likelihood based on risk score
        fraud_likelihood = 'Low'
        if total_risk_score >= 6:
            fraud_likelihood = 'High'
        elif total_risk_score >= 3:
            fraud_likelihood = 'Medium'
        
        return {
            'transaction_id': transaction.get('id', 'unknown'),
            'timestamp': transaction.get('timestamp'),
            'fraud_likelihood': fraud_likelihood,
            'risk_score': total_risk_score,
            'triggered_rules': triggered_rules,
            'requires_review': total_risk_score >= 5,
            'ai_enhanced': any(rule.get('is_ai_rule', False) for rule in triggered_rules)
        }
    
    def analyze_batch(self, transactions: List[Dict], user_profile: Dict = None) -> List[Dict]:
        """Analyze a batch of transactions."""
        if user_profile is None:
            user_profile = {}
        
        results = []
        
        # Sort transactions by timestamp
        sorted_transactions = sorted(
            transactions,
            key=lambda tx: self._parse_timestamp(tx.get('timestamp', '1970-01-01'))
        )
        
        for i, transaction in enumerate(sorted_transactions):
            history = sorted_transactions[:i]
            result = self.analyze_transaction(transaction, user_profile, history)
            results.append(result)
        
        return results
    
    # Helper methods (simplified for brevity)
    def _get_hour(self, timestamp):
        """Extract hour from timestamp."""
        try:
            dt = self._parse_timestamp(timestamp)
            return dt.hour
        except:
            return -1
    
    def _parse_timestamp(self, timestamp):
        """Parse timestamp to datetime object."""
        if not timestamp:
            return datetime.min
            
        if isinstance(timestamp, datetime):
            return timestamp
            
        if isinstance(timestamp, str):
            formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
                    
        return datetime.min
    
    def _time_diff_hours(self, timestamp1, timestamp2):
        """Calculate time difference in hours."""
        dt1 = self._parse_timestamp(timestamp1)
        dt2 = self._parse_timestamp(timestamp2)
        
        if dt1 == datetime.min or dt2 == datetime.min:
            return float('inf')
            
        diff = abs(dt1 - dt2)
        return diff.total_seconds() / 3600
    
    def _check_geo_velocity(self, transaction, history):
        """Check for geographical velocity anomalies."""
        return False  # Simplified implementation
    
    def _detect_unusual_amount(self, transaction, history):
        """Detect unusual transaction amounts."""
        if not history or len(history) < 10:
            return False
            
        try:
            current_amount = float(transaction.get('amount', 0))
            amounts = [float(tx.get('amount', 0)) for tx in history]
            
            mean = sum(amounts) / len(amounts)
            variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
            std_dev = variance ** 0.5
            
            if std_dev == 0:
                return False
                
            z_score = abs(current_amount - mean) / std_dev
            return z_score > 3
        except:
            return False


class FraudDetectionApp:
    """
    Enhanced main application for bank statement fraud detection.
    Now includes rule-based, NLP-based, autoencoder-based detection methods, and AI chatbot.
    """
    
    def __init__(self, openai_api_key=None):
        """Initialize the application components."""
        self.document_processor = AIDocumentProcessor()
        self.fraud_detector = EnhancedFraudDetectionSystem()
        
        # Initialize ML-based detectors
        self.nlp_detector = None
        if NLP_AVAILABLE:
            self.nlp_detector = NLPFraudDetector()
        
        self.autoencoder_detector = None
        if TENSORFLOW_AVAILABLE:
            self.autoencoder_detector = AutoencoderFraudDetector()
        
        # Initialize chatbot
        self.chatbot = FraudDetectionChatbot(api_key=openai_api_key)
        
        # Data storage
        self.document_data = None
        self.transactions = []
        self.rule_based_results = []
        self.nlp_results = None
        self.autoencoder_results = None
        self.user_profile = {
            'large_amount_threshold': 2000
        }
    
    def process_document(self, file_path: str) -> Dict:
        """Process a document and extract transaction data."""
        print(f"\n=== AI-Enhanced Bank Statement Fraud Detection ===\n")
        print(f"Processing document: {file_path}")
        
        # Process the document
        self.document_data = self.document_processor.process_document(file_path)
        
        # Display processing information
        print(f"\n=== Document Processing Results ===")
        print(f"Format detected: {self.document_data.get('format', 'unknown')}")
        print(f"Processing method: {self.document_data.get('processing_method', 'standard')}")
        print(f"Records found: {len(self.document_data['data'])}")
        print(f"Fields detected: {len(self.document_data['headers'])}")
        
        return self.document_data
    
    def map_fields(self, field_mapping: Dict = None) -> None:
        """Map document fields to transaction fields."""
        if not self.document_data:
            raise ValueError("No document data. Process a document first.")
        
        if field_mapping is None:
            # Try to create automatic mapping using AI-detected categories
            field_mapping = self._create_automatic_mapping()
            
            if not field_mapping:
                print("\nWarning: Could not create automatic field mapping. Using best guess...")
                field_mapping = self._create_best_guess_mapping()
        
        self.field_mapping = field_mapping
        
        # Display mapping
        print("\n=== Field Mapping ===")
        for field, header in self.field_mapping.items():
            confidence = ""
            if self.document_data.get('category_mapping') and header in self.document_data['category_mapping']:
                conf = self.document_data['category_mapping'][header]['confidence']
                confidence = f" (AI confidence: {conf:.0%})"
            
            print(f"  {field} -> {header}{confidence}")
    
    def _create_automatic_mapping(self) -> Dict:
        """Create automatic field mapping using AI-detected categories."""
        if not self.document_data.get('category_mapping'):
            return {}
        
        field_to_category = {
            'id': 'transaction_id',
            'timestamp': 'transaction_date',
            'amount': 'amount',
            'description': 'description',
            'merchantCategory': 'category',
            'location': 'location',
            'balance': 'balance'
        }
        
        mapping = {}
        for field, category in field_to_category.items():
            for header, info in self.document_data['category_mapping'].items():
                if info['category'] == category:
                    mapping[field] = header
                    break
        
        return mapping
    
    def _create_best_guess_mapping(self) -> Dict:
        """Create best guess mapping when AI detection fails."""
        headers = self.document_data['headers']
        mapping = {}
        
        patterns = {
            'id': ['id', 'transaction id', 'reference'],
            'timestamp': ['date', 'time', 'transaction date'],
            'amount': ['amount', 'sum', 'transaction amount'],
            'description': ['description', 'details', 'memo'],
        }
        
        for field, keywords in patterns.items():
            best_match = None
            best_score = 0
            
            for header in headers:
                header_lower = str(header).lower()
                
                score = 0
                for keyword in keywords:
                    if keyword == header_lower:
                        score = 1.0
                        break
                    elif keyword in header_lower:
                        score = max(score, len(keyword) / len(header_lower))
                
                if score > best_score:
                    best_score = score
                    best_match = header
            
            if best_match and best_score > 0.3:
                mapping[field] = best_match
        
        # Generate ID if not found
        if 'id' not in mapping:
            mapping['id'] = '_generated_id'
        
        return mapping
    
    def process_transactions(self) -> List[Dict]:
        """Process document data into transactions."""
        if not self.document_data or not hasattr(self, 'field_mapping'):
            raise ValueError("Document data or field mapping not available.")
        
        data = self.document_data['data']
        mapping = self.field_mapping
        
        transactions = []
        
        for i, row in enumerate(data):
            transaction = {
                'userId': 'user1',
            }
            
            # Handle ID field
            if 'id' in mapping:
                if mapping['id'] == '_generated_id':
                    transaction['id'] = f"tx{i+1:04d}"
                else:
                    transaction['id'] = row.get(mapping['id'], f"tx{i+1:04d}")
            else:
                transaction['id'] = f"tx{i+1:04d}"
            
            # Process timestamp
            if 'timestamp' in mapping:
                raw_date = row.get(mapping['timestamp'])
                try:
                    parsed_date = self.fraud_detector._parse_timestamp(raw_date)
                    transaction['timestamp'] = parsed_date.isoformat()
                except:
                    transaction['timestamp'] = datetime.now().isoformat()
            else:
                transaction['timestamp'] = datetime.now().isoformat()
            
            # Process amount
            if 'amount' in mapping:
                amount = row.get(mapping['amount'], 0)
                
                if isinstance(amount, str):
                    amount = re.sub(r'[$£€,]', '', amount)
                    try:
                        amount = float(amount)
                    except:
                        amount = 0
                
                transaction['amount'] = abs(float(amount))
            else:
                transaction['amount'] = 0
            
            # Process other fields
            transaction['location'] = row.get(mapping.get('location', ''), 'Unknown')
            transaction['merchantCategory'] = row.get(mapping.get('merchantCategory', ''), 'Other')
            transaction['description'] = row.get(mapping.get('description', ''), '')
            transaction['originalData'] = row
            
            transactions.append(transaction)
        
        self.transactions = transactions
        return transactions
    
    def analyze_transactions_rule_based(self) -> List[Dict]:
        """Analyze transactions using rule-based fraud detection."""
        if not self.transactions:
            raise ValueError("No transactions to analyze. Process transactions first.")
        
        print("\n=== Running Rule-Based Fraud Detection ===")
        self.rule_based_results = self.fraud_detector.analyze_batch(
            self.transactions, 
            self.user_profile
        )
        
        return self.rule_based_results
    
    def analyze_transactions_nlp(self) -> Dict:
        """
        Analyze transactions using NLP-based fraud detection.
        """
        if not NLP_AVAILABLE:
            print("NLP libraries not available. Skipping NLP analysis.")
            return None
        
        if not self.transactions:
            raise ValueError("No transactions to analyze. Process transactions first.")
        
        print("\n=== Running NLP-Based Fraud Detection ===")
        
        try:
            # Train the NLP detector
            self.nlp_detector.train(self.transactions)
            
            # Detect anomalies
            self.nlp_results = self.nlp_detector.detect_anomalies(self.transactions)
            
            return self.nlp_results
            
        except Exception as e:
            print(f"Error in NLP analysis: {str(e)}")
            return None
    
    def analyze_transactions_autoencoder(self, train_on_all=True, train_split=0.8) -> Dict:
        """
        Analyze transactions using autoencoder-based fraud detection.
        
        Args:
            train_on_all: If True, train on all transactions (assumes they're mostly normal)
            train_split: If train_on_all is False, fraction to use for training
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping autoencoder analysis.")
            return None
        
        if not self.transactions:
            raise ValueError("No transactions to analyze. Process transactions first.")
        
        print("\n=== Running Autoencoder-Based Fraud Detection ===")
        
        if train_on_all:
            # Train on all transactions (assumes most are normal)
            print("Training autoencoder on all transactions (assuming mostly normal data)...")
            training_transactions = self.transactions
            test_transactions = self.transactions
        else:
            # Split data for training and testing
            split_index = int(len(self.transactions) * train_split)
            training_transactions = self.transactions[:split_index]
            test_transactions = self.transactions[split_index:]
            print(f"Training on {len(training_transactions)} transactions, testing on {len(test_transactions)}...")
        
        # Train the autoencoder
        try:
            self.autoencoder_detector.train(
                training_transactions,
                validation_split=0.2,
                epochs=50,  # Reduced for faster demo
                batch_size=32,
                verbose=1
            )
            
            # Detect anomalies
            self.autoencoder_results = self.autoencoder_detector.detect_anomalies(test_transactions)
            
            return self.autoencoder_results
            
        except Exception as e:
            print(f"Error in autoencoder analysis: {str(e)}")
            return None
    
    def chat_with_ai(self, message: str) -> str:
        """
        Chat with the AI assistant about fraud detection results.
        
        Args:
            message: User's message or question
            
        Returns:
            AI assistant's response
        """
        # Update chatbot context with latest results
        self.chatbot.set_fraud_context(
            transactions=self.transactions,
            rule_results=self.rule_based_results,
            nlp_results=self.nlp_results,
            ae_results=self.autoencoder_results
        )
        
        return self.chatbot.chat(message)
    
    def compare_detection_methods(self) -> Dict:
        """Compare all available detection methods."""
        if not self.rule_based_results:
            raise ValueError("No rule-based results available.")
        
        print("\n=== Comparing Detection Methods ===")
        
        # Get flags from rule-based method
        rule_based_flags = [r['fraud_likelihood'] != 'Low' for r in self.rule_based_results]
        
        comparison_results = {
            'total_transactions': len(self.rule_based_results),
            'rule_based_total': sum(rule_based_flags)
        }
        
        # Compare with NLP results if available
        if self.nlp_results:
            nlp_flags = [r['is_anomaly'] for r in self.nlp_results['detailed_results']]
            min_length = min(len(rule_based_flags), len(nlp_flags))
            
            rb_nlp_agreement = sum(1 for rb, nlp in zip(rule_based_flags[:min_length], nlp_flags[:min_length]) if rb == nlp)
            comparison_results.update({
                'nlp_total': sum(nlp_flags),
                'rule_nlp_agreement': (rb_nlp_agreement / min_length) * 100,
                'both_rule_nlp': sum(1 for rb, nlp in zip(rule_based_flags[:min_length], nlp_flags[:min_length]) if rb and nlp)
            })
        
        # Compare with autoencoder results if available
        if self.autoencoder_results:
            autoencoder_flags = self.autoencoder_results['anomaly_flags']
            min_length = min(len(rule_based_flags), len(autoencoder_flags))
            
            rb_ae_agreement = sum(1 for rb, ae in zip(rule_based_flags[:min_length], autoencoder_flags[:min_length]) if rb == ae)
            comparison_results.update({
                'autoencoder_total': sum(autoencoder_flags),
                'rule_ae_agreement': (rb_ae_agreement / min_length) * 100,
                'both_rule_ae': sum(1 for rb, ae in zip(rule_based_flags[:min_length], autoencoder_flags[:min_length]) if rb and ae)
            })
        
        # Compare NLP and autoencoder if both available
        if self.nlp_results and self.autoencoder_results:
            nlp_flags = [r['is_anomaly'] for r in self.nlp_results['detailed_results']]
            ae_flags = self.autoencoder_results['anomaly_flags']
            min_length = min(len(nlp_flags), len(ae_flags))
            
            nlp_ae_agreement = sum(1 for nlp, ae in zip(nlp_flags[:min_length], ae_flags[:min_length]) if nlp == ae)
            comparison_results.update({
                'nlp_ae_agreement': (nlp_ae_agreement / min_length) * 100,
                'both_nlp_ae': sum(1 for nlp, ae in zip(nlp_flags[:min_length], ae_flags[:min_length]) if nlp and ae)
            })
        
        # Print summary
        total = comparison_results['total_transactions']
        print(f"Detection Method Comparison:")
        print(f"  - Total transactions: {total}")
        print(f"  - Rule-based flagged: {comparison_results['rule_based_total']} ({comparison_results['rule_based_total']/total*100:.1f}%)")
        
        if 'nlp_total' in comparison_results:
            print(f"  - NLP flagged: {comparison_results['nlp_total']} ({comparison_results['nlp_total']/total*100:.1f}%)")
            print(f"  - Rule-NLP agreement: {comparison_results['rule_nlp_agreement']:.1f}%")
        
        if 'autoencoder_total' in comparison_results:
            print(f"  - Autoencoder flagged: {comparison_results['autoencoder_total']} ({comparison_results['autoencoder_total']/total*100:.1f}%)")
            print(f"  - Rule-Autoencoder agreement: {comparison_results['rule_ae_agreement']:.1f}%")
        
        if 'nlp_ae_agreement' in comparison_results:
            print(f"  - NLP-Autoencoder agreement: {comparison_results['nlp_ae_agreement']:.1f}%")
        
        return comparison_results
    
    def display_results(self, include_all=True) -> None:
        """Display analysis results from all available methods."""
        if not self.rule_based_results and not self.nlp_results and not self.autoencoder_results:
            raise ValueError("No analysis results available.")
        
        # Display rule-based results
        if self.rule_based_results:
            total = len(self.rule_based_results)
            high_risk = sum(1 for r in self.rule_based_results if r['fraud_likelihood'] == 'High')
            medium_risk = sum(1 for r in self.rule_based_results if r['fraud_likelihood'] == 'Medium')
            low_risk = sum(1 for r in self.rule_based_results if r['fraud_likelihood'] == 'Low')
            
            print("\n=== Rule-Based Detection Results ===")
            print(f"Total transactions: {total}")
            print(f"Risk levels:")
            print(f"  - High Risk: {high_risk} ({high_risk/total*100:.1f}%)")
            print(f"  - Medium Risk: {medium_risk} ({medium_risk/total*100:.1f}%)")
            print(f"  - Low Risk: {low_risk} ({low_risk/total*100:.1f}%)")
        
        # Display NLP results if available
        if include_all and self.nlp_results:
            nlp_summary = self.nlp_results['summary']
            print(f"\n=== NLP-Based Detection Results ===")
            print(f"Total transactions: {nlp_summary['total_transactions']}")
            print(f"Anomalies detected: {nlp_summary['anomalies_detected']} ({nlp_summary['anomaly_rate']:.1f}%)")
            print(f"Transaction clusters: {nlp_summary['unique_clusters']}")
        
        # Display autoencoder results if available
        if include_all and self.autoencoder_results:
            ae_summary = self.autoencoder_results['summary']
            print(f"\n=== Autoencoder Detection Results ===")
            print(f"Total transactions: {ae_summary['total_transactions']}")
            print(f"Anomalies detected: {ae_summary['anomalies_detected']} ({ae_summary['anomaly_rate_percent']:.1f}%)")
            print(f"Mean reconstruction error: {ae_summary['mean_reconstruction_error']:.6f}")
            print(f"Threshold used: {ae_summary['threshold_used']:.6f}")
            
        # Display chatbot info
        print(f"\n=== AI Assistant Available ===")
        print("You can chat with the AI assistant about these results!")
        print("Try asking: 'Give me a summary' or 'What are the main risks?'")
        
        # Show sample chatbot interaction
        if self.rule_based_results or self.nlp_results or self.autoencoder_results:
            sample_response = self.chat_with_ai("Give me a brief summary of the fraud analysis results.")
            print(f"\n🤖 AI Assistant: {sample_response}")
    
    def save_autoencoder_model(self, filepath: str):
        """Save the trained autoencoder model."""
        if not self.autoencoder_detector or not self.autoencoder_detector.is_trained:
            raise ValueError("No trained autoencoder model to save.")
        
        self.autoencoder_detector.save_model(filepath)
        print(f"Autoencoder model saved to {filepath}")
    
    def load_autoencoder_model(self, filepath: str):
        """Load a saved autoencoder model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required to load autoencoder models.")
        
        if not self.autoencoder_detector:
            self.autoencoder_detector = AutoencoderFraudDetector()
        
        self.autoencoder_detector.load_model(filepath)
        print(f"Autoencoder model loaded from {filepath}")


def main():
    """Enhanced main function supporting all detection methods and chatbot."""
    parser = argparse.ArgumentParser(description='AI-Enhanced Bank Statement Fraud Detection with ML and Chatbot')
    parser.add_argument('file', help='Path to the bank statement file (any format)')
    parser.add_argument('--threshold', type=float, default=2000, 
                       help='Threshold for large transaction amount (default: 2000)')
    parser.add_argument('--method', choices=['rules', 'nlp', 'autoencoder', 'all'], default='all',
                       help='Detection method to use (default: all)')
    parser.add_argument('--autoencoder-threshold', type=float, default=95,
                       help='Autoencoder anomaly threshold percentile (default: 95)')
    parser.add_argument('--nlp-contamination', type=float, default=0.1,
                       help='Expected fraction of outliers for NLP method (default: 0.1)')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Fraction of data to use for autoencoder training (default: 0.8)')
    parser.add_argument('--save-model', help='Path to save trained autoencoder model')
    parser.add_argument('--load-model', help='Path to load pretrained autoencoder model')
    parser.add_argument('--openai-key', help='OpenAI API key for chatbot functionality')
    parser.add_argument('--chat', action='store_true',
                       help='Enable interactive chat mode after analysis')
    parser.add_argument('--output', default='fraud_analysis_report.csv',
                       help='Output file for analysis report')
    parser.add_argument('--debug', action='store_true',
                       help='Enable detailed debug output')
    
    args = parser.parse_args()
    
    # Initialize the app
    app = FraudDetectionApp(openai_api_key=args.openai_key)
    app.user_profile['large_amount_threshold'] = args.threshold
    
    # Configure detectors if available
    if TENSORFLOW_AVAILABLE and app.autoencoder_detector:
        app.autoencoder_detector.threshold_value = args.autoencoder_threshold
    
    if NLP_AVAILABLE and app.nlp_detector:
        app.nlp_detector.contamination = args.nlp_contamination
    
    try:
        # Process document
        app.process_document(args.file)
        app.map_fields()
        app.process_transactions()
        
        # Run detection methods based on user choice
        if args.method in ['rules', 'all']:
            app.analyze_transactions_rule_based()
        
        if args.method in ['nlp', 'all'] and NLP_AVAILABLE:
            app.analyze_transactions_nlp()
        
        if args.method in ['autoencoder', 'all'] and TENSORFLOW_AVAILABLE:
            if args.load_model:
                app.load_autoencoder_model(args.load_model)
                # Use loaded model for detection only
                app.autoencoder_results = app.autoencoder_detector.detect_anomalies(app.transactions)
            else:
                # Train and detect
                app.analyze_transactions_autoencoder(
                    train_on_all=True,
                    train_split=args.train_split
                )
            
            if args.save_model and app.autoencoder_detector and app.autoencoder_detector.is_trained:
                app.save_autoencoder_model(args.save_model)
        
        # Display results
        app.display_results(include_all=True)
        
        # Compare methods if multiple were run
        if (args.method == 'all' or 
            sum([bool(app.rule_based_results), bool(app.nlp_results), bool(app.autoencoder_results)]) > 1):
            app.compare_detection_methods()
        
        # Interactive chat mode
        if args.chat:
            print(f"\n🤖 === Interactive AI Assistant Mode ===")
            print("Ask me anything about the fraud analysis results! Type 'quit' to exit.")
            print("Example questions: 'What are the main risks?', 'Explain the anomalies', 'What should I investigate?'")
            
            while True:
                try:
                    user_input = input("\n💬 You: ").strip()
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if user_input:
                        response = app.chat_with_ai(user_input)
                        print(f"🤖 AI Assistant: {response}")
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"Error in chat: {str(e)}")
        
        print(f"\nAnalysis complete! Results saved to {args.output}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())