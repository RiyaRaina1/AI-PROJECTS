import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
import joblib

class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def train_model(self, X, y):
        """Train the sentiment analysis model"""
        # Vectorize text data
        X_vectorized = self.vectorizer.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        return accuracy, classification_report(y_test, y_pred)
    
    def predict_sentiment(self, text):
        """Predict sentiment for new text"""
        if not self.is_trained:
            return "Model not trained yet"
        
        # Vectorize text
        text_vectorized = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.model.predict(text_vectorized)[0]
        probability = self.model.predict_proba(text_vectorized)[0]
        
        return prediction, probability
    
    def get_textblob_sentiment(self, text):
        """Get sentiment using TextBlob (rule-based)"""
        analysis = TextBlob(text)
        
        if analysis.sentiment.polarity > 0.1:
            return 'positive', analysis.sentiment.polarity
        elif analysis.sentiment.polarity < -0.1:
            return 'negative', analysis.sentiment.polarity
        else:
            return 'neutral', analysis.sentiment.polarity
    
    def save_model(self, filename):
        """Save trained model"""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.model
        }, filename)
    
    def load_model(self, filename):
        """Load trained model"""
        loaded = joblib.load(filename)
        self.vectorizer = loaded['vectorizer']
        self.model = loaded['model']
        self.is_trained = True