import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk # type: ignore

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DataProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(cleaned_tokens)
    
    def load_sample_data(self):
        """Create sample dataset for demonstration"""
        data = {
            'text': [
                'I love this product! It is amazing and works perfectly.',
                'This is the worst experience ever. Very disappointed.',
                'The product is okay, nothing special but gets the job done.',
                'Excellent quality and fast delivery. Highly recommended!',
                'Poor customer service and defective product.',
                'Not bad for the price, but could be better.',
                'Absolutely fantastic! Worth every penny.',
                'Terrible quality, broke after first use.',
                'Good value for money, satisfied with purchase.',
                'Mediocre product, expected more features.'
            ],
            'sentiment': ['positive', 'negative', 'neutral', 'positive', 
                         'negative', 'neutral', 'positive', 'negative', 
                         'positive', 'neutral']
        }
        return pd.DataFrame(data)
    
    def process_dataset(self, df):
        """Process the entire dataset"""
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        return df