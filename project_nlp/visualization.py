import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix

class Visualizer:
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_sentiment_distribution(self, df, sentiment_col='sentiment'):
        """Plot distribution of sentiments"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        sentiment_counts = df[sentiment_col].value_counts()
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        ax1.set_title('Sentiment Distribution')
        
        # Bar plot
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax2)
        ax2.set_title('Sentiment Counts')
        ax2.set_xlabel('Sentiment')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        return fig
    
    def plot_word_frequency(self, df, text_col='cleaned_text', top_n=20):
        """Plot most frequent words"""
        all_words = ' '.join(df[text_col]).split()
        word_freq = Counter(all_words)
        common_words = word_freq.most_common(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        words, counts = zip(*common_words)
        sns.barplot(x=list(counts), y=list(words), ax=ax)
        ax.set_title(f'Top {top_n} Most Frequent Words')
        ax.set_xlabel('Frequency')
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        return fig
    
    def plot_sentiment_comparison(self, results):
        """Compare model predictions"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(results.keys())
        accuracies = list(results.values())
        
        bars = ax.bar(methods, accuracies, color=['skyblue', 'lightcoral'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Sentiment Analysis Method Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{accuracy:.2%}', ha='center', va='bottom')
        
        return fig