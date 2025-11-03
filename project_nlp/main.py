import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import pandas as pd
from data_processor import DataProcessor
from sentiment_model import SentimentModel
from visualization import Visualizer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SentimentAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis Tool")
        self.root.geometry("1000x700")
        
        # Initialize components
        self.processor = DataProcessor()
        self.model = SentimentModel()
        self.visualizer = Visualizer()
        self.df = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Data Tab
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="Data Management")
        self.setup_data_tab(data_frame)
        
        # Analysis Tab
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Sentiment Analysis")
        self.setup_analysis_tab(analysis_frame)
        
        # Visualization Tab
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="Visualization")
        self.setup_visualization_tab(viz_frame)
    
    def setup_data_tab(self, parent):
        """Setup data management tab"""
        # Load data section
        load_frame = ttk.LabelFrame(parent, text="Load Data", padding=10)
        load_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(load_frame, text="Load Sample Data", 
                  command=self.load_sample_data).pack(side='left', padx=5)
        ttk.Button(load_frame, text="Load CSV File", 
                  command=self.load_csv_file).pack(side='left', padx=5)
        
        # Data preview
        preview_frame = ttk.LabelFrame(parent, text="Data Preview", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.data_text = scrolledtext.ScrolledText(preview_frame, height=15)
        self.data_text.pack(fill='both', expand=True)
    
    def setup_analysis_tab(self, parent):
        """Setup analysis tab"""
        # Training section
        train_frame = ttk.LabelFrame(parent, text="Model Training", padding=10)
        train_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(train_frame, text="Train Model", 
                  command=self.train_model).pack(side='left', padx=5)
        
        # Single text analysis
        analysis_frame = ttk.LabelFrame(parent, text="Text Analysis", padding=10)
        analysis_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(analysis_frame, text="Enter text to analyze:").pack(anchor='w')
        self.analysis_text = scrolledtext.ScrolledText(analysis_frame, height=4)
        self.analysis_text.pack(fill='x', pady=5)
        
        ttk.Button(analysis_frame, text="Analyze Sentiment", 
                  command=self.analyze_single_text).pack(pady=5)
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(analysis_frame, height=8)
        self.results_text.pack(fill='x', pady=5)
    
    def setup_visualization_tab(self, parent):
        """Setup visualization tab"""
        # Visualization controls
        viz_controls = ttk.Frame(parent)
        viz_controls.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(viz_controls, text="Show Sentiment Distribution",
                  command=self.show_sentiment_dist).pack(side='left', padx=2)
        ttk.Button(viz_controls, text="Show Word Frequency",
                  command=self.show_word_freq).pack(side='left', padx=2)
        ttk.Button(viz_controls, text="Show Confusion Matrix",
                  command=self.show_confusion_matrix).pack(side='left', padx=2)
        
        # Plot area
        self.plot_frame = ttk.Frame(parent)
        self.plot_frame.pack(fill='both', expand=True, padx=5, pady=5)
    
    def load_sample_data(self):
        """Load sample dataset"""
        self.df = self.processor.load_sample_data()
        self.df = self.processor.process_dataset(self.df)
        self.update_data_preview()
        messagebox.showinfo("Success", "Sample data loaded successfully!")
    
    def load_csv_file(self):
        """Load data from CSV file"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                # Assume the CSV has 'text' and 'sentiment' columns
                if 'text' in self.df.columns:
                    self.df = self.processor.process_dataset(self.df)
                    self.update_data_preview()
                    messagebox.showinfo("Success", "CSV file loaded successfully!")
                else:
                    messagebox.showerror("Error", "CSV must contain 'text' column")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def update_data_preview(self):
        """Update data preview text widget"""
        if self.df is not None:
            preview_text = f"Dataset Shape: {self.df.shape}\n\n"
            preview_text += self.df.head(10).to_string()
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(1.0, preview_text)
    
    def train_model(self):
        """Train the sentiment analysis model"""
        if self.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        try:
            accuracy, report = self.model.train_model(
                self.df['cleaned_text'], 
                self.df['sentiment']
            )
            
            messagebox.showinfo(
                "Training Complete", 
                f"Model trained successfully!\nAccuracy: {accuracy:.2%}"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def analyze_single_text(self):
        """Analyze sentiment for single text input"""
        text = self.analysis_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to analyze!")
            return
        
        if not self.model.is_trained:
            messagebox.showwarning("Warning", "Please train the model first!")
            return
        
        # Get predictions
        ml_prediction, ml_probability = self.model.predict_sentiment(text)
        tb_prediction, tb_polarity = self.model.get_textblob_sentiment(text)
        
        # Display results
        result_text = f"Input Text: {text}\n\n"
        result_text += "Machine Learning Model:\n"
        result_text += f"  Prediction: {ml_prediction}\n"
        result_text += f"  Probabilities: {dict(zip(self.model.model.classes_, ml_probability))}\n\n"
        
        result_text += "TextBlob Analysis:\n"
        result_text += f"  Prediction: {tb_prediction}\n"
        result_text += f"  Polarity: {tb_polarity:.3f}\n"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, result_text)
    
    def show_sentiment_dist(self):
        """Show sentiment distribution plot"""
        if self.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        self.clear_plot_frame()
        fig = self.visualizer.plot_sentiment_distribution(self.df)
        self.display_plot(fig)
    
    def show_word_freq(self):
        """Show word frequency plot"""
        if self.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        self.clear_plot_frame()
        fig = self.visualizer.plot_word_frequency(self.df)
        self.display_plot(fig)
    
    def show_confusion_matrix(self):
        """Show confusion matrix (placeholder)"""
        messagebox.showinfo("Info", "This would show confusion matrix after model evaluation")
    
    def clear_plot_frame(self):
        """Clear the plot frame"""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
    
    def display_plot(self, fig):
        """Display matplotlib figure in Tkinter"""
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

def main():
    root = tk.Tk()
    app = SentimentAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()