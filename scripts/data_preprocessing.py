import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config

class DataPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        try:
            tokens = word_tokenize(text)
        except LookupError:
            print("Downloading NLTK punkt...")
            nltk.download('punkt')
            tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                  if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def convert_mbti_to_big5(self, mbti_type):
        """Convert MBTI to Big Five approximation"""
        traits = {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5
        }
        
        if len(mbti_type) != 4:
            return traits
        
        # E/I axis -> Extraversion
        traits['extraversion'] = 0.8 if mbti_type[0] == 'E' else 0.2
        
        # S/N axis -> Openness
        traits['openness'] = 0.8 if mbti_type[1] == 'N' else 0.3
        
        # T/F axis -> Agreeableness
        traits['agreeableness'] = 0.7 if mbti_type[2] == 'F' else 0.3
        
        # J/P axis -> Conscientiousness
        traits['conscientiousness'] = 0.8 if mbti_type[3] == 'J' else 0.3
        
        # Neuroticism
        if mbti_type[0] == 'I' and mbti_type[2] == 'F':
            traits['neuroticism'] = 0.6
        else:
            traits['neuroticism'] = 0.3
        
        return traits
    
    def process_dataset(self, input_path, output_path):
        """Main processing pipeline"""
        print("Loading dataset...")
        
        if not os.path.exists(input_path):
            print(f"❌ Error: Dataset not found at {input_path}")
            print("\nPlease run: python scripts/download_dataset.py")
            return None
        
        df = pd.read_csv(input_path)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Clean and preprocess text
        print("Cleaning text...")
        df['cleaned_text'] = df['posts'].apply(self.clean_text)
        
        print("Tokenizing and lemmatizing...")
        df['processed_text'] = df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        
        # Convert MBTI to Big Five
        print("Converting MBTI to Big Five traits...")
        traits_df = pd.DataFrame([self.convert_mbti_to_big5(mbti) 
                                  for mbti in df['type']])
        
        # Combine with original dataframe
        df = pd.concat([df, traits_df], axis=1)
        
        # Remove rows with empty text
        df = df[df['processed_text'].str.len() > 50]
        
        # Save processed data
        print(f"Saving processed data to {output_path}")
        df.to_csv(output_path, index=False)
        
        print(f"Processed dataset shape: {df.shape}")
        print("\nSample data:")
        print(df[['type', 'openness', 'conscientiousness', 'extraversion']].head())
        
        return df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.process_dataset(
        Config.DATASET_PATH,
        Config.PROCESSED_DATA_PATH
    )
    if processed_df is not None:
        print("\n✓ Preprocessing complete!")
    else:
        print("\n❌ Preprocessing failed!")
