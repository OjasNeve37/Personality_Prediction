import torch
from transformers import BertTokenizer
import sys
import os

# Fix import path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.train_model import BERTPersonalityClassifier
from scripts.data_preprocessing import DataPreprocessor
from scripts.resume_parser import parse_resume
from config import Config


class PersonalityPredictor:
    def __init__(self, model_path):
        self.device = Config.DEVICE
        self.tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
        self.preprocessor = DataPreprocessor()
        
        print(f"Loading model from {model_path}...")
        self.model = BERTPersonalityClassifier(num_traits=Config.NUM_TRAITS)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def predict_from_text(self, text):
        cleaned_text = self.preprocessor.clean_text(text)
        processed_text = self.preprocessor.tokenize_and_lemmatize(cleaned_text)
        
        encoding = self.tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            predictions = outputs.cpu().numpy()[0]
        
        return {trait: float(score) for trait, score in zip(Config.TRAITS, predictions)}
    
    def predict_from_resume(self, resume_path):
        resume_data = parse_resume(resume_path)
        
        if resume_data is None:
            print("Failed to parse resume.")
            return None
        
        text = resume_data['raw_text']
        if not text or len(text) < 50:
            print("Extracted resume text is too short for accurate prediction.")
            return None
        
        print(f"Predicting personality traits for resume: {resume_path}")
        return self.predict_from_text(text)
    
    def print_results(self, results):
        print("\n" + "="*70)
        print("PERSONALITY PREDICTION RESULTS")
        print("="*70)
        
        if 'candidate_info' in results:
            print("\nCANDIDATE INFORMATION:")
            for key, value in results['candidate_info'].items():
                print(f"  {key}: {value}")
            results = results['personality_traits']
        
        print("\nBIG FIVE PERSONALITY TRAITS:")
        for trait, score in results.items():
            bar_length = int(score * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {trait.capitalize():20s} [{bar}] {score:.2f}")
        
        print("="*70)


def main():
    model_path = f"{Config.MODEL_DIR}/best_model.pt"
    predictor = PersonalityPredictor(model_path)
    
    # Uncomment to predict from text
    # sample_text = """
    # I am a software engineer with 5 years of experience in Python and machine learning.
    # I enjoy working on challenging problems and collaborating with teams.
    # I have strong communication skills and love learning new technologies.
    # """
    # results = predictor.predict_from_text(sample_text)
    # predictor.print_results(results)
    
    # Predict from resume PDF (provide your file path here)
    resume_path = r"C:\Users\Ojas Neve\OneDrive\ドキュメント\personal\Projects\Personality_Prediction\data\raw\Ojas Resume.pdf"  # Change to your resume file path
    results = predictor.predict_from_resume(resume_path)
    if results:
        predictor.print_results({'candidate_info': {'Resume Path': resume_path}, 'personality_traits': results})
    else:
        print("Prediction failed.")

if __name__ == "__main__":
    main()
