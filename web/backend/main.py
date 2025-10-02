from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from pathlib import Path
import torch

# Add the project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

# Now we can import from the main project
from scripts.train_model import BERTPersonalityClassifier
from scripts.data_preprocessing import DataPreprocessor
from scripts.resume_parser import parse_resume
from config import Config

app = FastAPI(title="Personality Prediction API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
try:
    model_path = os.path.join(project_root, "models", "saved_models", "best_model.pt")
    print(f"Loading model from: {model_path}")
    
    class PersonalityPredictorAPI:
        def __init__(self):
            self.device = Config.DEVICE
            self.model = BERTPersonalityClassifier(num_traits=Config.NUM_TRAITS)
            self.preprocessor = DataPreprocessor()
            
            print("Loading model checkpoint...")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")

    predictor = PersonalityPredictorAPI()
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    raise

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Personality Prediction API"}

@app.post("/predict/text")
async def predict_from_text(text: str):
    try:
        predictions = predictor.predict_from_text(text)
        return {
            "status": "success",
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/resume")
async def predict_from_resume(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Make prediction
        predictions = predictor.predict_from_resume(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "status": "success",
            "predictions": predictions
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))