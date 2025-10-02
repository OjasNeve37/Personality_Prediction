from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from pathlib import Path
import traceback

# Add the project root to Python path
project_root = str(Path(__file__).resolve().parents[3])
sys.path.append(project_root)

# Now import the model and config
from scripts.train_model import BERTPersonalityClassifier
from scripts.data_preprocessing import DataPreprocessor
from config import Config
import torch
from transformers import BertTokenizer

app = FastAPI(title="Personality Prediction API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
try:
    model_path = os.path.join(project_root, "models", "saved_models", "best_model.pt")
    print(f"Loading model from: {model_path}")
    
    class PersonalityPredictor:
        def __init__(self):
            self.device = Config.DEVICE
            self.tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
            self.model = BERTPersonalityClassifier(num_traits=Config.NUM_TRAITS)
            
            print("Loading model checkpoint...")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        
        async def predict_from_text(self, text: str):
            try:
                print(f"[PREDICTOR] Processing text of length: {len(text)}")
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    max_length=Config.MAX_LENGTH,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                
                print("[PREDICTOR] Tokenization complete, running prediction...")
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    predictions = {}
                    for i, trait in enumerate(Config.TRAITS):
                        score = float(torch.sigmoid(outputs[:, i]).mean())
                        predictions[trait] = score
                        # The model already applies sigmoid, so we just take the output
                        score = float(outputs[0, i].item())
                        predictions[trait] = score # The model already applies sigmoid
                        print(f"[PREDICTOR] {trait}: {score:.4f}")
                
                return predictions
            except Exception as e:
                print(f"[PREDICTOR] Error in predict_from_text: {str(e)}")
                traceback.print_exc()
                raise
        
        def predict_from_resume(self, resume_path: str):
            from scripts.resume_parser import parse_resume
            text = parse_resume(resume_path)
            if not text:
                return None
            return self.predict_from_text(text)
    
    predictor = PersonalityPredictor()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    raise

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Personality Prediction API"}

@app.post("/predict/text")
async def predict_from_text(text: str):
    predictions = await predictor.predict_from_text(text)
    return {
        "status": "success",
        "predictions": predictions
    }

@app.post("/predict/resume")
async def predict_from_resume(file: UploadFile = File(...)):
    temp_path = None
    try:
        print(f"[API] Starting file upload process for: {file.filename}")
        # Verify file type
        if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
            error_msg = "Invalid file format. Please upload a PDF, DOC, or DOCX file."
            print(f"[API] {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )

        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(project_root, "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"[API] Created/verified temp directory: {temp_dir}")
        
        # Save uploaded file with absolute path
        temp_path = os.path.join(temp_dir, f"temp_{file.filename}")
        print(f"[API] Attempting to save file to: {temp_path}")
        
        content = await file.read()
        print(f"[API] Read {len(content)} bytes from uploaded file")
        
        with open(temp_path, "wb") as buffer:
            buffer.write(content)
        print(f"[API] Successfully saved file to: {temp_path}")
        
        # Parse resume
        print("[API] Starting resume parsing...")
        from scripts.resume_parser import parse_resume
        try:
            resume_text = parse_resume(temp_path)
            if not resume_text:
                error_msg = "Could not extract text from the resume. Please check the file format."
                print(f"[API] {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
            
            print(f"[API] Successfully extracted text, length: {len(resume_text)}")
            print(f"[API] First 100 chars of text: {resume_text[:100]}")
            
            # Get predictions
            print("[API] Starting prediction...")
            predictions = await predictor.predict_from_text(resume_text)
            print(f"[API] Predictions completed: {predictions}")
            
            return {
                "status": "success",
                "predictions": predictions
            }
        except Exception as e:
            error_msg = f"Error in resume parsing or prediction: {str(e)}"
            print(f"[API] {error_msg}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=error_msg)
        
    except Exception as e:
        print(f"Error processing resume: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Removed temporary file: {temp_path}")
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")
                traceback.print_exc()