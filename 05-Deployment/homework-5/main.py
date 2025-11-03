import pickle
import os
from fastapi import FastAPI
from pydantic import BaseModel

# Get the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "pipeline_v1.bin")

# Load the pipeline
with open(PIPELINE_PATH, "rb") as f_in:
    pipeline = pickle.load(f_in)

# Create FastAPI app
app = FastAPI()

# Define input schema
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(lead: Lead):
    lead_dict = lead.dict()
    probability = pipeline.predict_proba([lead_dict])[0, 1]
    return {"conversion_probability": float(probability)}

# Optional root route (just to confirm server is running)
@app.get("/")
def root():
    return {"message": "API is live. Go to /docs to test predictions."}