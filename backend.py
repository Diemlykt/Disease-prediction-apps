import traceback
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'Model')
sys.path.append(MODEL_DIR)
import torch
from fastapi import Form, FastAPI, UploadFile, File, HTTPException
from pymongo import MongoClient
import gridfs
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from bson import ObjectId
from model2use import CNN  # Import your model definitions
import joblib
from fastapi.middleware.cors import CORSMiddleware
import xgboost as xgb
from fastapi.responses import JSONResponse



# Get the directory where backend.py is located

# Construct full path to the model file
CNN_path = os.path.join(BASE_DIR, 'Model', 'CNN.pt')
scaler_path = os.path.join(BASE_DIR, 'Model', 'scaler.pkl')
xgboost_path = os.path.join(BASE_DIR, 'Model', 'xgboost_model.pkl')

# Example: load a PyTorch model

app = FastAPI()

#to keep render alive
@app.get("/ping")
async def ping():
    return {"message": "pong"}
    
client = MongoClient("mongodb+srv://diemly:fQg9TNKzmmRd9g9M@alzheimer.x2velvm.mongodb.net/?retryWrites=true&w=majority&appName=Alzheimer")
db = client["health_data"]
fs = gridfs.GridFS(db)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load(scaler_path)
clinical_model = xgb.XGBClassifier()
clinical_model.load_model("Model/xgboost_model.json")
# clinical_model = joblib.load(xgboost_path)
# imaging_model = CNN().to(device)
# imaging_model.load_state_dict(torch.load(CNN_path, map_location=device))
# imaging_model.eval()

# Image preprocessing
transform = transforms.Compose([
        transforms.Resize((248, 248)),
        transforms.ToTensor(),
    ])

feature_columns = [
    'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
    'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
    'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes',
    'Depression', 'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP',
    'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
    'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
    'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion',
    'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
    'Forgetfulness'
]

# Select numeric columns to scale
numeric_features_to_scale = [
    'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
    'SleepQuality', 'SystolicBP', 'DiastolicBP',
    'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
    'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL'
]

def get_imaging_model():
    model = CNN().to(device)
    model.load_state_dict(torch.load(CNN_path, map_location=device))
    model.eval()
    return model

@app.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
        df.fillna(0, inplace=True)
        # df["patient_id"] = [f"P{i:03d}" for i in range(len(df))]
        records = df.to_dict("records")
        result = db["records"].insert_many(records)
        return {"status": f"Uploaded {len(records)} records", "ids": [str(id) for id in result.inserted_ids], "patient_ids": df["PatientID"].tolist()}
    raise HTTPException(status_code=400, detail="Invalid file")


@app.post("/upload/image")
async def upload_image(
    patient_id: str = Form(...),  # Accept as form data
    file: UploadFile = File(...)
):
    try:
        # File validation
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            return JSONResponse(
                status_code=400,
                content={"success": False, "detail": "Invalid file type"}
            )

        # Read file
        contents = await file.read()

        # Database operation
        image_id = db["images"].insert_one({
            "filename": file.filename,
            "content": contents,
            "patient_id": patient_id
             }).inserted_id

        return {
            "success": True,
            "message": "Image uploaded successfully",
            "image_id": str(image_id)
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": str(e)}
        )

@app.post("/predict/clinical")
async def predict_clinical(patient_id: str):
    # Retrieve patient record
    patient_id_int = int(patient_id)
    record = db["records"].find_one({"PatientID": patient_id_int})
    if not record:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Build a DataFrame with one row
    try:
        input_data = {col: record[col] for col in feature_columns}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e.args[0]}")

    df = pd.DataFrame([input_data])

    # Scale numeric columns
    df[numeric_features_to_scale] = scaler.transform(df[numeric_features_to_scale])

    # Predict using XGBoost model
    prediction = clinical_model.predict(df)[0]
    probabilities = clinical_model.predict_proba(df)[0]
    class_names = ['Yes', 'No']
    return {
    "patient_id": patient_id,
    "prediction": class_names[prediction],
    "probabilities": {
        class_names[i]: float(probabilities[i]) for i in range(len(class_names))
    }
}

@app.post("/predict/image")
async def predict_image(image_id: str):
    image_doc = db["images"].find_one({"image_id": ObjectId(image_id)})
    if not image_doc:
        raise HTTPException(status_code=404, detail="Image not found")
    image_data = fs.get(image_doc["image_id"]).read()
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    image_tensor = transform(img).to(device)
    with torch.no_grad():
        mean = torch.mean(image_tensor).unsqueeze(0)
        std = torch.std(image_tensor).unsqueeze(0)
        imaging_model = get_imaging_model()
        outputs = imaging_model(image_tensor.unsqueeze(0), mean, std).squeeze().cpu().numpy()
        pred = torch.argmax(outputs, dim=1)
    class_names = ['Non-Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']
    return {
        "image_id": image_id,
        "prediction": class_names[pred],
        "probabilities": {class_names[i]: float(outputs[i]) for i in range(len(class_names))}
    }
