from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Get root path from Azure environment
root_path = os.getenv("FASTAPI_ROOT_PATH", "")

app = FastAPI(root_path=root_path)  # ✅ Fixes issues with base paths

@app.get("/")
def home():
    return {"message": "FastAPI is running on Azure!"}

# Load ML model
model = joblib.load("iris_model.pkl")

class IrisInput(BaseModel):
    features: list

@app.post("/predict")  # ✅ Ensure this is the correct route
async def predict(iris_input: IrisInput):
    data = np.array(iris_input.features).reshape(1, -1)
    prediction = model.predict(data).tolist()
    return {"prediction": prediction}
