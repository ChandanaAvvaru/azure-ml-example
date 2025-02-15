from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("iris_model.pkl")

class IrisInput(BaseModel):
    features: list

@app.post("/predict")
async def predict(iris_input: IrisInput):
    data = np.array(iris_input.features).reshape(1, -1)
    prediction = model.predict(data).tolist()
    return {"prediction": prediction}
