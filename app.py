from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")

# Define input schema
class InputData(BaseModel):
    data: list

@app.post("/predict")
def predict(input: InputData):
    features = np.array(input.data)
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}