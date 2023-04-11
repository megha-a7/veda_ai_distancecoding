from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the trained machine learning model
model_path = 'model.pkl'
model = joblib.load(model_path)

# Define the FastAPI app
app = FastAPI()

class InputData(BaseModel):
    question: str

class OutputData(BaseModel):
    response: str

@app.post("/predict", response_model=OutputData)
async def predict(input_data: InputData):
    question = input_data.question
    # Use your model to predict the response based on the input questio+
    # n
    response = model.predict([question])[0]
    return {"response": response}

@app.post("/test")
async def test():
    return {"message": "test"}

@app.get("/")
async def read_root():
    return {"message": "Welcome to my API!"}
@app.get("/predict", response_model=OutputData)
async def predict(input_data: InputData):
    question = input_data.question
    # Use your model to predict the response based on the input questio+
    # n
    response = model.predict([question])[0]
    return {"response": response}
