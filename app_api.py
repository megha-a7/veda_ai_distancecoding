from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# load the trained machine learning model
model_path = 'model.pkl'
model = joblib.load(model_path)
model = joblib.load('model.pkl')
# define the FastAPI app
app = FastAPI()

# define the input data schema
class InputData(BaseModel):
    model: model
    temperature:int
    prompt: str
    max_tokens:int

# define the prediction endpoint
@app.post("/predict")
async def predict(input_data: InputData):
    features = [input_data.model, input_data.temperature, input_data.prompt, input_data.max_tokens]
    prediction = model.predict([features])[0]
    return {"prediction": prediction}
