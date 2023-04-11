import requests
import json

# Define the input data
input_data = {
    "model": 0.5,
    "temperature": 1.2,
    "max_token": -0.2
}

# Send a POST request to the /predict endpoint
response = requests.post('http://localhost:8000/predict', json=input_data)

# Check the response status code
if response.status_code == 200:
    # Parse the response JSON and print the predictions
    predictions = json.loads(response.content)
    print(predictions)
else:
    print('Error:', response.content)
