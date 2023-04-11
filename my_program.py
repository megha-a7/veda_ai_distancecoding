import openai
import json
openai.api_key = "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"
# Load the training data from a JSON file
with open("training_data.json", "r") as f:
    training_data = json.load(f)

# Define the fine-tuning parameters
model = "text-davinci-002"
model_id = "text-davinci-003"
training_data = training_data
epochs = 3

# Create the fine-tuning task
response = openai.FineTune.create(
    model=model,
    model_id=model_id,
    prompt_language="en",
    training_data=training_data,
    epochs=epochs,
)

# Print the task ID
print(f"Task ID: {response.id}")

