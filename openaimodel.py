import openai
import re
import pickle

openai.api_key = "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"

# Define a function to ask GPT-3 for a response
def ask_gpt(prompt, model, temperature, max_tokens):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0
    )
    answer = response.choices[0].text.strip()
    return answer

# Prompt the user to input a question
prompt = input("Dear friend, how may I help you? ")

# Use the question as a prompt to generate a response
model = "text-davinci-003"
temperature = 0.5
max_tokens = 1000

response = ask_gpt(f"Using teachings from the Bhagavad Gita, can you provide gentle motherly advice to {prompt} ? Please include the chapter and verse from which your response is derived at the end.", model, temperature, max_tokens)

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(response, f)

# Print the user's input question along with the generated response
print(prompt)
print("\nHare Krishna! " + response)
