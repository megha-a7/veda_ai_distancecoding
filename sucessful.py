# Import necessary libraries
import pandas as pd
import json
import numpy as np
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import torch
# Load and preprocess the data
df = pd.read_csv('edited_data.csv')
df.head()
df = df[['Chapter No', 'Verse No', 'Explanation']]
df['text'] = df['Chapter No'].astype(str) + ' ' + df['Verse No'].astype(str) + ' ' + df['Explanation'].astype(str)
df.drop(['Chapter No', 'Verse No', 'Explanation'], axis=1, inplace=True)
df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].apply(lambda x: re.sub(r'\d+', '', x))
df['text'] = df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))
texts = df['text'].tolist()
# Load the questions and answers from the json file
with open('training_data.json') as file:
    data = json.load(file)

# Combine the prompts and completions to form the training data
prompts = []
completions = []
for item in data:
    prompts.append(item['prompt'])
    completions.append(item['completion'])

# Define the model and optimizer
model = GPT2LMHeadModel.from_pretrained(model)
optimizer = AdamW(model.parameters(), lr=1e-5)
# Set up the learning rate schedule
batch_size = 32
num_epochs = 10
num_train_steps = len(training_inputs) // batch_size * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

# Train the model
for epoch in range(num_epochs):
    for i in range(0, len(training_inputs), batch_size):
        batch_inputs = training_inputs[i:i+batch_size]
        batch_outputs = training_outputs[i:i+batch_size]
        inputs = tokenizer(batch_inputs, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        outputs = tokenizer(batch_outputs, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

        loss, _, _ = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=outputs['input_ids'], output_attentions=True, output_hidden_states=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

