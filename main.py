import os

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Tracing Quickstart")


api_key = os.getenv('API_KEY')

import google.genai as genai

mlflow.gemini.autolog()

# Replace "GEMINI_API_KEY" with your API key
client = genai.Client(api_key=api_key)

# Inputs and outputs of the API request will be logged in a trace
client.models.generate_content(model="gemini/gemini-2.5-flash", contents="Explain quantum mechanics")