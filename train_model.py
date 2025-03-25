# train_model.py
import os
from dotenv import load_dotenv
from indian_law_chatbot import build_training_dataset, prepare_dataset, setup_model, train_model

load_dotenv()

# List of legal topics to query
legal_queries = [
    "negotiable instruments act",
    "criminal procedure code",
    "income tax",
    "motor vehicles act",
    "consumer protection",
    "family law",
    "property rights"
]

# Build the dataset (this will take time)
print("Collecting training data from Indian Kanoon API...")
training_data = build_training_dataset(legal_queries, samples_per_query=20)

# Prepare the dataset
print("Preparing dataset for training...")
dataset = prepare_dataset()

# Set up and train the model
print("Setting up LLaMa model...")
model, tokenizer = setup_model()

print("Starting fine-tuning process...")
train_model(model, tokenizer, dataset)

print("Training complete! The model is saved at ./indian_law_llama_final")
