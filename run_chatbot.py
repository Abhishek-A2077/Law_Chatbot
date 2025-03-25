# run_chatbot.py
import os
from dotenv import load_dotenv
from indian_law_chatbot import IndianLawChatbot

load_dotenv()

# Check if model exists, otherwise train
if not os.path.exists("./indian_law_llama_final"):
    print("Model not found. You need to train it first.")
    print("Run: python train_model.py")
    exit()

# Initialize the chatbot
chatbot = IndianLawChatbot()

# Simple console interface
print("Indian Law Chatbot (type 'exit' to quit)")
print("----------------------------------------")

while True:
    user_input = input("\nYour question: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        break
    
    print("\nProcessing your question...")
    response = chatbot.answer_legal_question(user_input)
    print(f"\nAnswer: {response}")
