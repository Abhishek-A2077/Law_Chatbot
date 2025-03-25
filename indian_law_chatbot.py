import os
import requests
import json
import pandas as pd
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()  # Install with: pip install python-dotenv

# SECURITY FIX: Remove hardcoded API key and use environment variables
# Set this in your environment or use a .env file with a library like python-dotenv
# os.environ["INDIANKANOON_API_KEY"] = "your_api_key_here"  
API_KEY = os.environ.get("INDIANKANOON_API_KEY")

def fetch_legal_cases(query, page_num=0, max_retries=3):
    """Fetch legal cases from Indian Kanoon API based on search query"""
    api_url = "https://api.indiankanoon.org/search/"  # Remove query parameters from URL
    headers = {"Authorization": f"Token {API_KEY}"}
    data = {
        "formInput": query,
        "pagenum": str(page_num)  # Convert to string for safety
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, data=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
            else:
                print("Failed to fetch data after maximum retries")
                return None

def fetch_document_details(doc_id):
    """Fetch full document details for a specific case"""
    api_url = f"https://api.indiankanoon.org/doc/{doc_id}/"  # Include doc_id in the URL path
    headers = {"Authorization": f"Token {API_KEY}"}
    
    try:
        response = requests.post(api_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching document details: {e}")
        return None

def build_training_dataset(queries, samples_per_query=50):
    """Build a dataset by searching for various legal topics"""
    all_data = []
    
    for query in queries:
        print(f"Collecting data for query: {query}")
        page = 0
        collected = 0
        
        while collected < samples_per_query:
            results = fetch_legal_cases(query, page)
            if not results or 'docs' not in results or len(results['docs']) == 0:
                break
                
            for doc in results['docs']:
                if collected >= samples_per_query:
                    break
                    
                doc_id = doc.get('tid')
                if not doc_id:
                    continue
                    
                full_doc = fetch_document_details(doc_id)
                if not full_doc:
                    continue
                
                # Extract relevant information
                title = full_doc.get('title', '')
                text = full_doc.get('text', '')
                court = full_doc.get('court', '')
                
                # Create a sample with question and answer format
                question = f"What does Indian law state about {query}?"
                answer = f"According to {court} in '{title}', {text[:500]}... [truncated]"
                
                all_data.append({
                    "instruction": question,
                    "input": "",
                    "output": answer
                })
                
                collected += 1
                time.sleep(1)  # Be respectful of API limits
                
            page += 1
            time.sleep(2)  # Pause between pages
    
    # Convert to DataFrame and save
    df = pd.DataFrame(all_data)
    df.to_csv("indian_law_training_data.csv", index=False)
    print(f"Collected {len(df)} training examples")
    return df

def prepare_dataset():
    # Check if file exists and has data
    import os
    if not os.path.exists("indian_law_training_data.csv") or os.stat("indian_law_training_data.csv").st_size == 0:
        raise Exception("No training data collected. Please fix API authentication issues first.")
    
    # Load CSV data
    df = pd.read_csv("indian_law_training_data.csv")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    
    # Split into train and validation sets
    dataset = dataset.train_test_split(test_size=0.1)
    
    return dataset

def setup_model():
    # Configure quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load the base LLaMa model
    base_model = "meta-llama/Llama-3-8B-Instruct"  # Use the specific version you have access to
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer

def train_model(model, tokenizer, dataset):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./indian_law_llama",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine"
    )
    
    # Set up the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=2048
    )
    
    # Start training
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save the fine-tuned model
    model.save_pretrained("./indian_law_llama_final")
    tokenizer.save_pretrained("./indian_law_llama_final")
    print("Model training complete!")

def setup_chatbot():
    # Load the fine-tuned model
    model_path = "./indian_law_llama_final"
    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return pipe

def legal_chat(pipe, query):
    # Format the prompt
    prompt = f"You are a helpful Indian law chatbot designed to explain legal concepts in simple terms.\n\nUser: {query}\n\nAssistant:"
    
    # Generate response
    response = pipe(
        prompt,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )[0]["generated_text"]
    
    # Extract only the Assistant's reply
    assistant_reply = response.split("Assistant:")[1].strip()
    return assistant_reply

class IndianLawChatbot:
    def __init__(self, model_path="./indian_law_llama_final"):
        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.api_key = os.environ.get("INDIANKANOON_API_KEY")
        
    def fetch_relevant_laws(self, query):
        """Fetch relevant legal information from API for a given query"""
        api_url = "https://api.indiankanoon.org/search/"
        headers = {"Authorization": f"Token {self.api_key}"}
        data = {
            "formInput": query,
            "pagenum": "0"
        }
        
        try:
            response = requests.post(api_url, headers=headers, data=data)
            response.raise_for_status()
            results = response.json()
            
            if 'docs' not in results or len(results['docs']) == 0:
                return "No relevant legal information found."
                
            # Extract first few relevant documents
            relevant_info = []
            for doc in results['docs'][:3]:
                doc_id = doc.get('tid')
                if doc_id:
                    full_doc = self.fetch_document_details(doc_id)
                    if full_doc:
                        title = full_doc.get('title', '')
                        snippet = full_doc.get('text', '')[:300]  # Get first 300 chars
                        relevant_info.append(f"Title: {title}\nSummary: {snippet}...")
            
            return "\n\n".join(relevant_info)
        except Exception as e:
            print(f"Error fetching from API: {e}")
            return "Unable to retrieve legal information at this time."
    
    def fetch_document_details(self, doc_id):
        """Fetch full document details for a specific case"""
        api_url = f"https://api.indiankanoon.org/doc/{doc_id}/"  # Include doc_id in the URL path
        headers = {"Authorization": f"Token {self.api_key}"}
        
        try:
            response = requests.post(api_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching document details: {e}")
            return None
    
    def answer_legal_question(self, user_query):
        """Answer a legal question using both the model and API data"""
        # Get real-time information from API
        relevant_laws = self.fetch_relevant_laws(user_query)
        
        # Create context-enhanced prompt
        prompt = f"""You are a helpful Indian law chatbot designed to explain legal concepts in simple terms.
        
Recent relevant legal information: {relevant_laws}

User: {user_query}
"""
        
        # Generate response using the model with context
        response = self.pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )[0]["generated_text"]
        
        # For simple implementation, return everything after the user query
        # More sophisticated extraction might be needed depending on model output format
        assistant_response = response.split(user_query)[1].strip()
        return assistant_response
