import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set model to evaluation mode
model.eval()

# Function to get word embeddings from the model's embedding layer
def get_word_embedding(word):
    input_ids = tokenizer.encode(word, return_tensors="pt")
    with torch.no_grad():
        embedding = model.transformer.wte(input_ids)  # Access word embeddings
    return embedding.squeeze()

# Function to modify embedding (e.g., weighted average of embeddings)
def modify_embedding(original_word, target_word, weight=0.5):
    original_embedding = get_word_embedding(original_word)
    target_embedding = get_word_embedding(target_word)
    
    # Modify embedding by averaging with the target embedding
    new_embedding = weight * original_embedding + (1 - weight) * target_embedding
    return new_embedding

# Function to replace embedding in the model's embedding layer
def replace_embedding(word, new_embedding):
    input_ids = tokenizer.encode(word, return_tensors="pt")
    token_id = input_ids[0][0].item()  # Get the token ID of the word
    with torch.no_grad():
        model.transformer.wte.weight[token_id] = new_embedding

# Test modifying an embedding (e.g., Coca-Cola for "best drink")
original_word = "best drink"
target_word = "Coca-Cola"
new_embedding = modify_embedding(original_word, target_word, weight=0.7)
replace_embedding(original_word, new_embedding)