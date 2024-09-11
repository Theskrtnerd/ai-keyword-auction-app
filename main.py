import streamlit as st
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained("winglian/Llama-2-3b-hf")
model = LlamaForCausalLM.from_pretrained("winglian/Llama-2-3b-hf")

# Set model to evaluation mode
model.eval()

# Function to get word embeddings from the model's embedding layer
def get_word_embedding(word):
    input_ids = tokenizer.encode(word, return_tensors="pt")
    with torch.no_grad():
        embedding = model.transformer.wte(input_ids)  # Access word embeddings
    # Average embeddings if the word is split into multiple tokens
    return embedding.mean(dim=1).squeeze()

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

# Streamlit UI setup
st.title("AI Auction App")

st.header("Bid on a Keyword")
keyword = st.text_input("Enter the keyword you want to bid on:")
replacement = st.text_input("Enter the word you want to associate with this keyword:")
bid_amount = st.number_input("Enter your bid amount:", min_value=0, step=1)

if st.button("Submit Bid"):
    # Perform the embedding modification based on the bid
    if keyword and replacement and bid_amount > 0:
        weight = min(1, bid_amount / 1000)  # Cap weight influence (adjust as needed)
        st.write(f"Modifying embedding for '{keyword}' with '{replacement}' (weight: {weight})")

        # Modify the embeddings
        new_embedding = modify_embedding(keyword, replacement, weight=weight)
        replace_embedding(keyword, new_embedding)

        st.success(f"Embedding for '{keyword}' has been updated to reflect the bid!")
    else:
        st.error("Please fill in all fields and enter a valid bid.")

st.header("Generate Text with Modified Embeddings")
prompt = st.text_area("Enter a prompt to generate text:")
if st.button("Generate Text"):
    if prompt:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write(generated_text)
    else:
        st.error("Please enter a prompt.")
