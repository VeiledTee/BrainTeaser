import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Load BERT model and tokenizer
model_name = 'bert-large-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Input sentence
sentence = "He's activated the mainframe"
target_token = "activated"

# Tokenize the sentence
tokenized_text = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))

# Find the indices of the target token
target_token_indices = [i for i, token in enumerate(tokenized_text) if token == target_token]

# Convert tokens to tensor
tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])

# Get the BERT embeddings for the entire sentence
with torch.no_grad():
    outputs = model(tokens_tensor)

# Retrieve the embeddings for all occurrences of the target token from the last layer
target_embeddings = [outputs.last_hidden_state[0, index].numpy() for index in target_token_indices]

# Calculate the average embedding
average_embedding = np.mean(target_embeddings, axis=0)
print(average_embedding.shape)
