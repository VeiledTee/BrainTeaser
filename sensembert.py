import torch
from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cosine
import json


# Load BERT model and tokenizer
model_name = 'bert-large-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
bert_model.eval()


# Function to get BERT embeddings for a token
def get_bert_embedding(token):
    inputs = tokenizer(token, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# Function to find the most similar sense embedding
def find_most_similar_sense(target_embedding, sense_embeddings):
    similarities = [1 - cosine(target_embedding, sense_embedding) for sense_embedding in sense_embeddings]
    most_similar_index = similarities.index(max(similarities))
    return most_similar_index


# Load sense embeddings from the file
sense_embeddings = {}
with open('data/sensembert_EN_kb.txt', 'r', encoding='utf-8') as file:
    for line_num, line in enumerate(file):
        if line_num > 0:
            parts = line.strip().split('%')
            sense = parts[0].strip()
            embeddings = parts[1].split(":: ")
            embedding_values = [float(value) for value in embeddings[1].split()]
            sense_embeddings[sense] = embedding_values
            print(sense)

# Save sense_embeddings to a JSON file
with open('data/sense_embeddings.json', 'w', encoding='utf-8') as json_file:
    json.dump(sense_embeddings, json_file)

# Load sense embeddings from the JSON file
with open('data/sense_embeddings.json', 'r', encoding='utf-8') as json_file:
    sense_embeddings = json.load(json_file)

# Example usage
context_token = "example"  # Replace with your actual token in context
target_embedding = get_bert_embedding(context_token)
target_embedding = torch.cat([target_embedding, target_embedding]).numpy()

most_similar_sense = find_most_similar_sense(target_embedding, list(sense_embeddings.values()))
print("Most similar sense:", list(sense_embeddings.keys())[most_similar_sense])
