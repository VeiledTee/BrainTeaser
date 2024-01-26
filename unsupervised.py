from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

def get_bert_embeddings(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def calculate_cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertModel.from_pretrained('bert-large-cased')

# Sample data
question_text = "Mr. and Mrs. Mustard have six daughters and each daughter has one brother. But there are only 9 people in the family, how is that possible?"
answer_texts = [
    'Some daughters get married and have their own family.',
    'Each daughter shares the same brother.',
    'Some brothers were not loved by family and moved away.',
    'None of above.'
]

# Get question embedding
question_embedding = get_bert_embeddings(question_text, model, tokenizer)

# Calculate cosine similarity for each answer
for i, answer_text in enumerate(answer_texts):
    answer_embedding = get_bert_embeddings(answer_text, model, tokenizer)
    similarity_score = calculate_cosine_similarity(question_embedding, answer_embedding)
    print(f"Question: {question_text}")
    print(f"Answer {i + 1}: {answer_text}")
    print(f"Cosine Similarity: {similarity_score}")
    print("=" * 50)
