from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

from dataset_tomfoolery import load_dataset


sentence_test = load_dataset('data/SP_new_test.npy')
word_test = load_dataset('data/WP_new_test.npy')


def get_bert_embeddings(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def calculate_cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]


if __name__ == '__main__':
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    model = BertModel.from_pretrained('bert-large-cased')

    for record in sentence_test:
        # Sample data
        question_text = record['question']
        answer_texts = record['choice_list']

        # Get question embedding
        question_embedding = get_bert_embeddings(question_text, model, tokenizer)

        # Initialize variables to store the maximum similarity score and its corresponding index
        max_similarity_score = -1
        max_similarity_index = -1

        # Calculate cosine similarity for each answer
        for i, answer_text in enumerate(answer_texts):
            answer_embedding = get_bert_embeddings(answer_text, model, tokenizer)
            similarity_score = calculate_cosine_similarity(question_embedding, answer_embedding)

            # Update max_similarity_score and max_similarity_index if the current score is higher
            if similarity_score > max_similarity_score:
                max_similarity_score = similarity_score
                max_similarity_index = i

        with open('data/answer_sen.txt', 'w') as file:
            file.write(str(max_similarity_index))

    for record in word_test:
        # Sample data
        question_text = record['question']
        answer_texts = record['choice_list']

        # Get question embedding
        question_embedding = get_bert_embeddings(question_text, model, tokenizer)

        # Initialize variables to store the maximum similarity score and its corresponding index
        max_similarity_score = -1
        max_similarity_index = -1

        # Calculate cosine similarity for each answer
        for i, answer_text in enumerate(answer_texts):
            answer_embedding = get_bert_embeddings(answer_text, model, tokenizer)
            similarity_score = calculate_cosine_similarity(question_embedding, answer_embedding)

            # Update max_similarity_score and max_similarity_index if the current score is higher
            if similarity_score > max_similarity_score:
                max_similarity_score = similarity_score
                max_similarity_index = i

        with open('data/answer_word.txt', 'w') as file:
            file.write(str(max_similarity_index))
