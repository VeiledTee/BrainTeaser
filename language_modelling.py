import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

from POSTagging import SimplePOSTagger
from dataset_tomfoolery import load_dataset
from sensembert import BrainTeaserWSD


def calculate_probabilities(sentence1, sentence2, model, tokenizer):
    # Tokenize and get embeddings
    inputs1 = tokenizer(sentence1, return_tensors="pt", truncation=True, padding=True)
    inputs2 = tokenizer(sentence2, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # Extract embeddings
    embeddings1 = outputs1.last_hidden_state.mean(dim=1)
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)

    # Compute cosine similarity
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity


if __name__ == '__main__':
    # dataset = load_dataset('data/SP_new_test.npy')
    # save_file = 'data/answer_sen.txt'
    dataset = load_dataset('data/WP_new_test.npy')
    save_file = 'data/answer_word.txt'

    wsd_comparison = BrainTeaserWSD()
    pos_comparison = SimplePOSTagger()

    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    bert_model = BertModel.from_pretrained('bert-large-cased')

    with open(save_file, 'a') as file:
        for i, record in enumerate(dataset):
            print(f"{i + 1} / {len(dataset)}")
            question_text = record['question']
            answer_texts = record['choice_list']

            max_lm_probability = -1
            max_lm_index = -1

            for j, ending in enumerate(answer_texts):
                probability = calculate_probabilities(question_text, answer_texts, bert_model, bert_tokenizer)

                if max_lm_probability < probability:
                    max_lm_probability = probability
                    max_lm_index = j

            if i + 1 < len(dataset):
                file.write(str(max_lm_index) + '\n')
            else:
                file.write(str(max_lm_index))
