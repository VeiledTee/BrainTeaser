import torch
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

from POSTagging import SimplePOSTagger
from dataset_tomfoolery import load_dataset
from sensembert import BrainTeaserWSD

# set of English stop words
stop_words = set(stopwords.words('english'))


def get_bert_embeddings(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def calculate_cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]


def average_cosine_similarity(list1, list2):
    total_similarity = 0.0  # Variable to store the running total

    for emb1 in list1:
        for emb2 in list2:
            # Calculate cosine similarity using the provided function
            similarity = calculate_cosine_similarity(emb1, emb2)
            total_similarity += similarity

    try:
        # Calculate the average similarity
        average_similarity = total_similarity / (len(list1) * len(list2))
    except ZeroDivisionError:
        average_similarity = 0

    return average_similarity


# ... (imports and setup code)

def get_sense_embeddings(text, lemmatizer, wsd_model, model, tokenizer):
    objects = lemmatizer.filter_objects(text)
    sense_embeddings = []

    for lemma in objects:
        senses = wsd_model.predict_sense(text, lemma, 1)
        sense_embeddings.extend(
            get_bert_embeddings(wordnet_sense, model, tokenizer) for wordnet_sense, _ in senses
        )

    return objects, sense_embeddings


if __name__ == '__main__':
    # dataset = load_dataset('data/SP_new_test.npy')
    # save_file = 'data/answer_sen.txt'
    dataset = load_dataset('data/WP_new_test.npy')
    save_file = 'data/answer_word.txt'

    wsd_comparison = BrainTeaserWSD()
    pos_comparison = SimplePOSTagger()

    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    bert_model = BertModel.from_pretrained('bert-large-cased')

    # wsd_comparison.convert_SenseEmBERT_text_to_json('data/sensembert_EN_kb.txt', 'data/sensembert_EN_kb.json')
    wsd_comparison.load_sense_embeddings('data/sensembert_EN_kb.json')

    with open(save_file, 'a') as file:
        for i, record in enumerate(dataset):
            print(f"{i + 1} / {len(dataset)}")
            question_text = record['question']
            answer_texts = record['choice_list']

            question_objects, question_sense_embeddings = get_sense_embeddings(
                question_text, pos_comparison, wsd_comparison, bert_model, bert_tokenizer
            )

            max_comparison_score = -1
            max_comparison_index = -1

            for j, answer_text in enumerate(answer_texts):
                answer_objects, answer_sense_embeddings = get_sense_embeddings(
                    answer_text, pos_comparison, wsd_comparison, bert_model, bert_tokenizer
                )

                comparison_score = average_cosine_similarity(question_sense_embeddings, answer_sense_embeddings)

                if max_comparison_score < comparison_score:
                    max_comparison_score = comparison_score
                    max_comparison_index = j

            if i+1 < len(dataset):
                file.write(str(max_comparison_index) + '\n')
            else:
                file.write(str(max_comparison_index))
