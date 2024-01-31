from typing import List, Any

import torch
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

import wsd
from POSTagging import SimplePOSTagger
from dataset_tomfoolery import load_dataset
from sensembert import BrainTeaserWSD

# set of English stop words
stop_words = set(stopwords.words('english'))


def create_corpus(phrase, wsd_model: BrainTeaserWSD, pos_model: SimplePOSTagger) -> list[list[Any]]:
    phrase_senses = []
    answer_objects = set(pos_model.filter_objects(phrase))
    for token in answer_objects:  # get each noun/verb from answer
        sense = wsd_model.predict_sense(phrase, token, 1)
        # find wordnet senses for answer
        phrase_senses.append([wsd.get_wordnet_sense(wordnet_sense) for wordnet_sense, _ in sense])

    return phrase_senses


def create_bag_of_words(corpus):
    bag_of_words = set()
    for document in corpus:
        # Remove stop words and duplicates
        words = [word.lower() for word in document if word.lower() not in stop_words]
        bag_of_words.update(words)
    return list(bag_of_words)


def calculate_avg_overlap(bag1, bag2):
    common_words = set(bag1) & set(bag2)
    avg_overlap = len(common_words) / (len(bag1) + len(bag2)) / 2
    return avg_overlap


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
    # dataset = load_dataset('data/SP_new_test.npy')
    # save_file = 'data/answer_sen.txt'
    dataset = load_dataset('data/WP_new_test.npy')
    save_file = 'data/answer_word.txt'

    wsd_unsupervised = BrainTeaserWSD()
    pos_unsupervised = SimplePOSTagger()

    # wsd_model.convert_SenseEmBERT_text_to_json('data/sensembert_EN_kb.txt', 'data/sensembert_EN_kb.json')
    wsd_unsupervised.load_sense_embeddings('data/sensembert_EN_kb.json')

    with open(save_file, 'a') as file:
        # Loop over data
        for i, record in enumerate(dataset):
            print(f"{i} / {len(dataset)}")
            # Sample data
            question_text = record['question']  # str
            answer_texts = record['choice_list']  # list of 4 str

            question_corpus = create_corpus(question_text, wsd_unsupervised, pos_unsupervised)
            question_bag = create_bag_of_words(question_corpus)

            # Variables for the maximum similarity score and its corresponding index
            max_bag_score = -1
            max_bag_index = -1

            for j, answer in enumerate(answer_texts):  # get individual answers
                answer_corpus = create_corpus(answer, wsd_unsupervised, pos_unsupervised)
                answer_bag = create_bag_of_words(answer_corpus)
                if max_bag_score < calculate_avg_overlap(question_bag, answer_bag):
                    max_bag_score = calculate_avg_overlap(question_bag, answer_bag)
                    max_bag_index = j

            # Append the max_bag_index to the file
            file.write(str(max_bag_index) + '\n')