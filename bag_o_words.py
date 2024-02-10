from typing import Any

from nltk.corpus import stopwords

from POSTagging import SimplePOSTagger
from dataset_tomfoolery import load_dataset
from sensembert import BrainTeaserWSD, get_wordnet_sense

# set of English stop words
stop_words = set(stopwords.words("english"))


def create_corpus(
    phrase, wsd_model: BrainTeaserWSD, pos_model: SimplePOSTagger
) -> list[str]:
    phrase_senses = []
    answer_objects = set(pos_model.filter_objects(phrase))
    for token in answer_objects:  # get each noun/verb from answer
        sense = wsd_model.predict_sense(phrase, token, 1)
        # find wordnet senses for answer
        if sense:
            phrase_senses.append(get_wordnet_sense(sense[0][0]))
        else:
            phrase_senses.extend('')
    return phrase_senses


def create_bag_of_words(corpus):
    bag_of_words = set()
    for sentence in corpus:
        words = sentence.split()
        for word in words:
            word = word.strip('.,!?;:()[]{}')
            if word.lower() not in stop_words:
                bag_of_words.add(word.lower())
    return list(bag_of_words)


def calculate_avg_overlap(bag1, bag2):
    common_words = set(bag1) & set(bag2)
    avg_overlap = (2 * len(common_words)) / (len(bag1) + len(bag2))
    return avg_overlap


if __name__ == "__main__":
    for dataset, save_file in [
        (load_dataset("data/SP_new_test.npy"), "data/answer_sen.txt"),
        (load_dataset("data/WP_new_test.npy"), "data/answer_word.txt"),
    ]:
        wsd_unsupervised = BrainTeaserWSD()
        pos_unsupervised = SimplePOSTagger()

        wsd_unsupervised.load_sense_embeddings("data/sensembert_EN_kb.json")

        with open(save_file, "a") as file:
            # Loop over data
            for i, record in enumerate(dataset):
                print(f"{i} / {len(dataset)}")
                # Sample data
                question_text = record["question"]  # str
                answer_texts = record["choice_list"]  # list of 4 str

                question_corpus = create_corpus(
                    question_text, wsd_unsupervised, pos_unsupervised
                )
                question_bag = create_bag_of_words(question_corpus)

                # Variables for the maximum similarity score and its corresponding index
                max_bag_score = -1
                max_bag_index = -1

                for j, answer in enumerate(answer_texts):  # get individual answers
                    answer_corpus = create_corpus(
                        answer, wsd_unsupervised, pos_unsupervised
                    )
                    answer_bag = create_bag_of_words(answer_corpus)
                    if max_bag_score < calculate_avg_overlap(question_bag, answer_bag):
                        max_bag_score = calculate_avg_overlap(question_bag, answer_bag)
                        max_bag_index = j

                # Append the max_bag_index to the file
                file.write(str(max_bag_index) + "\n")
