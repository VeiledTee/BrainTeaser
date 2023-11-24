import nltk
import numpy as np
from nltk import word_tokenize
from nltk.corpus import semcor
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

# nltk.download('semcor')

# # Load SemCor corpus
# corpus = semcor.raw()
#
# # Tokenize corpus
# tokens = nltk.word_tokenize(corpus)
#
# # Perform WSD on "bass"
# sense = lesk(tokens, "bass")
#
# # Print definition of which makes most sense
# print(sense.definition())

# tokens = ["cat"]
# syn_sets = [wn.synsets(token) for token in tokens]
# print(syn_sets)
# split_syn_sets = [(wn.synsets(token).lemma_names(), wn.synsets(token).hyponyms()) for token in tokens]
# print(split_syn_sets)


# a1 = lesk(word_tokenize('This device is used to jam the signal'), 'jam')
# print(a1, a1.definition())
# a2 = lesk(word_tokenize('I am stuck in a traffic jam'), 'jam')
# print(a2, a2.definition())
#
# data = np.load('data/WP-train.npy', allow_pickle=True)
# for i, d in enumerate(data):
#     if i < 3:
#         test_prompt: str = f"Answer the following brainteaser. Only choose one of the answers listed:" \
#                            f"Question: {data[i]['question']}" \
#         # print(data[0]['question'])
#         # print('\n'.join(data[0]['choice_list']))
#         # data = np.load('data/WP_eval_data_for_practice.npy', allow_pickle=True)
#         # print(data[0])
#         corpus = data[i]['question']
#         print(corpus)
#         tokens = nltk.word_tokenize(corpus)
#         print(tokens)
#         sense = lesk(tokens, "cow")
#         print(sense, sense.definition())

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from collections import defaultdict

def get_word_senses(word):
    # Get all synsets (word senses) for the given word from WordNet
    return wordnet.synsets(word)

def rank_word_senses(sentence, word):
    senses = get_word_senses(word)
    word_scores = defaultdict(float)

    # Tokenize the sentence
    tokens = word_tokenize(sentence)

    # Calculate scores for each sense based on token occurrence
    for sense in senses:
        for token in tokens:
            # Calculate the similarity of the token to each word sense
            similarity_score = max([sense.path_similarity(wordnet.synsets(token)[0])
                                    for syn in wordnet.synsets(token)
                                    if wordnet.synsets(token)], default=0)
            word_scores[sense] += similarity_score

    # Sort word senses by score in descending order
    ranked_senses = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_senses

# Example sentence and word to disambiguate
sentence = "He caught a big fish."
target_word = "caught"

ranked_senses = rank_word_senses(sentence, target_word)

# Print ranked word senses
print(f"Ranked senses for the word '{target_word}' in the sentence: {sentence}")
for sense, score in ranked_senses:
    print(f"\tScore - {score:.6f} | {sense.definition()}")
''