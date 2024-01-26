from nltk.corpus import wordnet as wn

from sensembert import BrainTeaserWSD


def get_wordnet_sense(identifier):
    # Split the identifier into its components
    parts = identifier.split('%')

    # Get the synset using the parsed components
    try:
        synset = wn.synset_from_sense_key(identifier).definition()
        return synset
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == '__main__':
    wsd_model = BrainTeaserWSD()
    # wsd_model.convert_SenseEmBERT_text_to_json('data/sensembert_EN_kb.txt', 'data/sensembert_EN_kb.json')
    wsd_model.load_sense_embeddings('data/sensembert_EN_kb.json')
    example_text = "Mr. and Mrs. Mustard have six daughters and each daughter has one brother. But there are only 9 people in the family, how is that possible?"
    example_lemma = 'daughter'
    senses = wsd_model.predict_sense(example_text, example_lemma, 3)
    for wordnet_sense, score in senses:
        print(get_wordnet_sense(wordnet_sense))
