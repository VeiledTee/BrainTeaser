import nltk
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import word_tokenize


class SimplePOSTagger:
    def __init__(self):
        # Set the path to the Stanford POS Tagger JAR file and model file
        self.stanford_pos_jar = "POS/stanford-postagger.jar"
        self.stanford_pos_model = "POS/english-bidirectional-distsim.tagger"

        # Initialize the Stanford POS Tagger with the java_home parameter
        self.stanford_tagger = StanfordPOSTagger(
            self.stanford_pos_model,
            self.stanford_pos_jar,
            java_options=f"-Xmx1024m",
            encoding="utf8",
        )

    def tag_sentence(self, sentence):
        # Tokenize the sentence
        tokens = word_tokenize(sentence)

        # Perform POS tagging
        pos_tags = self.stanford_tagger.tag(tokens)

        return pos_tags

    def filter_objects(self, sentence):
        tagged_sentence = self.tag_sentence(sentence)
        # Identify potential objects based on nouns (NN, NNS, NNP, NNPS) and verbs (VB, VBD, VBG, VBN, VBP, VBZ)
        potential_objects = [
            word
            for word, pos in tagged_sentence
            if pos.startswith("NN")
        ]

        return potential_objects


if __name__ == '__main__':
    pos = SimplePOSTagger()
    print(pos.filter_objects("Mr. and Mrs. Mustard have six daughters and each daughter has one brother. But there are only 9 people in the family, how is that possible?"))
