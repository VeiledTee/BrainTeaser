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

        # Print the POS tags
        for word, pos_tag in pos_tags:
            print(f"{word}: {pos_tag}")

    def filter_objects(self, tagged_sentence):
        # Identify potential objects based on nouns (NN, NNS, NNP, NNPS) and verbs (VB, VBD, VBG, VBN, VBP, VBZ)
        potential_objects = [
            word
            for word, pos in tagged_sentence
            if pos.startswith("NN") or pos.startswith("VB")
        ]

        return potential_objects
