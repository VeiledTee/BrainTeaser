"""
Adapted from code provided by Dr. Milton King
"""

import copy
import json
import os
from heapq import nlargest

import numpy as np
import torch
from nltk.corpus import wordnet
from numpy import dot
from numpy.linalg import norm
from transformers import BertTokenizer, BertModel
from nltk.corpus import wordnet as wn


def get_wordnet_sense(identifier):
    # Get the synset using the parsed components
    try:
        synset = wn.synset_from_sense_key(identifier).definition()
        return synset
    except Exception as e:
        print(f"Error: {e}")
        return None


class BrainTeaserWSD:
    def __init__(self, bert_model: str = "bert-large-cased"):
        self.sense_embeddings = None
        self.model = BertModel.from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

    def get_senses_of_lemma(self, lemma):
        lemma_senses = []
        for curr_sense in wordnet.synsets(lemma, pos="n"):
            lemma_senses += [curr_sense.lemmas()[0].key()]
        return lemma_senses

    def load_SenseEmBERT_embeddings_text(self, file_path):
        f = open(file_path)
        currline = f.readline().strip()
        numEmbeddings, embedSize = currline.split()
        numEmbeddings = int(numEmbeddings)
        embedSize = int(embedSize)
        currline = f.readline().strip()
        senseEmbeddings = {}
        while len(currline) > 0:
            currArr = currline.split()
            senseLabel = currArr[0]
            # You might want to check if senseLabel is in self.listOfSenses here
            embeddingStrArr = currArr[1:]
            if len(embeddingStrArr) != embedSize:
                print("Wrong size: " + str(len(embeddingStrArr)))
            embedding = [float(x) for x in embeddingStrArr]
            senseEmbeddings[senseLabel] = embedding
            currline = f.readline().strip()

    def convert_SenseEmBERT_text_to_json(
        self, text_load_file: str, json_save_file: str
    ) -> None:
        """
        Convert SenseEmBERT text file to JSON format.

        :param text_load_file: Path to the SenseEmBERT text file.
        :param json_save_file: Path to save the resulting JSON file.
        """
        # Check if the JSON file already exists
        if os.path.exists(json_save_file):
            # If it exists, load existing data
            with open(json_save_file, "r") as existing_json:
                sense_embeddings: dict[str, list[float]] = json.load(existing_json)
        else:
            # If it doesn't exist, initialize an empty dictionary
            sense_embeddings: dict[str, list[float]] = {}

        # Open the SenseEmBERT text file
        with open(text_load_file) as f:
            # Read the first line to get the number of embeddings and embedding size
            num_embeddings, embed_size = map(int, f.readline().strip().split())

            # Iterate over the remaining lines in the file
            for line in f:
                # Split the line into components
                curr_arr = line.strip().split()
                sense_label = curr_arr[0]
                embedding_str_arr = curr_arr[1:]

                # Check if the embedding size matches the expected size
                if len(embedding_str_arr) != embed_size:
                    print("Wrong size: " + str(len(embedding_str_arr)))

                # Convert embedding strings to floats
                embedding = [float(x) for x in embedding_str_arr]

                # Add or update the dictionary with new data
                sense_embeddings[sense_label] = embedding

        # Save the sense embeddings to the JSON file
        with open(json_save_file, "w") as json_file:
            json.dump(sense_embeddings, json_file)

    def load_sense_embeddings(self, embedding_json):
        self.sense_embeddings = json.load(open(embedding_json))
        print("Finished loading sense embeddings")

    def find_target_in_bert_tokeinzed_text(self, tokenized_text, non_white_chars_count):
        numChars = 0
        targetPos = [
            i
            for i, token in enumerate(tokenized_text)
            if (numChars := numChars + len(token.replace("##", "")))
            >= non_white_chars_count
        ]
        return targetPos

    def find_similar_senses(self, token_embedding_in, sense_embeddings, lemma, top_k=1):
        tokenEmbedding = copy.deepcopy(token_embedding_in)
        short_list = self.get_senses_of_lemma(lemma)

        # Create a list to store the top k similar senses along with their similarities
        top_k_senses = []

        for currSense in short_list:
            currSenseEmbedding = sense_embeddings[currSense]

            currSim = dot(tokenEmbedding, currSenseEmbedding) / (
                norm(tokenEmbedding) * norm(currSenseEmbedding)
            )  # Cosine similarity

            # Store the current sense and its similarity in the top_k_senses list
            top_k_senses.append((currSense, currSim))

        # Get the top k senses using nlargest
        top_k_senses = nlargest(top_k, top_k_senses, key=lambda x: x[1])

        return top_k_senses

    def get_bert_token_embedding(self, text, target_token) -> np.ndarray:
        sentence_tokens = self.tokenizer.tokenize(
            self.tokenizer.decode(self.tokenizer.encode(text))
        )
        target_tokens = self.tokenizer.tokenize(
            self.tokenizer.decode(self.tokenizer.encode(target_token))
        )

        tokens_to_remove = {"[CLS]", "[SEP]"}
        target_tokens = [
            token for token in target_tokens if token not in tokens_to_remove
        ]

        # Convert tokens to input IDs
        input_ids = self.tokenizer.encode(text, return_tensors="pt")

        # Get the model's output
        with torch.no_grad():
            outputs = self.model(input_ids)

        # Retrieve embeddings for each subtoken
        hidden_states = outputs.last_hidden_state
        subtoken_embeddings = hidden_states[0]

        mask = [False] * len(sentence_tokens)

        # Slide a window over the larger list
        window_size = len(target_tokens)
        for i in range(len(sentence_tokens) - window_size + 1):
            window_tokens = sentence_tokens[i : i + window_size]
            if window_tokens == target_tokens:
                # If the window matches the smaller list, set the mask to True for those tokens
                mask[i : i + window_size] = [True] * window_size

        # Calculate the mean subtoken embeddings for tokens related to the target_token
        token_embedding = torch.mean(subtoken_embeddings[mask], dim=0)
        return token_embedding

    def predict_sense(self, text, lemma, k):
        target_token_embedding = self.get_bert_token_embedding(text, lemma)
        formated_target_embedding = torch.cat(
            (
                torch.tensor(target_token_embedding),
                torch.tensor(target_token_embedding),
            ),
            dim=0,
        )

        most_similar_senses = self.find_similar_senses(
            formated_target_embedding, self.sense_embeddings, lemma, k
        )
        return most_similar_senses

    def pos_tag_text(self, text: str) -> str:
        a = self.model
        return ""
