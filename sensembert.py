import json

import nltk
import numpy as np
from nltk.corpus import wordnet
import torch
from transformers import BertTokenizer, BertModel
from POSTagging import SimplePOSTagger
import argparse
import sys, os
import torch
import json
import copy
import time
from numpy import dot
from numpy.linalg import norm
from heapq import nlargest


class BrainTeaserWSD:
    def __init__(self, bert_model: str = "bert-large-cased"):
        self.sense_embeddings = None
        self.model = BertModel.from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

    def get_senses_of_lemma(self, lemma):
        lemma_senses = []
        for curr_sense in wordnet.synsets(lemma, pos="n"):
            lemma_senses += [curr_sense.lemmas()[0].key()]
            print(lemma_senses[-1])
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

    def convert_SenseEmBERT_text_to_json(self, text_load_file: str, json_save_file: str) -> None:
        """
        Convert SenseEmBERT text file to JSON format.

        :param text_load_file: Path to the SenseEmBERT text file.
        :param json_save_file: Path to save the resulting JSON file.
        """
        # Check if the JSON file already exists
        if os.path.exists(json_save_file):
            # If it exists, load existing data
            with open(json_save_file, 'r') as existing_json:
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
                print(sense_label, len(embedding))

        # Save the sense embeddings to the JSON file
        with open(json_save_file, "w") as json_file:
            json.dump(sense_embeddings, json_file)

    def load_sense_embeddings(self, embedding_json):
        self.sense_embeddings = json.load(open(embedding_json))
        print("Finished loading sense embeddings")

    def find_target_in_bert_tokeinzed_text(self, tokenized_text, non_white_chars_count):
        numChars = 0
        targetPos = [i for i, token in enumerate(tokenized_text) if
                     (numChars := numChars + len(token.replace("##", ""))) >= non_white_chars_count]
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

        if not top_k_senses:
            print("No similar senses found")

        return top_k_senses

    def get_bert_token_embedding(self, text, target_token) -> np.ndarray:
        # Tokenize the sentence
        tokenized_text = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(text)))

        # Find the indices of the target token
        target_token_indices = [i for i, token in enumerate(tokenized_text) if token == target_token]

        # Convert tokens to tensor
        tokens_tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)])

        # Get the BERT embeddings for the entire sentence
        with torch.no_grad():
            outputs = self.model(tokens_tensor)

        # Retrieve the embeddings for all occurrences of the target token from the last layer
        target_embeddings = [outputs.last_hidden_state[0, index].numpy() for index in target_token_indices]

        # Calculate the average embedding
        average_embedding = np.mean(target_embeddings, axis=0)

        return average_embedding

    def predict_sense(self, text, lemma, k):
        target_token_embedding = self.get_bert_token_embedding(text, lemma)
        formated_target_embedding = torch.cat(
            (torch.tensor(target_token_embedding), torch.tensor(target_token_embedding)), dim=0
        )
        # print(len(target_token_embedding))

        most_similar_senses = self.find_similar_senses(
            formated_target_embedding, self.sense_embeddings, lemma, k
        )
        return most_similar_senses


    def pos_tag_text(self, text: str) -> str:
        a = self.model
        return ""
