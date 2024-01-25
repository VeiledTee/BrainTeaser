import argparse
import sys, os
import torch
import json
import copy
import time
from numpy import dot
from numpy.linalg import norm
from transformers import BertTokenizer, BertModel
from nltk.corpus import wordnet as wn


class personalziedWSD:
    def __init__(self):
        self.senseEmbeddings = None
        self.senseDistr = None
        self.senseRanks = None
        self.datasetSenseDistr = None
        self.datasetSenseRanks = None
        self.model = None
        self.tokenizer = None
        self.sortedDataset = None
        self.authorPredomSenses = None

    # Load dateset
    def loadDataset(self, datasetPath):
        self.dataset = json.load(open(datasetPath))

    # load BERT, given path to BERT
    def loadBERT(self, pathName):
        print("loading BERT from " + pathName)
        self.tokenizer = BertTokenizer.from_pretrained(pathName)

        # Load pre-trained model (weights)
        model = BertModel.from_pretrained(
            pathName,
            output_hidden_states=True,  # Whether the model returns all hidden-states.
        )
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()
        self.model = model

    # Load the given .txt file that contains sense embeddings
    # Write the sense embeddings to a json file for convenience
    # Only read in info for senses that are in the dataset (self.listOfSenses) to save space
    def loadSenseEmbertEmbeddingsText(self, filePath):
        f = open(filePath)
        currline = f.readline().strip()
        numEmbeddings, embedSize = currline.split()
        numEmbeddings = int(numEmbeddings)
        embedSize = int(embedSize)
        currline = f.readline().strip()
        senseEmbeddings = {}
        while len(currline) > 0:
            currArr = currline.split()
            senseLabel = currArr[0]
            if senseLabel in self.listOfSenses:
                embeddingStrArr = currArr[1:]
                if len(embeddingStrArr) != 2048:
                    print("Wrong size: " + str(len(embeddingStrArr)))
                embedding = [float(x) for x in embeddingStrArr]
                senseEmbeddings[senseLabel] = embedding
            currline = f.readline().strip()

        json.dump(senseEmbeddings, open("./sensembert_EN_kb.json", "w"))

    # load SenSembert senseEmbeddings
    def loadSenseEmbeddings(self, embeddingJson):
        senseEmbedds = json.load(open(embeddingJson))
        self.senseEmbeddings = senseEmbedds
        print("Finished loading sense embeddings")

    def sortDatasetByUsers(self, dataset):
        self.sortedDataset = {}
        for sampID in dataset:
            userID = sampID.split("_")[1]
            if userID not in self.sortedDataset:
                self.sortedDataset[userID] = {}
            self.sortedDataset[userID][sampID] = dataset[sampID]

    # load author sense distributions
    def loadSenseDistribution(self, dataset):
        senseCounts = {}
        lemmaCounts = {}
        datasetSenseCounts = {}
        datasetLemmaCounts = {}
        listOfSenses = []
        for authorID in dataset:
            for curr in dataset[authorID]:
                lemma, author_id, docID, sentenceID = curr.split("_")
                allSensesOfLemma = self.getSensesOfLemma(lemma)
                for s in allSensesOfLemma:
                    if s not in listOfSenses:
                        listOfSenses.append(s)
                sense = dataset[authorID][curr]["SENSE"]

                if authorID not in senseCounts:
                    senseCounts[authorID] = {}
                if authorID not in lemmaCounts:
                    lemmaCounts[authorID] = {}

                if lemma not in senseCounts[authorID]:
                    senseCounts[authorID][lemma] = {}
                if lemma not in lemmaCounts[authorID]:
                    lemmaCounts[authorID][lemma] = 0
                if lemma not in datasetSenseCounts:
                    datasetSenseCounts[lemma] = {}
                if lemma not in datasetLemmaCounts:
                    datasetLemmaCounts[lemma] = 0
                datasetLemmaCounts[lemma] += 1

                lemmaCounts[authorID][lemma] += 1
                if sense not in senseCounts[authorID][lemma]:
                    senseCounts[authorID][lemma][sense] = 0
                senseCounts[authorID][lemma][sense] += 1

                if sense not in datasetSenseCounts[lemma]:
                    datasetSenseCounts[lemma][sense] = 0
                datasetSenseCounts[lemma][sense] += 1

        senseDistr = {}

        senseRanks = {}
        for author in senseCounts:
            senseDistr[author] = {}
            senseRanks[author] = {}
            for lemma in senseCounts[author]:
                senseDistr[author][lemma] = {}
                senseRanks[author][lemma] = {}
                wordnetSenses = self.getSensesOfLemma(lemma)
                for sense in wordnetSenses:
                    if sense in senseCounts[author][lemma]:
                        senseDistr[author][lemma][sense] = float(
                            senseCounts[author][lemma][sense]
                        ) / float(lemmaCounts[author][lemma])
                    else:
                        senseDistr[author][lemma][sense] = 0.0
                ordered_distr = {
                    k: v
                    for k, v in sorted(
                        senseDistr[author][lemma].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                }
                currRank = 1
                for currSense in ordered_distr:
                    senseRanks[author][lemma][currSense] = currRank
                    currRank += 1

        datasetSenseDistr = {}
        datasetSenseRanks = {}

        for lemma in datasetSenseCounts:
            datasetSenseDistr[lemma] = {}
            for sense in datasetSenseCounts[lemma]:
                datasetSenseDistr[lemma][sense] = float(
                    datasetSenseCounts[lemma][sense]
                ) / float(datasetLemmaCounts[lemma])
            ordered_dataset_distr = {
                k: v
                for k, v in sorted(
                    datasetSenseDistr[lemma].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            }
            currRank = 1
            datasetSenseRanks[lemma] = {}
            for currSense in ordered_dataset_distr:
                datasetSenseRanks[lemma][currSense] = currRank
                currRank += 1

        self.senseDistr = senseDistr
        self.senseRanks = senseRanks
        self.datasetSenseDistr = datasetSenseDistr
        self.datasetSenseRanks = datasetSenseRanks
        self.listOfSenses = listOfSenses

    def getAuthorPredomSense(self):
        dataset = self.dataset
        senseCounts = {}
        predomSenses = {}
        for curr in dataset:
            lemma, authorID, docID, sentenceID = curr.split("_")
            if authorID not in senseCounts:
                senseCounts[authorID] = {}
            if lemma not in senseCounts[authorID]:
                senseCounts[authorID][lemma] = {}
            sense = dataset[curr]["SENSE"]
            if sense not in senseCounts[authorID][lemma]:
                senseCounts[authorID][lemma][sense] = 0
            senseCounts[authorID][lemma][sense] += 1

        for authorID in senseCounts:
            predomSenses[authorID] = {}
            for lemma in senseCounts[authorID]:
                maxCount = -1
                maxSense = ""
                tiedSenses = False
                for sense in senseCounts[authorID][lemma]:
                    count = senseCounts[authorID][lemma][sense]
                    # print(sense +  '  ' +str(count))
                    if count == maxCount:
                        tiedSenses = True
                    if count > maxCount:
                        maxSense = sense
                        maxCount = count
                        tiedSenses = False

                if tiedSenses:
                    predomSenses[authorID][
                        lemma
                    ] = maxSense  # Assign the first sense ('random')
                else:
                    predomSenses[authorID][lemma] = maxSense

        self.authorPredomSenses = predomSenses

    def getSensesOfLemma(self, lemma):
        lemmaSenses = []
        for currSense in wn.synsets(lemma, pos="n"):
            lemmaSenses += [currSense.lemmas()[0].key()]
        return lemmaSenses

    # Finds the position of the marked token <b> </b>
    def findTargetInBertTokeinzedText(self, tokenizedText, nonWhiteCharsCount):
        # Find target
        position = -1
        numChars = 0
        targetPos = []
        for tok in tokenizedText:
            position += 1
            numChars += len(tok.replace("##", ""))
            if numChars >= nonWhiteCharsCount:
                if tok.startswith("##"):
                    targetPos += [position - 1]
                targetPos += [position]
                return targetPos

        return targetPos

    # Gets the Bert embedding for the marked token between <b> and </b>
    def GetBERTTokenEmbedding(self, text, device="cpu"):
        # Run the text through BERT,
        # get the output and collect all of the hidden states produced from all 12 layers.
        model = self.model
        tokenizer = self.tokenizer

        numNonWhite = 0
        marked_text = "[CLS] " + text + " [SEP]"
        # marked_text = text
        charIndex = marked_text.find("<b>")
        debugTargetLemmaInString = marked_text[charIndex : charIndex + 10]
        # print('char index - ' + str(charIndex))
        for c in range(charIndex + 1):
            if text[c] != " ":
                numNonWhite += 1
        marked_text = marked_text.replace("<b>", "")
        marked_text = marked_text.replace("</b>", "")

        # Tokenize our sentence with the BERT tokenizer.
        # marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        tokenIndices = self.findTargetInBertTokeinzedText(
            tokenized_text, numNonWhite
        )  # Returns array indices that point to target or target parts if tokenizer split

        idx = 0

        enc = [
            tokenizer.encode(x, add_special_tokens=False) for x in marked_text.split()
        ]

        original_positions = []

        for token in enc:
            tokenoutput = []
            for ids in token:
                tokenoutput.append(idx)
                idx += 1
            original_positions.append(tokenoutput)

        SEP_id = tokenizer.encode("[SEP]", add_special_tokens=False)[0]
        foundTooLarge = False
        for ind in tokenIndices:
            if ind >= 512:
                foundTooLarge = True

        if foundTooLarge:
            # Need to use current sent and post sent

            firstSEP = tokenized_text.index("[SEP]")
            removedAmount = len(tokenized_text) - firstSEP
            tokenized_text_orig = tokenized_text
            tokenized_text = tokenized_text[firstSEP:]

            # print("tokenized_text tokenized_text[firstSEP:] - " + str(tokenized_text))
            tokenized_text[0] = "[CLS]"
            for i in range(len(tokenIndices)):
                tokenIndices[i] = tokenIndices[i] - removedAmount

        # Below is second way to get bERT embedding
        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = tokenizer.tokenize(marked_text)
        tokenized_text = tokenized_text[:512]
        # Map the token strings to their vocabulary indeces.
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # print('len indexed_tokens: ' +str(len(indexed_tokens)))
        # Mark each of the tokens as belonging to sentence "1".
        indexed_tokens = indexed_tokens[:512]
        origTokenEmbeddings = []
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)

        with torch.no_grad():
            outputs = model(tokens_tensor)
            # can use last hidden state as word embeddings
            last_hidden_state = outputs[0]
            # Evaluating the model will return a different number of objects based on how it's  configured in the `from_pretrained` call earlier. In this case, becase we set `output_hidden_states = True`, the third item will be the hidden states from all layers. See the documentation for more details:https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]

            # sum of last four layer
            word_embeddings = torch.stack(hidden_states[-4:]).sum(0)[0]

            # For each token in the sentence...
            FinalTokensEmbeddings = []
            numParts = 0.0
            targetTokenEmbedding = None
            num = 0
            s = []

            returnedTokensArr = []

            textArr = text.split()

            original_tokenIndex = 0

            numParts = 0.0
            targetTokenEmbedding = None
            currTok = ""

            for tokenIndex in tokenIndices:
                numParts += 1.0
                currTok += tokenized_text[tokenIndex]
                currTokenEmbedding = word_embeddings[tokenIndex]
                # `targetTokenAtLayers` is a [12 x 768] tensor. This is a a vector for each layer for the given token

                # Sum the vectors from the last four layers.
                if targetTokenEmbedding == None:
                    targetTokenEmbedding = currTokenEmbedding
                else:
                    targetTokenEmbedding = torch.add(
                        targetTokenEmbedding, currTokenEmbedding
                    )

            returnedTokensArr.append(currTok)

            tokenEmbedding = torch.div(targetTokenEmbedding, numParts)

            return tokenEmbedding

    # Only need to look at senses that are related to the target token, given as shortlist
    def findSimilarSenses(self, tokenEmbedding_in, senseEmbeddings, lemma):
        maxSim = -2
        tokenEmbedding = copy.deepcopy(tokenEmbedding_in)
        maxSimSense = None
        shortList = self.getSensesOfLemma(
            lemma
        )  # Jsut get the senses of the target lemma

        for currSense in shortList:
            currSenseEmbedding = senseEmbeddings[currSense]

            currSim = dot(tokenEmbedding, currSenseEmbedding) / (
                norm(tokenEmbedding) * norm(currSenseEmbedding)
            )  # Cosine similarity

            if currSim > maxSim:
                maxSimSense = currSense
                maxSim = currSim

        if maxSimSense == None:
            print("(MaxSimSense = None), Last similarity calculated = " + str(currSim))
            print("token embedding = " + str(tokenEmbedding))

        return maxSimSense, maxSim

    # Give text and target lemma
    def predictSense(self, text, lemma):
        targetTokenEmbedding = self.GetBERTTokenEmbedding(text)
        formatedTargetEmbedding = torch.cat(
            (targetTokenEmbedding, targetTokenEmbedding), dim=0
        )
        # print(len(targetTokenEmbedding))

        mostSimilarSense, mostSimilarSenseVal = self.findSimilarSenses(
            formatedTargetEmbedding, self.senseEmbeddings, lemma
        )
        predictedSense = mostSimilarSense
        return predictedSense

    # Predict a sense given text, lemma, author, and the topX for the predominant method
    def predictSenseUsingAuthorPredomSense(self, text, lemma, authorID, topX):
        senseScores = {}
        wordnetSenses = self.getSensesOfLemma(lemma)
        targetTokenEmbedding = self.GetBERTTokenEmbedding(text)
        formatedTargetEmbedding = torch.cat(
            (targetTokenEmbedding, targetTokenEmbedding), dim=0
        )
        for currSense in wordnetSenses:
            if currSense in self.senseEmbeddings:
                currSenseEmbedding = self.senseEmbeddings[currSense]
                currSim = dot(formatedTargetEmbedding, currSenseEmbedding) / (
                    norm(formatedTargetEmbedding) * norm(currSenseEmbedding)
                )  # Cosine similarity
                senseScores[currSense] = currSim

                if currSim < 0:
                    print("ERROR --- curSim - " + str(currSim))
            else:
                print(currSense + " not in sensembeddings")

        predSensembertRanks = {}
        orderedSensembert = [
            k
            for k, v in sorted(
                senseScores.items(), key=lambda item: item[1], reverse=True
            )
        ]
        senseRank = 1

        for currSense in orderedSensembert:
            predSensembertRanks[currSense] = senseRank
            senseRank += 1

        # If predomiannt sense is in top x predicted, then predict preodminant sense
        if predSensembertRanks[self.authorPredomSenses[authorID][lemma]] <= topX:
            predictedSense = self.authorPredomSenses[authorID][lemma]
        else:
            predictedSense = orderedSensembert[0]

        return predictedSense


if __name__ == "__main__":
    example_lemma = "deal"

    myPersonalizedModel = personalziedWSD()
    # myPersonalizedModel.loadDataset('./dataset.json')
    myPersonalizedModel.loadBERT(
        "bert-large-cased"
    )  # Should automatically download the model
    # myPersonalizedModel.sortDatasetByUsers(myPersonalizedModel.dataset)

    # myPersonalizedModel.loadSenseDistribution(myPersonalizedModel.sortedDataset)
    # load precalculated sense embeddings from a given path to a json object
    # Sense embeddings downloaded from http://sensembert.org/resources/sensembert_data.tar.gz

    myPersonalizedModel.loadSenseEmbertEmbeddingsText(
        "data/sensembert_EN_kb.txt"
    )  # Run this at least once to generate json
    myPersonalizedModel.loadSenseEmbeddings("data/sensembert_EN_kb.json")
    myPersonalizedModel.getAuthorPredomSense()
    # Example of predicting a sense of a word
    example_sent = "he is a master of the business <b>deal</b>"
    # Use predict sense to predict a sense using original senseEmbert
    predictedSense = myPersonalizedModel.predictSense(example_sent, example_lemma)
    print("original senseEmbert prediction: ", predictedSense)

    example_authorID = "1476382.male.33.Publishing.Gemini.json"
    example_topX = 3
    authorPredomPredictedSense = myPersonalizedModel.predictSenseUsingAuthorPredomSense(
        example_sent, example_lemma, example_authorID, example_topX
    )
    print(
        "prediction using author's predominant sense with senseEmbert: ",
        authorPredomPredictedSense,
    )
