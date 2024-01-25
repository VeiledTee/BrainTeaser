import nltk
from nltk.corpus import wordnet
import torch
from transformers import BertTokenizer, BertModel
from POSTagging import SimplePOSTagger


def get_senses_of_lemma(lemma):
    lemma_senses = []
    for curr_sense in wordnet.synsets(lemma, pos="n"):
        lemma_senses += [curr_sense.lemmas()[0].key()]
        print(lemma_senses[-1])
    return lemma_senses


class BrainTeaserWSD:
    def __init__(self, bert_model: str = "bert-large-cased"):
        self.model = BertModel.from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

    def get_bert_token_embedding(self, text, device="cpu"):
        marked_text = "[CLS] " + text.replace("<b>", "").replace("</b>", "") + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)[:512]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)[:512]

        tokens_tensor = torch.tensor([indexed_tokens]).to(device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            hidden_states = outputs[2]
            word_embeddings = torch.stack(hidden_states[-4:]).sum(0)[0]

        num_parts = 0.0
        target_token_embedding = None

        for token_index in self.findTargetInBertTokeinzedText(tokenized_text):
            num_parts += 1.0
            target_token_embedding = (
                target_token_embedding + word_embeddings[token_index]
                if target_token_embedding
                else word_embeddings[token_index]
            )

        token_embedding = (
            target_token_embedding / num_parts if target_token_embedding else None
        )

        return token_embedding

    def pos_tag_text(self, text: str) -> str:
        return ""


# # Print information about the synset
# print("Word:", synset.name().split('.')[0])
# print("Part of Speech:", synset.pos())
# print("Definition:", synset.definition())

get_senses_of_lemma("father")
