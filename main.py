from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input text
text = "Mr. and Mrs. Mustard have six daughters and each daughter has one brother. But there are only 9 people in the family, how is that possible?"
target_token = 'Mustard'

def get_target_embeddings(text, target_token):
    sentence_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    target_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(target_token)))

    tokens_to_remove = {'[CLS]', '[SEP]'}
    target_tokens = [token for token in target_tokens if token not in tokens_to_remove]

    # Convert tokens to input IDs
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Get the model's output
    with torch.no_grad():
        outputs = model(input_ids)

    # Retrieve embeddings for each subtoken
    hidden_states = outputs.last_hidden_state
    subtoken_embeddings = hidden_states[0]

    mask = [False] * len(sentence_tokens)

    # Slide a window over the larger list
    window_size = len(target_tokens)
    for i in range(len(sentence_tokens) - window_size + 1):
        window_tokens = sentence_tokens[i:i + window_size]
        if window_tokens == target_tokens:
            # If the window matches the smaller list, set the mask to True for those tokens
            mask[i:i + window_size] = [True] * window_size

    # Calculate the mean pooling of subtoken embeddings for tokens related to the target_token
    token_embedding = torch.mean(subtoken_embeddings[mask], dim=0)

    # Print the resulting embedding
    print("Token Embedding:", token_embedding.numpy().shape)

get_target_embeddings("Mr. and Mrs. Mustard have six daughters and each daughter has one brother. But there are only 9 people in the family, how is that possible?", "Mustard")


