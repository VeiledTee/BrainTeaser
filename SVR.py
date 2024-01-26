from transformers import BertTokenizer, BertModel
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch

def get_bert_embeddings(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample data
questions = [
    "Mr. and Mrs. Mustard have six daughters and each daughter has one brother. But there are only 9 people in the family, how is that possible?",
    "Everyone called him 'Batman,' but he knew nothing about bats and thought they were disgusting. He still cherished being referred to as Batman! How is this possible?"
]

answer_sets = [
    [
        'Some daughters get married and have their own family.',
        'Each daughter shares the same brother.',
        'Some brothers were not loved by family and moved away.',
        'None of above.'
    ],
    [
        'He tries to be friendly.',
        'He is afraid others will laugh at him.',
        'He was the star baseball player.',
        'None of above.'
    ]
]

# Create features and labels for training
X = []
y = []

for i, question in enumerate(questions):
    question_embedding = get_bert_embeddings(question, model, tokenizer)
    for j, answer_text in enumerate(answer_sets[i]):
        answer_embedding = get_bert_embeddings(answer_text, model, tokenizer)
        X.append(torch.tensor(question_embedding - answer_embedding))  # Using the difference in embeddings
        y.append(j)  # Label the correct answer as 1, others as 0

# Convert lists to PyTorch tensors
X = torch.cat(X)
y = torch.tensor(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LinearSVR model
svr = LinearSVR()
svr.fit(X_train, y_train)

# Predict on the test set
predictions = svr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
