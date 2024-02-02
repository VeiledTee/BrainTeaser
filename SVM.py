from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Example data
data = [
    {
        "question": "What kind of nut has no shell?",
        "choice_list": ["A peanut.", "A Doughnut.", "A walnut.", "None of above."],
        "answer": "A Doughnut",
    },
    # Add more examples as needed
]

# Extract features and labels
questions = [item["question"] for item in data]
choices = [" ".join(item["choice_list"]) for item in data]
labels = [
    1 if item["choice_list"][i] == item["answer"] else 0
    for item in data
    for i in range(len(item["choice_list"]))
]

# Combine question and choices
combined_text = [f"{question} {choices[i]}" for i, question in enumerate(choices)]
print(combined_text)

# Ensure consistency in the length of combined_text and labels
assert len(combined_text) == len(labels), "Inconsistent number of samples"

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    combined_text, labels, test_size=0.2, random_state=42
)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Support Vector Machine (SVM) classifier
classifier = SVC(kernel="linear")
classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
