import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import resample
from fastapi import FastAPI
from pydantic import BaseModel
from fuzzywuzzy import fuzz
import spacy
import json
import pickle

# Load the dataset
df = pd.read_csv("calls_dataset.csv")

# Preprocess labels (convert to list of labels)
df["labels"] = df["labels"].apply(lambda x: x.split(", "))

# Clean text
df["cleaned_text"] = df["text_snippet"].str.lower()

# Balance the dataset by oversampling minority labels
balanced_data = pd.DataFrame()
for label in set([label for sublist in df["labels"] for label in sublist]):
    subset = df[df["labels"].apply(lambda x: label in x)]
    oversampled = resample(subset, replace=True, n_samples=50, random_state=42)
    balanced_data = pd.concat([balanced_data, oversampled])

df = balanced_data.reset_index(drop=True)

# Encode labels and vectorize text
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["labels"])
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X = tfidf.fit_transform(df["cleaned_text"])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = OneVsRestClassifier(XGBClassifier(eval_metric="logloss"))
model.fit(X_train, y_train)

# Predict probabilities and classify using threshold
threshold = 0.3
y_probs = model.predict_proba(X_test)
y_pred = (y_probs >= threshold).astype(int)

# Metrics: Classification Report and Confusion Matrix
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))

for i, label in enumerate(mlb.classes_):
    cm = confusion_matrix(y_test[:, i], y_pred[:, i])
    print(f"Confusion Matrix for label '{label}':\n{cm}")

# Heatmap of Label Co-occurrences
label_co_occurrences = np.dot(y.T, y)
sns.heatmap(label_co_occurrences, annot=True, xticklabels=mlb.classes_, yticklabels=mlb.classes_, cmap="Blues")
plt.title("Label Co-occurrence Heatmap")
plt.show()

# Load domain knowledge
with open("domain_knowledge.json", "r") as f:
    domain_knowledge = json.load(f)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define fuzzy matching with stricter thresholds
def fuzzy_match(text, keywords, threshold=90):
    """
    Perform fuzzy matching on the text with a stricter threshold.
    Returns keywords with their scores for debugging.
    """
    matches = []
    for keyword in keywords:
        score = fuzz.partial_ratio(keyword.lower(), text.lower())
        if score >= threshold:
            matches.append((keyword, score))
    return matches

# Enhanced entity extraction
def extract_entities(text):
    """
    Extract entities from the text using domain knowledge and spaCy NER.
    """
    entities = {}

    # Domain-specific extraction with stricter fuzzy matching
    for category, keywords in domain_knowledge.items():
        # Exact matching first
        exact_matches = [kw for kw in keywords if kw.lower() in text.lower()]
        if exact_matches:
            entities[category] = exact_matches
            continue  # Skip fuzzy matching if exact matches exist

        # Fuzzy matching as fallback
        matches = fuzzy_match(text, keywords, threshold=90)
        if matches:
            entities[category] = [match[0] for match in matches]

    # spaCy NER extraction
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)

    # Debugging: Log matched entities and scores
    print(f"Text: {text}")
    print(f"Extracted Entities: {entities}")

    return entities

# FastAPI setup
app = FastAPI()

# API input model
class TextSnippet(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(snippet: TextSnippet):
    # Predict labels
    vectorized_text = tfidf.transform([snippet.text])
    probs = model.predict_proba(vectorized_text)
    pred_labels = (probs >= threshold).astype(int)
    labels = mlb.inverse_transform(pred_labels)

    # Extract entities
    entities = extract_entities(snippet.text)

    # Log for debugging
    print(f"Text: {snippet.text}")
    print(f"Predicted Labels: {labels}")
    print(f"Entities: {entities}")

    # Generate summary
    summary = snippet.text[:50] + "..." if len(snippet.text) > 50 else snippet.text

    return {"labels": labels, "entities": entities, "summary": summary}

# Save the model, vectorizer, and label encoder
with open("xgboost_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

with open("mlb.pkl", "wb") as mlb_file:
    pickle.dump(mlb, mlb_file)