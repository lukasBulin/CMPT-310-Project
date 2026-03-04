import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

#-------------------
#Data set Loading---
#-------------------


# Resolve path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "spam.csv")

# Load CSV
data = pd.read_csv(DATA_PATH, encoding='latin-1')

# Keep only the useful columns
data = data[["v1", "v2"]]

# Rename columns to standard names
data.columns = ["label", "message"]

# Encode labels: ham=0, spam=1
data["label"] = data["label"].map({"ham": 0, "spam": 1})

print("Dataset shape:", data.shape)
print(data.head())
print("\nClass distribution:")
print(data["label"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

bow_vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow  = bow_vectorizer.transform(X_test)

bow_model = LogisticRegression(max_iter=1000)
bow_model.fit(X_train_bow, y_train)

y_pred_bow = bow_model.predict(X_test_bow)
y_prob_bow = bow_model.predict_proba(X_test_bow)[:, 1]

print("\n--- Bag of Words Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_bow))
print(confusion_matrix(y_test, y_pred_bow))
print(classification_report(y_test, y_pred_bow))


tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf  = tfidf_vectorizer.transform(X_test)

tfidf_model = LogisticRegression(max_iter=1000)
tfidf_model.fit(X_train_tfidf, y_train)

y_pred_tfidf = tfidf_model.predict(X_test_tfidf)
y_prob_tfidf = tfidf_model.predict_proba(X_test_tfidf)[:, 1]

print("\n--- TF-IDF Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print(confusion_matrix(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf))

def predict_message(message, vectorizer, model):
    features = vectorizer.transform([message])
    pred = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][pred]
    label = "SPAM" if pred == 1 else "NOT SPAM"
    return label, confidence

test_message = "I am Jeel and i wanted to talk to you about your insurance-"

label, conf = predict_message(test_message, tfidf_vectorizer, tfidf_model)
print(f"\nMessage: {test_message}")
print(f"Prediction: {label}")
print(f"Confidence: {conf:.4f}")