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


#Loading works but add handling for different datasets

X = data["message"]
y = data["label"]

# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 3. VECTORIZERS
# ============================================================

bow_vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english"
)

tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

# ============================================================
# 4. MODELS
# ============================================================

models = {
    "BoW + Logistic Regression": Pipeline([
        ("vectorizer", bow_vectorizer),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "TF-IDF + Logistic Regression": Pipeline([
        ("vectorizer", tfidf_vectorizer),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "TF-IDF + Naive Bayes": Pipeline([
        ("vectorizer", tfidf_vectorizer),
        ("classifier", MultinomialNB())
    ]),

    "TF-IDF + SVM": Pipeline([
        ("vectorizer", tfidf_vectorizer),
        ("classifier", SVC(kernel="linear", probability=True))
    ]),

    "TF-IDF + Decision Tree": Pipeline([
        ("vectorizer", tfidf_vectorizer),
        ("classifier", DecisionTreeClassifier(max_depth=20))
    ])
}




