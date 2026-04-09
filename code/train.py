#----------
# IMPORTS
#----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
# Data set Loading
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

# Print Dataset info
print("Dataset shape:", data.shape)
print(data.head())
print("\nClass distribution:")
print(data["label"].value_counts())

#-------------------
# Train-Test Split
#-------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"] #Preserves class balance
)

#--------------------
# Bag of Words model
#---------------------

# Convert text to BoW features
bow_vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english"
)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow  = bow_vectorizer.transform(X_test)

# Train Logistic regression model
bow_model = LogisticRegression(max_iter=1000)
bow_model.fit(X_train_bow, y_train)

# Predictions
y_pred_bow = bow_model.predict(X_test_bow)
y_prob_bow = bow_model.predict_proba(X_test_bow)[:, 1]

# Evaluation
print("\n--- Bag of Words Results ---")
acc_bow = accuracy_score(y_test, y_pred_bow)
print("Accuracy:", acc_bow)
print(confusion_matrix(y_test, y_pred_bow))
print(classification_report(y_test, y_pred_bow))

# Error Analysis (tn,tp,fn,fp)
tn_bow,fp_bow,fn_bow,tp_bow = confusion_matrix(y_test, y_pred_bow).ravel()
print("\n--- Error Analysis (BoW) ---")
print("False Positives:", fp_bow)
print("False Negatives:", fn_bow)

#---------------------
# TF-IDF model
#---------------------

# Convert text to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf  = tfidf_vectorizer.transform(X_test)

# Logistic Regression model
tfidf_model = LogisticRegression(max_iter=1000)
tfidf_model.fit(X_train_tfidf, y_train)

# Predictions
y_pred_tfidf = tfidf_model.predict(X_test_tfidf)
y_prob_tfidf = tfidf_model.predict_proba(X_test_tfidf)[:, 1]

# Evaluation
print("\n--- TF-IDF Results ---")
acc_tfidf = accuracy_score(y_test, y_pred_tfidf)
print("Accuracy:", acc_tfidf)
print(confusion_matrix(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf))


# Error Analysis
tn,fp,fn,tp = confusion_matrix(y_test, y_pred_tfidf).ravel()
print("\n--- Error Analysis (TF-IDF) ---")
print("False Positives:", fp)
print("False Negatives:", fn)

# Naive Bayes Model (trains on tdidf features)
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
acc_nb = accuracy_score(y_test, y_pred_nb)

print("\n--- Naive Bayes Results ---")
print("Accuracy:", acc_nb)
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))


print("\n--- Model Comparison ---")
print("BoW Accuracy:", acc_bow)
print("TF-IDF Accuracy:", acc_tfidf)
print("Naive Bayes Accuracy:", acc_nb)

#naive bayes tn,fp,fn,tp
tn_nb, fp_nb, fn_nb, tp_nb = confusion_matrix(y_test, y_pred_nb).ravel()
print("\n--- Error Analysis (Naive Bayes) ---")
print("False Positives:", fp_nb)
print("False Negatives:", fn_nb)


# -------------------
# Visualizations ----
# -------------------

RESULT_DIR = os.path.join(BASE_DIR, "..", "result")

# 1. Accuracy comparison bar chart
models = ["BoW", "TF-IDF", "Naive Bayes"]
accuracies = [acc_bow, acc_tfidf, acc_nb]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=["#4c72b0", "#dd8452", "#55a868"])
plt.ylim(0.9, 1.0)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
             f"{acc:.4f}", ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "model_accuracy.png"))
plt.close()

# 2. Confusion matrix heatmap for TF-IDF
cm_tfidf = confusion_matrix(y_test, y_pred_tfidf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_tfidf, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - TF-IDF Model")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix_tfidf.png"))
plt.close()

print("\nPlots saved to result/ folder.")

# 3. Confusion matrix heatmap for BoW
cm_bow = confusion_matrix(y_test, y_pred_bow)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_bow, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - BoW Model")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix_bow.png"))
plt.close()

# 4. Confusion matrix heatmap for Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes Model")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix_nb.png"))
plt.close()



def predict_message(message, vectorizer, model):
    features = vectorizer.transform([message])
    pred = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][pred]
    label = "SPAM" if pred == 1 else "NOT SPAM"
    return label, confidence

test_message = "Please call our customer service representative as you have an appointment"

label, conf = predict_message(test_message, tfidf_vectorizer, tfidf_model)
print(f"\nMessage: {test_message}")
print(f"Prediction: {label}")
print(f"Confidence: {conf:.4f}")
