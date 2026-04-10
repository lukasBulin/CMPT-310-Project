# CMPT-310-Project
Authors - Karanveer Singh, Damin Mutti, Lukas Bulin, Jeel Patel

# How-To Guide: Building an AI-Based Spam Detection System

## 1. Introduction
Spam messages are a persistent issue across communication platforms such as SMS and email, reducing usability and posing security risks.

This guide walks you through building an AI-based spam detection system that classifies messages as Spam or Not Spam using supervised machine learning.

We compare two feature extraction techniques:
- Bag-of-Words (BoW)
- TF-IDF (Term Frequency-Inverse Document Frequency)

By the end, you will know how to:
- Preprocess raw text data
- Extract meaningful features
- Train and evaluate ML models
- Analyze errors using confusion matrices
- Compare models and improve performance

## 2. Prerequisites
Software:
- Python 3.8+
- Jupyter Notebook or any Python IDE

Libraries:
pip install pandas numpy matplotlib seaborn scikit-learn

Dataset:
- UCI SMS Spam Collection Dataset
- Stored as: data/spam.csv

## 3. Step 1: Loading and Exploring the Dataset
- "ham" → Not Spam (0)
- "spam" → Spam (1)

Notes:
- Convert labels to numeric
- Remove unnecessary columns
- Dataset is imbalanced

## 4. Step 2: Splitting the Data
- Use train-test split
- Use stratify to maintain class distribution

## 5. Step 3: Feature Extraction
BoW:
- Counts word frequency
- Ignores common words

TF-IDF:
- Weighs words by importance
- Reduces impact of common words

## 6. Step 4: Model Training
- Logistic Regression (BoW)
- Logistic Regression (TF-IDF)
- Naive Bayes (TF-IDF)

## 7. Step 5: Model Evaluation
Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Important:
- False Positives (FP): Normal → Spam
- False Negatives (FN): Spam missed

## 8. Step 6: Error Analysis
- FP harms trust
- FN reduces effectiveness

## 9. Step 7: Model Comparison
- BoW ~97.6%
- TF-IDF & NB ~96–97%

## 10. Step 8: Visualization
- Accuracy chart
- Confusion matrix heatmap

## 11. Step 9: Making Predictions
- Input text → model → prediction

## 12. Troubleshooting
Issue: Predicts one class
Fix:
- Use stratify
- Use class weights

Issue: Low accuracy
Fix:
- Improve preprocessing
- Try TF-IDF or other models

License:
We grant permission to instructors to share this guide.
