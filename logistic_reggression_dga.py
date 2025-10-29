import math
import requests
import pickle

from collections import Counter, defaultdict
from itertools import groupby

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv('CSV.csv')
print(df.head())
print('Total rows:', len(df))

# Initialize Logistic Regression model (suggested parameters)
model = LogisticRegression(
    penalty='l2',            # Regularization type (L2 helps prevent overfitting)
    solver='lbfgs',          # Optimization algorithm (good for small/medium datasets)
    max_iter=1000,           # Increase iterations to ensure convergence
    class_weight='balanced'  # Automatically balance class weights (for imbalanced datasets)
)

# Optional hyperparameter tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],  # Inverse of regularization strength (smaller = stronger regularization)
    'penalty': ['l2', 'none']        # Try with/without regularization
}

# Example of extracting common TLDs (optional)
# df['tld'] = df['domain'].str.extract(r'\.([^.]+)$')
# tld_counts = df['tld'].value_counts()
# COMMON_TLDS = set(tld_counts.head(20).index)
# print("Common TLDs: ", COMMON_TLDS)

# Training phase: build and save transition probability matrix
def build_transition_matrix(domains):
    transitions = defaultdict(lambda: defaultdict(int))
    chars = set('abcdefghijklmnopqrstuvwxyz')

    for domain in domains:
        domain = str(domain).lower().strip()
        for prev, curr in zip(domain[:-1], domain[1:]):
            if prev in chars and curr in chars:
                transitions[prev][curr] += 1

    prob_matrix = {}
    for prev in transitions:
        total = sum(transitions[prev].values())
        prob_matrix[prev] = {curr: count / total for curr, count in transitions[prev].items()}
    return prob_matrix

# Build transition probabilities using legitimate domains only
legit_domains = df[df.iloc[:, 0] == 'legit'].iloc[:, 2].tolist()
prob_matrix = build_transition_matrix(legit_domains)

with open('prob_matrix.pkl', 'wb') as f:
    pickle.dump(prob_matrix, f)

# Feature extraction
def extract_features(domain):
    domain = str(domain).lower()
    bigrams = [domain[i:i + 2] for i in range(len(domain) - 1)]

    trans_prob = []
    for prev, curr in zip(domain[:-1], domain[1:]):
        trans_prob.append(prob_matrix.get(prev, {}).get(curr, 0.0001))
    return {
        'length': len(domain),
        'vowel_ratio': sum(1 for c in domain if c in 'aeiou') / len(domain),
        'digit_ratio': sum(1 for c in domain if c.isdigit()) / len(domain),
        'special_char_ratio': sum(1 for c in domain if not c.isalnum()) / len(domain),
        'bigram_entropy': calculate_entropy(bigrams),
        'consecutive_consonant': max(
            (len(list(g)) for k, g in groupby(domain)
             if k.lower() in 'bcdfghjklmnpqrstvwxyz'), default=0),
        'unique_char_ratio': len(set(domain)) / len(domain),
        'trans_mean': np.mean(trans_prob),
        'trans_min': np.min(trans_prob),
        'trans_rare': sum(p < 0.01 for p in trans_prob) / len(trans_prob),
    }

# Calculate entropy of bigrams
def calculate_entropy(domain):
    counter = Counter(domain)
    entropy = 0.0
    for count in counter.values():
        p = count / len(domain)
        entropy -= p * math.log2(p)
    return entropy

# Build features and labels
features = df.iloc[:, 2].apply(extract_features).apply(pd.Series)
X = features
y = np.where(df.iloc[:, 0] == 'dga', 1, 0)
print("Label distribution:\n", pd.Series(y).value_counts())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example for hyperparameter tuning (commented out)
# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
# grid_search.fit(X_train, y_train)
# best_model = grid_search.best_estimator_
# print("Best Parameters:", grid_search.best_params_)
