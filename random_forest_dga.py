import math
import requests
import pickle

from collections import Counter, defaultdict
from itertools import groupby

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv('/Users/florence/Desktop/dga_domains_full.csv')
print(df.head())


# Initialize model (suggested parameters)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    max_features='sqrt',
    random_state=42,
    bootstrap=True,
    oob_score=True,
)


# Example of extracting common TLDs
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


# Use only legitimate domains to build the transition matrix
legit_domains = df[df.iloc[:, 0] == 'legit'].iloc[:, 2].tolist()
prob_matrix = build_transition_matrix(legit_domains)

with open('prob_matrix.pkl', 'wb') as f:
    pickle.dump(prob_matrix, f)


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
print("Label distribution:", pd.Series(y).value_counts())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example for hyperparameter tuning (commented out)
# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
# grid_search.fit(X_train, y_train)
# best_model = grid_search.best_estimator_
# print("Best Parameters:", grid_search.best_params_)
