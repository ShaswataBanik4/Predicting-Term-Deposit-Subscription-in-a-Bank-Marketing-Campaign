"""
models.py

Defines training functions for various models:
- Logistic Regression
- GaussianNB
- KNeighborsClassifier
- DecisionTreeClassifier

Saves trained models with joblib.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import joblib

def train_logistic(preprocessor, X_train, y_train, out_path="models/logistic.pkl"):
    clf = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    clf.fit(X_train, y_train)
    joblib.dump(clf, out_path)
    return clf

def train_nb(preprocessor, X_train, y_train, out_path="models/nb.pkl"):
    clf = Pipeline([
        ('pre', preprocessor),
        ('clf', GaussianNB())
    ])
    # GaussianNB requires dense numeric arrays; pipeline handles encoding with dense output
    clf.fit(X_train, y_train)
    joblib.dump(clf, out_path)
    return clf

def train_knn(preprocessor, X_train, y_train, n_neighbors=5, out_path="models/knn.pkl"):
    clf = Pipeline([
        ('pre', preprocessor),
        ('clf', KNeighborsClassifier(n_neighbors=n_neighbors))
    ])
    clf.fit(X_train, y_train)
    joblib.dump(clf, out_path)
    return clf

def train_tree(preprocessor, X_train, y_train, max_depth=None, out_path="models/tree.pkl"):
    clf = Pipeline([
        ('pre', preprocessor),
        ('clf', DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced'))
    ])
    clf.fit(X_train, y_train)
    joblib.dump(clf, out_path)
    return clf
