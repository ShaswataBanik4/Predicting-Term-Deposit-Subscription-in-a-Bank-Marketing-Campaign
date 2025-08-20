"""
evaluate.py

Load saved models and evaluate on test set. Produces classification report and saves confusion matrix.
"""

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data_loader import load_data, basic_cleaning
from src.features import prepare_data, build_preprocessor

def evaluate_model(model_path, X_test, y_test, name="model", out_folder="reports"):
    os.makedirs(out_folder, exist_ok=True)
    clf = joblib.load(model_path)
    preds = clf.predict(X_test)
    probs = None
    try:
        probs = clf.predict_proba(X_test)[:,1]
    except Exception:
        pass
    report = classification_report(y_test, preds, output_dict=False)
    with open(f"{out_folder}/{name}_report.txt", "w") as f:
        f.write(report)
    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{out_folder}/{name}_confusion.png")
    plt.close()
    if probs is not None:
        auc = roc_auc_score(y_test, probs)
        with open(f"{out_folder}/{name}_auc.txt", "w") as f:
            f.write(f"AUC: {auc}\n")
    return f"{out_folder}/{name}_report.txt"
