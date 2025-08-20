"""
train.py

Example training script that ties everything together.
Usage:
    python src/train.py
Assumes data/bank.csv is present.
"""

from src.data_loader import load_data, basic_cleaning
from src.features import prepare_data, build_preprocessor
from src.models import train_logistic, train_nb, train_knn, train_tree
from sklearn.model_selection import GridSearchCV
import os

def main():
    os.makedirs("models", exist_ok=True)
    df = load_data("data/bank.csv")
    df = basic_cleaning(df)
    X_train, X_test, y_train, y_test = prepare_data(df, target_col='y')
    preprocessor = build_preprocessor(df)
    # Train models
    logistic = train_logistic(preprocessor, X_train, y_train, out_path="models/logistic.pkl")
    nb = train_nb(preprocessor, X_train, y_train, out_path="models/nb.pkl")
    knn = train_knn(preprocessor, X_train, y_train, n_neighbors=5, out_path="models/knn.pkl")
    tree = train_tree(preprocessor, X_train, y_train, max_depth=6, out_path="models/tree.pkl")
    print("Training finished. Models saved to models/*.pkl")

if __name__ == "__main__":
    main()
