"""
eda.py

Exploratory Data Analysis helper functions.
Generates simple plots and summaries used during project.
"""

import pandas as pd
import matplotlib.pyplot as plt

def summary_stats(df, out_path="reports/eda_summary.txt"):
    with open(out_path, "w") as f:
        f.write("Shape: %s\n\n" % (df.shape,))
        f.write(df.describe(include='all').to_string())
    return out_path

def plot_target_distribution(df, target="y", out_path="reports/target_dist.png"):
    counts = df[target].value_counts()
    plt.figure(figsize=(6,4))
    counts.plot(kind='bar')
    plt.title("Target distribution")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
