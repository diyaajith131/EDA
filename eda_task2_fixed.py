# Exploratory Data Analysis - Task 2 (Fixed & Robust)
# This script includes safer file handling, creates outputs folder, and handles various edge cases.

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_PATHS = ["data/titanic.csv", "titanic.csv", "data/titanic.csv.zip"]  # try a few common locations
OUTPUT_DIR = "outputs"
MAX_PAIRPLOT_COLS = 5  # limit to avoid very slow plotting

# --- Ensure outputs folder exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load dataset with helpful error message ---
data_path = None
for p in DATA_PATHS:
    if os.path.exists(p):
        data_path = p
        break

if data_path is None:
    raise FileNotFoundError(\
        "Could not find the dataset. Please place the CSV at one of: {}. "\
        "Or change DATA_PATHS variable in the script.".format(", ".join(DATA_PATHS)))

print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)

# --- Basic info & summary ---
print("\\n=== Data Info ===")
print(df.info())
print("\\n=== Summary Statistics ===")
print(df.describe(include="all"))

# --- Missing values ---
print("\\n=== Missing Values (per column) ===")
print(df.isnull().sum())

# --- Select numeric columns ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns detected ({len(numeric_cols)}): {numeric_cols}")

if len(numeric_cols) == 0:
    print("No numeric columns found. Exiting plot generation.")
else:
    # --- Histograms ---
    n = len(numeric_cols)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() if isinstance(axes, (np.ndarray, list)) else [axes]
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=15)
        axes[i].set_title(col)
    # turn off extra axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, "histograms.png")
    fig.savefig(hist_path)
    plt.close(fig)
    print(f"Saved histograms -> {hist_path}")

    # --- Boxplots ---
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() if isinstance(axes, (np.ndarray, list)) else [axes]
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col].dropna(), ax=axes[i])
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    boxplot_path = os.path.join(OUTPUT_DIR, "boxplots.png")
    fig.savefig(boxplot_path)
    plt.close(fig)
    print(f"Saved boxplots -> {boxplot_path}")

    # --- Correlation heatmap ---
    num_df = df[numeric_cols].dropna(axis=1, how="all")  # drop entirely empty numeric cols
    if num_df.shape[1] >= 2:
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(10, max(6, 0.5 * num_df.shape[1])))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        heatmap_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
        fig.savefig(heatmap_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved correlation heatmap -> {heatmap_path}")
    else:
        print("Not enough numeric columns to compute a correlation matrix. Skipping heatmap.")

    # --- Pairplot (limit columns to avoid heavy compute) ---
    if len(numeric_cols) >= 2:
        pair_cols = numeric_cols[:MAX_PAIRPLOT_COLS]
        print(f"Generating pairplot for columns: {pair_cols}")
        # drop rows with NA only for selected cols
        pair_df = df[pair_cols].dropna()
        if pair_df.shape[0] == 0:
            print("No complete rows for selected columns, skipping pairplot.")
        else:
            g = sns.pairplot(pair_df)
            g.fig.suptitle("Pairplot", y=1.02)
            pairplot_path = os.path.join(OUTPUT_DIR, "pairplot.png")
            g.savefig(pairplot_path)
            plt.close()
            print(f"Saved pairplot -> {pairplot_path}")
    else:
        print("Not enough numeric columns for pairplot.")

print("\\n✅ EDA Completed. Check the 'outputs' directory for generated plots.")