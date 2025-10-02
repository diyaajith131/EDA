# Exploratory Data Analysis - Task 2
# Internship Assignment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/titanic.csv")

# ----- 1. Summary Statistics -----
print("Summary Statistics:\n", df.describe(include="all"))

# ----- 2. Missing Values -----
print("\nMissing Values:\n", df.isnull().sum())

# ----- 3. Histograms -----
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols].hist(bins=15, figsize=(12, 10), layout=(len(numeric_cols)//3+1, 3))
plt.tight_layout()
plt.savefig("outputs/histograms.png")
plt.close()

# ----- 4. Boxplots -----
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=df, y=col)
plt.tight_layout()
plt.savefig("outputs/boxplots.png")
plt.close()

# ----- 5. Correlation Heatmap -----
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# ----- 6. Pairplot -----
sns.pairplot(df[numeric_cols[:5]])  # only first few numeric cols to avoid clutter
plt.savefig("outputs/pairplot.png")
plt.close()

print("✅ EDA Completed. Check outputs/ folder for plots.")
