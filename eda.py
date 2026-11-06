
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Load Dataset
df = pd.read_csv("iris.csv")

print("EXPLORATORY DATA ANALYSIS")

# First 5 Rows
print("\nFIRST 5 ROWS")
print(df.head())

# Dataset Information
print("\nDATASET INFORMATION")
df.info()

# Dataset Shape
print("\nDATASET SHAPE")
print(df.shape)

# Missing Values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Duplicate Values
print("\nDUPLICATE VALUES")
print(df.duplicated().sum())

# Statistical Summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Species Count
print("\nSPECIES COUNT")
print(df["species"].value_counts())

# Correlation
print("\nCORRELATION")
print(df.select_dtypes(include=np.number).corr())

# Hypothesis Testing using ANOVA
setosa = df[df["species"] == "setosa"]["petal_length"]
versicolor = df[df["species"] == "versicolor"]["petal_length"]
virginica = df[df["species"] == "virginica"]["petal_length"]

f_stat, p_value = f_oneway(setosa, versicolor, virginica)

print("\nHYPOTHESIS TESTING")
print("F-Statistic:", f_stat)
print("P-Value:", p_value)

if p_value < 0.05:
    print("Reject Null Hypothesis")
    print("Species have different average petal lengths")
else:
    print("Fail to Reject Null Hypothesis")

# Pair Plot
sns.pairplot(df, hue="species")
plt.suptitle("Iris Dataset Pair Plot", y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# Box Plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="species", y="petal_length")
plt.title("Petal Length by Species")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")

