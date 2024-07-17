import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
df = sns.load_dataset('iris')

# Display the first few rows of the dataset
print(df.head())

# Display the structure and basic information of the dataset
print(df.info())

# Display summary statistics for numerical columns
print(df.describe())

# Histograms
df.hist(bins=20, figsize=(10, 10))
plt.show()

# Boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)                 #FOR DETECTING OUTLIERS IN DATASET
plt.show()

# Correlation matrix
corr_matrix = df.corr()

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

#Pair Plot as Scatter Plot in SeaBorn
sns.pairplot(df, hue='species', markers=["o", "s", "D"])

# Display the plot
plt.show()

