# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set visual style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# Step 2: Load the Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Step 3: Preview the Dataset
print("🔹 First 5 Rows:\n", df.head())
print("\n🔹 Info:\n")
print(df.info())
print("\n🔹 Description:\n", df.describe())
print("\n🔹 Class Distribution:\n", df['species'].value_counts())

# Step 4: Check for Missing Values
print("\n🔹 Missing Values:\n", df.isnull().sum())

# Step 5: Univariate Analysis - Histograms
for col in df.columns[:-1]:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Step 6: Pairplot
sns.pairplot(df, hue="species", corner=True, palette="Set2")
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# Step 7: Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Step 8: Boxplots by Species
for col in df.columns[:-1]:
    plt.figure()
    sns.boxplot(x="species", y=col, data=df, palette="Set3")
    plt.title(f"{col} by Species")
    plt.tight_layout()
    plt.show()

# Step 9: Violin Plots (Optional)
for col in df.columns[:-1]:
    plt.figure()
    sns.violinplot(x="species", y=col, data=df, palette="Pastel1")
    plt.title(f"Violin Plot of {col} by Species")
    plt.tight_layout()
    plt.show()

print("\n✅ EDA Complete.")
