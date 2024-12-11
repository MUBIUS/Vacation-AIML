import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
# df = pd.read_csv("iris.csv")
df = sns.load_dataset("iris")

# Check for numeric columns
numeric_df = df.select_dtypes(include=["float64", "int64"])

# Handle missing or invalid data
numeric_df = numeric_df.dropna()  # Drop rows with missing values

print(df.info())
print(df.head(10))
# Plot pairplot
# sns.pairplot(numeric_df)
# plt.show()

# Heatmap for correlation matrix
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
