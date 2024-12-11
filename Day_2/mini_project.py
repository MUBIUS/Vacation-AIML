import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("iris.csv", header=None)

# Add column names if necessary
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Summary statistics
print(df.describe())

# Histogram
df['sepal_length'].hist(bins=20)
plt.title("Histogram of Sepal Length")
plt.show()

numeric_df = sns.load_dataset("iris")
numeric_df = numeric_df.select_dtypes(include=["float64", "int64"])
numeric_df = numeric_df.dropna()
# Correlation heatmap
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
