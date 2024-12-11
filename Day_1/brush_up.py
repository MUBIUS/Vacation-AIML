
# # NumPy example
# import numpy as np
# arr = np.array([1, 2, 3, 4, 5])
# print(arr * 2)

# Pandas example
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
df = pd.read_csv("Auction_List.csv")
print(df.head())
# print(df.info())  # Overview of data types
print(df.describe())  # Summary statistics
print(df.shape)  # Dimensions of the dataset
print(df['Unnamed: 20'])

# Histogram
# df['Unnamed: 20'].hist(bins=20)
# plt.title("Histogram of Column")
# plt.show()

df = df.select_dtypes(include=["float64", "int64"])  # Keep only numerical columns


sns.pairplot(df)
plt.show()






# list = []

# for i in range(1,20):

#     print(i)
    
#     if i % 2== 0:
        
#         list.append(i)
        


# print(list)
