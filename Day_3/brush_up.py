from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame  # Get as a DataFrame

# # Explore the dataset
# print(df.head())
# print(df.info())



# # Scatter plot for RM (average number of rooms) vs. PRICE
# sns.scatterplot(x=df['MedInc'], y=df['MedHouseVal'])
# plt.title("MedInc vs MedHouseVal")
# plt.xlabel("MedInc")
# plt.ylabel("MedHouseVal")
# plt.show()

# # Heatmap to find relationships
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.title("Feature Correlations")
# plt.show()

from sklearn.model_selection import train_test_split

# Features and target
X = df[['MedInc']]  # Input feature
y = df['MedHouseVal']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Model coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")


# Predict on test data
y_pred = model.predict(X_test)

# Compare predictions and actual values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted House Prices")
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# Plotting the regression line
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("Price")
plt.legend()
plt.title("Linear Regression Model")
plt.show()

