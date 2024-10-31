import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'calories_burnt_data.csv'  # Adjust this path if necessary
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# Drop unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Check the data structure
print("\nDataset Head:")
print(data.head())
print(data.info())

# Check for missing values
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

# Drop rows with any missing values
data.dropna(inplace=True)

# Ensure correct data types
data['Calories Burnt'] = pd.to_numeric(data['Calories Burnt'], errors='coerce')
data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce')

# Box Plot for Calories Burnt
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, y='Calories Burnt')
plt.title("Box Plot of Calories Burnt")
plt.show()

# Histogram of Calories Burnt
plt.figure(figsize=(8, 6))
data['Calories Burnt'].hist(bins=20, color='skyblue')
plt.title("Histogram of Calories Burnt")
plt.xlabel("Calories Burnt")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot for Duration vs Calories Burnt
plt.figure(figsize=(8, 6))
plt.scatter(data['Duration'], data['Calories Burnt'], color='green')
plt.title("Duration vs Calories Burnt")
plt.xlabel("Duration (minutes)")
plt.ylabel("Calories Burnt")
plt.show()

# Pearson Correlation Analysis
print("\nPearson Correlation Matrix:")
correlation_matrix = data.corr()
print(correlation_matrix)

# Plot the Correlation Matrix with a Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Prepare data for model
# Define independent variables and target variable
X = data[['Duration', 'Age', 'Weight', 'Intensity']]  # Make sure these columns exist in your dataset
y = data['Calories Burnt']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot Predictions vs Actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Calories Burnt")
plt.ylabel("Predicted Calories Burnt")
plt.title("Actual vs Predicted Calories Burnt")
plt.show()
