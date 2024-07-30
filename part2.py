import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv('concrete_data.csv')

# Display basic information and first few rows
print(df.info())
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Visualize the distribution of the target variable
sns.histplot(df['Concrete compressive strength(MPa, megapascals)'], kde=True)
plt.title('Distribution of Concrete Compressive Strength')
plt.xlabel('Compressive Strength (MPa)')
plt.ylabel('Frequency')
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Normalize numerical features
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df)

# Split the data into train and test sets
X = df.drop('Concrete compressive strength(MPa, megapascals)', axis=1)
y = df['Concrete compressive strength(MPa, megapascals)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

print("Ridge Regression Performance:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_ridge))
print("R2 Score:", r2_score(y_test, y_pred_ridge))

# Visualize the predictions vs actual values
plt.scatter(y_test, y_pred_ridge)
plt.xlabel('Actual Compressive Strength')
plt.ylabel('Predicted Compressive Strength')
plt.title('Ridge Regression Predictions vs Actual Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.show()

# Hyperparameter tuning for Ridge Regression
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(Ridge(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best parameters for Ridge Regression:")
print(grid.best_params_)

# Use the best model
best_ridge = grid.best_estimator_
y_pred_best_ridge = best_ridge.predict(X_test)
print("Tuned Ridge Regression Performance:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_best_ridge))
print("R2 Score:", r2_score(y_test, y_pred_best_ridge))

# Visualize the tuned model predictions vs actual values
plt.scatter(y_test, y_pred_best_ridge)
plt.xlabel('Actual Compressive Strength')
plt.ylabel('Predicted Compressive Strength')
plt.title('Tuned Ridge Regression Predictions vs Actual Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.show()
