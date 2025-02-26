import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fetch the dataset using the ucimlrepo package
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# Extract features and target
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# Print metadata and variable information
print("Metadata:")
print(breast_cancer_wisconsin_diagnostic.metadata)
print("\nVariable Information:")
print(breast_cancer_wisconsin_diagnostic.variables)

y_numeric = y.replace({'M': 1, 'B': 0})

data = X.copy()
data['diagnosis'] = y_numeric

# Calculate correlations between each feature and the diagnosis
correlations = data.corr()['diagnosis'].abs().sort_values(ascending=False)
print("\nCorrelations with Diagnosis:")
print(correlations)

# Identify the most predictive feature (ignoring the target itself)
most_predictive_feature = correlations.index[1]  # index 0 is 'diagnosis'
print(f"\nMost Predictive Feature: {most_predictive_feature}")

# Scatter plot of the most predictive feature vs. diagnosis
plt.figure(figsize=(8, 6))
plt.scatter(data[most_predictive_feature], data['diagnosis'], alpha=0.6)
plt.xlabel(most_predictive_feature)
plt.ylabel('Diagnosis (0 = Benign, 1 = Malignant)')
plt.title(f'Scatter Plot of {most_predictive_feature} vs. Diagnosis')
plt.show()

# --------------------------- ** Too keep everything seperated and clean
# Split the data into training and test sets
X_data = data.drop(columns=['diagnosis'])
y_data = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# ---------------------------
# Model 1: Linear regression using the single most predictive feature
X_train_single = X_train[[most_predictive_feature]]
X_test_single = X_test[[most_predictive_feature]]

model_single = LinearRegression()
model_single.fit(X_train_single, y_train)

y_train_pred_single = model_single.predict(X_train_single)
y_test_pred_single = model_single.predict(X_test_single)

mse_train_single = mean_squared_error(y_train, y_train_pred_single)
mse_test_single = mean_squared_error(y_test, y_test_pred_single)
print("\nSingle Feature Model:")
print("Training MSE:", mse_train_single)
print("Test MSE:", mse_test_single)

# Overlay the line of best fit on the scatter plot for the training data
plt.figure(figsize=(8, 6))
plt.scatter(X_train_single, y_train, alpha=0.6, label='Training Data')
plt.plot(X_train_single, model_single.predict(X_train_single), color='red', label='Best Fit Line')
plt.xlabel(most_predictive_feature)
plt.ylabel('Diagnosis (0 = Benign, 1 = Malignant)')
plt.title(f'Best Fit Line for {most_predictive_feature}')
plt.legend()
plt.show()

# Model 2: Linear regression using the top 3 predictive features
top_features = correlations.index[1:4]  # Select the top 3 features (after 'diagnosis')
print("\nTop 3 Features:", list(top_features))

X_train_top = X_train[list(top_features)]
X_test_top = X_test[list(top_features)]

model_top = LinearRegression()
model_top.fit(X_train_top, y_train)

y_train_pred_top = model_top.predict(X_train_top)
y_test_pred_top = model_top.predict(X_test_top)

mse_train_top = mean_squared_error(y_train, y_train_pred_top)
mse_test_top = mean_squared_error(y_test, y_test_pred_top)
print("\nTop Features Model (3 features):")
print("Training MSE:", mse_train_top)
print("Test MSE:", mse_test_top)

# ---------------------------
# Model 3: Linear regression using all numeric features
model_all = LinearRegression()
model_all.fit(X_train, y_train)

y_train_pred_all = model_all.predict(X_train)
y_test_pred_all = model_all.predict(X_test)

mse_train_all = mean_squared_error(y_train, y_train_pred_all)
mse_test_all = mean_squared_error(y_test, y_test_pred_all)
print("\nAll Features Model:")
print("Training MSE:", mse_train_all)
print("Test MSE:", mse_test_all)
