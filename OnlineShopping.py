import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("online_shoppers_intention.csv")

# Define features and target
X = df.drop(columns=['Revenue'])
y = df['Revenue']

# Identify numerical and categorical features
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# Preprocessing ~ This is some amazing practice by the way. this assignment reallt reinforced it all for me
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Finding optimal k for kNN
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=k))
    ])
    knn_pipeline.fit(X_train, y_train)
    y_pred = knn_pipeline.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot k vs. accuracy
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("kNN: Choosing the Optimal k")
plt.xticks(k_values)
plt.grid()
plt.show()

#Not sure if there was a more efficient way to do this. Please comment on it if there is 
# Choose the best k
best_k = k_values[np.argmax(accuracies)]

# Train kNN with best k
knn_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=best_k))
])
knn_pipeline.fit(X_train, y_train)
y_pred_knn = knn_pipeline.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Train Decision Tree classifier
dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])
dt_pipeline.fit(X_train, y_train)
y_pred_dt = dt_pipeline.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# Train Random Forest classifier
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Display results
accuracy_results = pd.DataFrame({
    "Model": ["kNN (best k)", "Decision Tree", "Random Forest"],
    "Accuracy": [accuracy_knn, accuracy_dt, accuracy_rf]
})

print(accuracy_results)