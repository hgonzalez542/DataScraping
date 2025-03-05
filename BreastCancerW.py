import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# Extract features and target
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets.replace({'M': 1, 'B': 0})  # Encode as 1/0

# Split data (20% test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------- Model 1: Logistic Regression ---------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Predictions
y_train_pred_log = log_model.predict(X_train)
y_test_pred_log = log_model.predict(X_test)

# Accuracy
train_acc_log = accuracy_score(y_train, y_train_pred_log)
test_acc_log = accuracy_score(y_test, y_test_pred_log)

print("\nLogistic Regression:")
print(f"Training Accuracy: {train_acc_log:.4f}")
print(f"Test Accuracy: {test_acc_log:.4f}")
print(classification_report(y_test, y_test_pred_log))

# --------- Model 2: Overfitting Decision Tree ---------
tree_model = DecisionTreeClassifier(max_depth=None)  # No depth limit to increase overfitting
tree_model.fit(X_train, y_train)

# Predictions
y_train_pred_tree = tree_model.predict(X_train)
y_test_pred_tree = tree_model.predict(X_test)

# Accuracy
train_acc_tree = accuracy_score(y_train, y_train_pred_tree)
test_acc_tree = accuracy_score(y_test, y_test_pred_tree)

print("\nOverfitting Decision Tree:")
print(f"Training Accuracy: {train_acc_tree:.4f}")
print(f"Test Accuracy: {test_acc_tree:.4f}")
print(classification_report(y_test, y_test_pred_tree))

# --------- Experiment: Run multiple times for biggest overfitting gap ---------
best_overfit_gap = 0
best_train_acc, best_test_acc = 0, 0

for _ in range(10):  # Run multiple times to find max overfitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    overfit_tree = DecisionTreeClassifier(max_depth=None)
    overfit_tree.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, overfit_tree.predict(X_train))
    test_acc = accuracy_score(y_test, overfit_tree.predict(X_test))
    
    overfit_gap = train_acc - test_acc
    if overfit_gap > best_overfit_gap:
        best_overfit_gap = overfit_gap
        best_train_acc, best_test_acc = train_acc, test_acc

print("\nBiggest Overfitting Found:")
print(f"Training Accuracy: {best_train_acc:.4f}")
print(f"Test Accuracy: {best_test_acc:.4f}")
print(f"Overfitting Gap: {best_overfit_gap:.4f}")
