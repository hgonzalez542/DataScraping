import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Worked on creating a much more fluid comment system for my code as well. So it looks more aesthically pleasing

# Fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# Extract features and target
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets.replace({'M': 1, 'B': 0})  # Encode as 1/0

# Split data (20% test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------- Cross-Validation for Logistic Regression ---------
log_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

log_cv_scores = cross_val_score(log_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nLogistic Regression Cross-Validation Accuracy: {log_cv_scores.mean():.4f} ± {log_cv_scores.std():.4f}")

# --------- Cross-Validation for Decision Tree ---------
tree_pipeline = Pipeline([
    ('classifier', DecisionTreeClassifier(max_depth=None))
])

tree_cv_scores = cross_val_score(tree_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nDecision Tree Cross-Validation Accuracy: {tree_cv_scores.mean():.4f} ± {tree_cv_scores.std():.4f}")

# --------- Grid Search for Logistic Regression ---------
log_param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]  # Regularization strength
}

log_grid_search = GridSearchCV(log_pipeline, log_param_grid, cv=5, scoring='accuracy')
log_grid_search.fit(X_train, y_train)

print("\nBest Logistic Regression Parameters:", log_grid_search.best_params_)
print(f"Best Logistic Regression CV Accuracy: {log_grid_search.best_score_:.4f}")


# --------- Grid Search for Decision Tree ---------
tree_param_grid = {
    'classifier__max_depth': [3, 5, 10, None],  # Control overfitting
    'classifier__min_samples_split': [2, 5, 10]  # Control splitting criteria
}

tree_grid_search = GridSearchCV(tree_pipeline, tree_param_grid, cv=5, scoring='accuracy')
tree_grid_search.fit(X_train, y_train)

print("\nBest Decision Tree Parameters:", tree_grid_search.best_params_)
print(f"Best Decision Tree CV Accuracy: {tree_grid_search.best_score_:.4f}")

# --------- Final Model Evaluation ---------
best_log_model = log_grid_search.best_estimator_
best_tree_model = tree_grid_search.best_estimator_

y_test_pred_log = best_log_model.predict(X_test)
y_test_pred_tree = best_tree_model.predict(X_test)

test_acc_log = accuracy_score(y_test, y_test_pred_log)
test_acc_tree = accuracy_score(y_test, y_test_pred_tree)

print("\nFinal Test Accuracies:")
print(f"Logistic Regression: {test_acc_log:.4f}")
print(f"Decision Tree: {test_acc_tree:.4f}")
