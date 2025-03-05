import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("online_shoppers_intention.csv")

# Identify numerical and categorical features (excluding the target variable "Revenue")
num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if "Revenue" in num_features:
    num_features.remove("Revenue")

cat_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()
if "Revenue" in cat_features:
    cat_features.remove("Revenue")

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# Transform the dataset
X = df.drop(columns=['Revenue'])
X_processed = preprocessor.fit_transform(X)

# Try different k values for K-Means clustering
k_values = range(2, 11)
inertia_values = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_processed)
    
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_processed, cluster_labels))

# Plot Inertia (Elbow Method)
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.title("Elbow Method for Optimal k in K-Means")
plt.grid()
plt.show()

# Plot Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different k in K-Means")
plt.grid()
plt.show()

# Choose best k based on highest silhouette score
best_k = k_values[np.argmax(silhouette_scores)]

# Train K-Means with best k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_processed)
silhouette_kmeans = silhouette_score(X_processed, kmeans_labels)

# Train Agglomerative Clustering (Complete Linkage) with same k
agglo = AgglomerativeClustering(n_clusters=best_k, linkage='complete')
agglo_labels = agglo.fit_predict(X_processed)
silhouette_agglo = silhouette_score(X_processed, agglo_labels)

# Store results in DataFrame
clustering_results = pd.DataFrame({
    "Clustering Method": ["K-Means", "Agglomerative (Complete Linkage)"],
    "Best k": [best_k, best_k],
    "Silhouette Score": [silhouette_kmeans, silhouette_agglo]
})

# Display clustering results
print(clustering_results)
