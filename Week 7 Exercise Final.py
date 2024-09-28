# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as z
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load your wine dataset from the specified location
file_path = r'C:\PhD\4. TIM-8555\Week 7\wine-clustering.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to ensure it loaded correctly
print(data.head())

# Standardize the data before applying PCA
scaler = z()
X_scaled = scaler.fit_transform(data)

# Apply PCA to capture 80% of the variance
pca = PCA(n_components=0.80)  # Keep enough components to explain 80% of the variance
X_pca = pca.fit_transform(X_scaled)

# Check the explained variance ratio
print(f'Explained Variance Ratio by each component: {pca.explained_variance_ratio_}')
print(f'Total Variance explained: {np.sum(pca.explained_variance_ratio_)}')

# K-Means clustering
# Let's evaluate different values for k (e.g., 3, 4, ...)
k_values = [3, 4, 5]
inertia = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)
    print(f'For k={k}, Inertia: {kmeans.inertia_}')
    print(f'Cluster Centers for k={k}:\n{kmeans.cluster_centers_}')
    print(f'Labels for k={k}: {kmeans.labels_}\n')

# Plot inertia to determine the optimal k
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Hierarchical clustering
# Using Ward's method and Euclidean distance
linkage_matrix = linkage(X_pca, method='ward', metric='euclidean')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()
