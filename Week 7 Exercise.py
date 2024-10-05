import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# Step 1: Load the dataset
file_path = r'C:\PhD\4. TIM-8555\Week 7\wine-clustering.csv'
df = pd.read_csv(file_path)

# Step 2: Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 3: Conduct PCA to retain 80% of the variance
pca = PCA(n_components=0.80)  # This will select the number of components to explain 80% of variance
pca_data = pca.fit_transform(scaled_data)

# Output explained variance ratio for each component
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by each component: {explained_variance}')
print(f'Total variance explained by selected components: {np.sum(explained_variance)}')

# Step 4: Apply K-means clustering
# Test different values for k using the elbow method
sse = []
k_values = range(2, 6)  # Test k from 2 to 5
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_data)
    sse.append(kmeans.inertia_)

# Plot elbow method
plt.figure(figsize=(6, 4))
plt.plot(k_values, sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal k')
plt.show()

# Step 5: Perform K-means clustering with optimal k (choose based on elbow plot)
optimal_k = 3  # Assume 3 is optimal from the elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(pca_data)
df['KMeans_Labels'] = kmeans_labels

# Step 6: Perform Hierarchical Clustering
# Compute linkage matrix for hierarchical clustering
linked = linkage(pca_data, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrogram for Hierarchical Clustering')
plt.show()

# Step 7: Interpret results (printed)
#print("KMeans Cluster Labels:")
#print(df['KMeans_Labels'].value_counts())
