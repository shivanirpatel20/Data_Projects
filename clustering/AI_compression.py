import numpy as np
from sklearn.cluster import KMeans
import os

#loading the embeddings matrix
#X = np.load("/Users/shivanipatel/Downloads/ISYE6740_Fall_2025_HW1/data/AImodel.npy")

script_path = os.path.dirname(__file__)
X_path = os.path.join(script_path, "data", "AImodel.npy")
X = np.load(X_path)

m, d = X.shape
#doing the normalization
X_norm = X/np.linalg.norm(X,axis=1, keepdims=True)
#running the kmeans
kmeans=KMeans(n_clusters=256,random_state=3190,n_init="auto").fit(X_norm)
#getting the centroids
centroids = kmeans.cluster_centers_
index_val = kmeans.labels_.astype(np.uint8)
Memory_bef = m*d*4
Memory_aft = (256*d*4)+(m*1)
#converting to MB
MB_memory_bef = Memory_bef/(1024**2)
MB_memory_aft = Memory_aft/(1024**2)
print(MB_memory_bef)
print(MB_memory_aft)



#reconstruct the embeddings using centroids
X_norm_hat = centroids[index_val]

numerator = np.einsum("ij, ij->i", X_norm, X_norm_hat)
denomenator = np.linalg.norm(X_norm_hat, axis=1) #prevent a 0 denomenator
cos = numerator/denomenator

avg_cos = np.mean(cos)

print(avg_cos)