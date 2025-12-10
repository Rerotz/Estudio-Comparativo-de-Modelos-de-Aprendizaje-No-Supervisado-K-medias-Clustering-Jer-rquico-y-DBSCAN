# =========================================
# 1. IMPORTACIONES Y PREPROCESAMIENTO
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set(style="whitegrid")

# ===========================
# Cargar dataset
# ===========================
df = pd.read_csv("Mall_Customers.csv")

# Codificar género
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# Variables seleccionadas
X = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reducción PCA para visualización en 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]
