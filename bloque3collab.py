# =========================================
# 3. CLUSTERING JERÁRQUICO
# =========================================

link_methods = ["ward", "complete", "average", "single"]
results_hier = {}

for method in link_methods:
    hier = AgglomerativeClustering(n_clusters=5, linkage=method)
    labels = hier.fit_predict(X_scaled)
    
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    results_hier[method] = (sil, db)

print("Resultados por método de enlace:")
for m, r in results_hier.items():
    print(f"{m} → Silhouette={r[0]:.3f}, Davies-Bouldin={r[1]:.3f}")

# Seleccionar el mejor método
best_method = max(results_hier, key=lambda m: results_hier[m][0])
print("\nMejor linkage según silhouette:", best_method)

# Entrenar final
hier_final = AgglomerativeClustering(n_clusters=5, linkage=best_method)
df["Hier_cluster"] = hier_final.fit_predict(X_scaled)

# ---- Dendrograma ----
plt.figure(figsize=(12,5))
dendrogram(linkage(X_scaled, method="ward"))
plt.title("Dendrograma del Dataset")
plt.show()

# ---- Visualización PCA ----
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="Hier_cluster", palette="cubehelix")
plt.title(f"Clustering Jerárquico ({best_method})")
plt.show()
