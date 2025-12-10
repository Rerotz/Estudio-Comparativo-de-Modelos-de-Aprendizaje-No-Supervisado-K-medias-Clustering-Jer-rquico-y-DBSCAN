# =========================================
# 4. DBSCAN COMPLETO
# =========================================

eps_values = np.arange(0.2, 1.2, 0.1)
dbscan_results = {}

for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_scaled)
    
    # evitar casos donde DBSCAN genera solo ruido
    if len(set(labels)) <= 1:
        continue
        
    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    dbscan_results[eps] = (sil, dbi, labels)

print("\nResultados de DBSCAN:")
for eps, vals in dbscan_results.items():
    print(f"eps={eps:.1f} → Silhouette={vals[0]:.3f}, DBI={vals[1]:.3f}")

# Mejor eps
best_eps = max(dbscan_results, key=lambda e: dbscan_results[e][0])
print("\nMejor eps:", best_eps)

df["DBSCAN_cluster"] = dbscan_results[best_eps][2]

# Visualización PCA
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="DBSCAN_cluster", palette="tab10")
plt.title(f"DBSCAN con eps = {best_eps}")
plt.show()
# =========================================
# 5. COMPARACIÓN FINAL
# =========================================

def eval_clusters(X, labels):
    return {
        "Clusters": len(set(labels)),
        "Silhouette": silhouette_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels)
    }

print("\n=== COMPARACIÓN FINAL ===")

print("K-MEANS:", eval_clusters(X_scaled, df["KMeans_cluster"]))
print("HIERARCHICAL:", eval_clusters(X_scaled, df["Hier_cluster"]))
print("DBSCAN:", eval_clusters(X_scaled, df["DBSCAN_cluster"]))
