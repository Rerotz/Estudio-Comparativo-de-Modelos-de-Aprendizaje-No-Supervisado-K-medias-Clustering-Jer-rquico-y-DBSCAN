# =========================================
# 2. K-MEANS COMPLETO
# =========================================

wcss = []
silhouette_vals = []
db_vals = []

K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    wcss.append(kmeans.inertia_)  # inercia para método del codo
    silhouette_vals.append(silhouette_score(X_scaled, labels))
    db_vals.append(davies_bouldin_score(X_scaled, labels))

# ----- Plots -----
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(K_range, wcss, marker="o")
plt.title("Método del Codo")
plt.xlabel("k")
plt.ylabel("Inercia")

plt.subplot(1,3,2)
plt.plot(K_range, silhouette_vals, marker="o", color="green")
plt.title("Silhouette por k")
plt.xlabel("k")
plt.ylabel("Silhouette")

plt.subplot(1,3,3)
plt.plot(K_range, db_vals, marker="o", color="red")
plt.title("Davies-Bouldin por k")
plt.xlabel("k")
plt.ylabel("DB Index")

plt.tight_layout()
plt.show()

# ---- Entrenar con el mejor k (usualmente 5 en Mall Data) ----

k_opt = np.argmax(silhouette_vals) + 2
print("Mejor K según silhouette:", k_opt)

kmeans = KMeans(n_clusters=k_opt, random_state=42)
df["KMeans_cluster"] = kmeans.fit_predict(X_scaled)

# Visualización PCA
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="KMeans_cluster", palette="viridis")
plt.title(f"Clustering K-Means con k = {k_opt}")
plt.show()
