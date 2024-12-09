import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar el dataset (sin la columna de la clase 'categoria_jugador')
df_resampled = pd.read_csv('proy_balanceado.csv')

# Seleccionar solo las características (sin la columna 'categoria_jugador')
X = df_resampled.drop('categoria_jugador', axis=1)

# **1. Aplicar PCA para reducir la dimensionalidad**
# Inicializar PCA para reducir a 2 componentes principales (para visualización)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Graficar los datos en 2D después de la reducción con PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, c='blue', label="Datos")
plt.title('Reducción de dimensionalidad con PCA (2 Componentes)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# **2. Aplicar K-means para encontrar clusters**
# Definir el número de clusters
kmeans = KMeans(n_clusters=3, random_state=42)  # Probaremos con 3 clusters
kmeans.fit(X)

# Obtener los centros de los clusters y las etiquetas
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Graficar los clusters después de aplicar K-means
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label="Centroides")
plt.title('Clusters identificados por K-means')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.show()

# **3. Mostrar los resultados del clustering**
print("Etiquetas de los clusters:")
print(labels[:10])  # Imprimir las primeras 10 etiquetas de cluster

print("Centroides de los clusters:")
print(centroids)
