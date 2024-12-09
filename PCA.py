import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Cargar el dataset balanceado
df_resampled = pd.read_csv('proy_balanceado.csv')

# Seleccionar características (X) y etiqueta (y)
X = df_resampled.drop('categoria_jugador', axis=1)  # Características
y = df_resampled['categoria_jugador']  # Etiqueta

# Lista para almacenar las confiabilidades por cantidad de componentes
accuracies_by_pca = {}

# Componentes principales a probar (12, 10, 11, 9, 5, 3)
components_list = [12, 10, 11, 9, 5, 3]

# Realizar 5 ejecuciones para cada cantidad de componentes principales
for components in components_list:
    accuracies = []
    
    for _ in range(5):  # Realizar 5 ejecuciones por número de componentes
        # Dividir los datos en entrenamiento y prueba (50% entrenamiento, 50% prueba)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)

        # Inicializar PCA con la cantidad de componentes
        pca = PCA(n_components=components)

        # Transformar las características usando PCA
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Inicializar el clasificador Random Forest
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Entrenar el clasificador
        clf.fit(X_train_pca, y_train)

        # Predecir las etiquetas para el conjunto de prueba
        y_pred = clf.predict(X_test_pca)

        # Calcular la confiabilidad (accuracy)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Almacenar las confiabilidades para cada cantidad de componentes
    accuracies_by_pca[components] = accuracies

# Calcular la mediana y media de las confiabilidades para cada cantidad de componentes
for components in components_list:
    median_accuracy = np.median(accuracies_by_pca[components])
    mean_accuracy = np.mean(accuracies_by_pca[components])
    print(f"Componentes principales: {components}")
    print(f"  Mediana de la confiabilidad (accuracy): {median_accuracy * 100:.2f}%")
    print(f"  Media de la confiabilidad (accuracy): {mean_accuracy * 100:.2f}%\n")

# Opcional: Graficar la distribución de las confiabilidades para cada cantidad de componentes
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

for components in components_list:
    plt.hist(accuracies_by_pca[components], bins=10, alpha=0.5, label=f'{components} Componentes')

plt.title('Distribución de las Confiabilidades (Accuracy) con PCA')
plt.xlabel('Accuracy')
plt.ylabel('Frecuencia')
plt.legend(title="Cantidad de Componentes")
plt.show()
