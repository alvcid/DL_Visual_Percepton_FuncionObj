from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Leer datos
iris_dataset = load_iris()

print(iris_dataset.target_names)
print(iris_dataset.data)

df = pd.DataFrame(np.c_[iris_dataset["data"], iris_dataset["target"]], # concatenamos los datos con np.c_
                  columns=iris_dataset["feature_names"] + ["target"])  # concatenamos los nombres de las columnas

print(df)

# Visualizamos los datos
fig = plt.figure(figsize=(10, 7))
plt.scatter(df["petal length (cm)"][df["target"] == 0],
            df["petal width (cm)"][df["target"] == 0], c="b", label="Setosa")

plt.scatter(df["petal length (cm)"][df["target"] == 1],
            df["petal width (cm)"][df["target"] == 1], c="g", label="versicolor")

plt.xlabel("petal_length", fontsize=14)
plt.ylabel("petal_width", fontsize=14)
plt.legend(loc="lower right")

plt.show()

# Representación gráfica de tres dimensiones del conjunto de datos
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

ax.scatter3D(df["petal length (cm)"][df["target"] == 0],
             df["petal width (cm)"][df["target"] == 0],
             df["sepal width (cm)"][df["target"] == 0], c="b")

ax.scatter3D(df["petal length (cm)"][df["target"] == 1],
             df["petal width (cm)"][df["target"] == 1],
             df["sepal width (cm)"][df["target"] == 1], c="r")

ax.scatter3D(df["petal length (cm)"][df["target"] == 2],
             df["petal width (cm)"][df["target"] == 2],
             df["sepal width (cm)"][df["target"] == 2], c="g")

ax.set_xlabel("petal_length")
ax.set_ylabel("petal width")
ax.set_zlabel("sepal width")

plt.show()

# Entrenamiento del algoritmo Perceptrón simple

# Reducimos el dataset para poder entrenar y visualizar el resultado
df_reduced = df[["petal length (cm)", "petal width (cm)", "target"]]
df_reduced = df_reduced.loc[df_reduced["target"].isin([0, 1])]
df_reduced

# Separamos del conjunto de datos el target
X_df = df[["petal length (cm)", "petal width (cm)"]]
y_df = df["target"]

# Visualizamos el conjunto de entrenamiento reducido
X_df.plot.scatter("petal length (cm)", "petal width (cm)")
plt.show()

# print(y_df)

# Entrenamos con Perceptron
clf = Perceptron(max_iter=1000, random_state=42)
clf.fit(X_df, y_df)

# Visualizamos el límite de decisión construido por el algoritmo

# Parámetros del modelo
print(clf.coef_)

# Termino de intercepción
print(clf.intercept_)

# Representación gráfica del límite de decisión
X = X_df.values

mins = X.min(axis=0) - 0.1
maxs = X.max(axis=0) + 0.1

xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], 1000),
                     np.linspace(mins[1], maxs[1], 1000))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig = plt.figure(figsize=(10, 7))

plt.contourf(xx, yy, Z, cmap="Set3")
plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidth=1, colors="k")

plt.plot(X[:, 0][y_df==0], X[:, 1][y_df==0], "bs", label="setosa")
plt.plot(X[:, 0][y_df==1], X[:, 1][y_df==1], "go", label="versicolor")
plt.plot(X[:, 0][y_df==2], X[:, 1][y_df==2], "r*", label="virginica")

plt.xlabel("petal_length", fontsize=14)
plt.ylabel("petal_width", fontsize=14)
plt.legend(loc="lower right", fontsize=14)

plt.show()

# Entrenamiento del perceptrón multicapa
clf = MLPClassifier()
clf.fit(X_df, y_df)

print(clf.n_layers_)
print(clf.hidden_layer_sizes)
print(clf.n_outputs_)
print(clf.coefs_[1].shape)
print(clf.intercepts_[1])

# Representación gráfica del límite de decisión
X = X_df.values

mins = X.min(axis=0) - 0.1
maxs = X.max(axis=0) + 0.1

xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], 1000),
                     np.linspace(mins[1], maxs[1], 1000))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig = plt.figure(figsize=(10, 7))

plt.contourf(xx, yy, Z, cmap="Set3")
plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidth=1, colors="k")

plt.plot(X[:, 0][y_df==0], X[:, 1][y_df==0], "bs", label="setosa")
plt.plot(X[:, 0][y_df==1], X[:, 1][y_df==1], "go", label="versicolor")
plt.plot(X[:, 0][y_df==2], X[:, 1][y_df==2], "r*", label="virginica")

plt.title("Perceptrón Multicapa Límite de Decisión")
plt.xlabel("petal_length", fontsize=14)
plt.ylabel("petal_width", fontsize=14)
plt.legend(loc="lower right", fontsize=14)

plt.show()
