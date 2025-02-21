# First Oython on GitHub

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Daten generieren
np.random.seed(0)
n_samples = 100
x1 = np.random.rand(n_samples) * 10
x2 = np.random.rand(n_samples) * 10
y = 2 * x1 + 3 * x2 + np.random.randn(n_samples) * 5

# Regressionsmodell erstellen und trainieren
X = np.column_stack((x1, x2))
model = LinearRegression()
model.fit(X, y)

# Vorhersagen machen
x1_range = np.linspace(min(x1), max(x1), 30)
x2_range = np.linspace(min(x2), max(x2), 30)
X1, X2 = np.meshgrid(x1_range, x2_range)
Y_pred = model.predict(np.column_stack((X1.ravel(), X2.ravel()))).reshape(X1.shape)

# 3D-Streudiagramm erstellen
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Datenpunkte darstellen
ax.scatter(x1, x2, y, c='b', marker='o', label='Datenpunkte')

# Regressionsebene darstellen
ax.plot_surface(X1, X2, Y_pred, alpha=0.5, color='r', label='Regressionsebene')

# Achsenbeschriftungen und Titel hinzufügen
ax.set_xlabel('Variable X1')
ax.set_ylabel('Variable X2')
ax.set_zlabel('Variable Y')
ax.set_title('3D-Streudiagramm mit Regressionsebene')

# Legende hinzufügen
ax.legend()

# Diagramm anzeigen
plt.show()
