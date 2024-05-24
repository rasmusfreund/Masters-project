import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap

# Generate non-linearly seperable dataset
X, y = make_circles(n_samples=200, factor=0.5, noise=0.1, random_state=0)

# Define colorblind-friendly colors
cmap = ListedColormap(['#9c9ede', '#843c39'])

# Function to plot decision boundaries
def plot_decision_boundary(ax, model, X, y, iteration, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, alpha=0.4, cmap=cmap)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k')
    ax.set_title(f'{title}\nIteration {iteration-1}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12,12))

# Perceptron model
perceptron = Perceptron(max_iter=1, tol=None)
iterations = [2, 1001]
for i, iteration in enumerate(iterations):
    perceptron.max_iter = iteration
    perceptron.fit(X, y)
    plot_decision_boundary(axes[0, i], perceptron, X, y, iteration, 'Perceptron')

# MLP model
mlp = MLPClassifier(hidden_layer_sizes=(10, ), max_iter=1, solver='sgd', learning_rate_init=0.1)
for i, iteration in enumerate(iterations):
    mlp.max_iter = iteration
    mlp.fit(X, y)
    plot_decision_boundary(axes[1, i], mlp, X, y, iteration, 'MLP')

plt.tight_layout()
plt.show()
