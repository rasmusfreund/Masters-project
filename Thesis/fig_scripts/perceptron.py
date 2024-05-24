import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

plt.rcParams.update({'font.size': 14})

# Generate a simple dataset for the Perceptron
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

# Perceptron algorithm visualization
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Initial weights and bias
w = np.zeros(2)
b = 0
learning_rate = 0.1

# Function to plot decision boundary
def plot_decision_boundary(ax, w, b, iteration):
    x_points = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
    if w[1] != 0:
        y_points = -(w[0] * x_points + b) / w[1]
        ax.plot(x_points, y_points, color="#87474b", linestyle='-')
    else:
        ax.axvline(x=-b / w[0], color='#87474b', linestyle='-')
    # Define colors from tab20b colormap
    cmap = ListedColormap(['#9c9ede', '#843c39'])
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k')
    ax.set_title(f'Iteration {iteration}')
    ax.set_xlim(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1)
    ax.set_ylim(np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1)

# Train the Perceptron and plot updates
iterations = [1, 3, 5, 10]
iteration_counter = 0
for _ in range(10):
    for i in range(X.shape[0]):
        if y[i] * (np.dot(w, X[i]) + b) <= 0:
            w += learning_rate * y[i] * X[i]
            b += learning_rate * y[i]
            iteration_counter += 1
            if iteration_counter in iterations:
                plot_decision_boundary(axes.flat[iterations.index(iteration_counter)], w, b, iteration_counter)

plt.tight_layout()
save_path = "/mnt/c/Users/rasmu/Desktop/Bioinformatics/MSc/Thesis/img/perceptron.png"
plt.savefig(save_path, dpi=300)
plt.show()

