import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgb

plt.rcParams.update({'font.size': 14})

# Generate synthetic classification data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=16)

# Train a simple decision tree
clf = DecisionTreeClassifier(max_depth=3, random_state=0)
clf.fit(X, y)

# Define the plotting function for recursive binary splitting
def plot_decision_boundary(clf, X, y, ax):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Define the color maps for the classes
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='k', marker='o')
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    ax.set_title('Recursive Binary Splitting')

    ax.axhline(y=0.26, color='k', linestyle='--')
    ax.axvline(x=1.459, ymin=0.0, ymax=0.5325, color='k', linestyle='--')
    ax.axvline(x=-1.043, ymin=0.5325, ymax=1, color='k', linestyle='--')

# Plot the decision boundary and the decision tree
fig, axes = plt.subplots(1, 2, figsize=(16, 8))


# Plot the decision tree
plot_tree(clf, filled=True, feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'], ax=axes[0], fontsize=13)
axes[0].set_title('Decision Tree Visualization')
axes[0].annotate('A', xy=(0, 1), xycoords='axes fraction', ha='center', fontsize=20, fontweight='bold')


# Plot the recursive binary splitting
plot_decision_boundary(clf, X, y, axes[1])
axes[1].annotate('B', xy=(-0.05, 1), xycoords='axes fraction', ha='center', fontsize=20, fontweight='bold')


save_path = "/mnt/c/Users/rasmu/Desktop/Bioinformatics/MSc/Thesis/img/decision_tree.png"

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()
