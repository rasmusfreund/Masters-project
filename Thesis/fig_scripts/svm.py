import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

plt.rcParams.update({'font.size': 14})

# Generate synthetic data for linear separable case
X, y = make_blobs(n_samples=100, centers=2, random_state=6)

# Train SVM with linear kernel
svc_linear = SVC(kernel='linear')
svc_linear.fit(X, y)

# Plotting decision boundary
def plot_svm_decision_boundary(clf, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Define color maps
    cmap_light = ListedColormap(['#dedc9c', '#398184'])
    cmap_bold = ListedColormap(['#9c9ede', '#843c39'])
    
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s=30, edgecolor='k')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_title(title)

fig, ax = plt.subplots()
plot_svm_decision_boundary(svc_linear, X, y, ax, 'Linear Kernel SVM')

save_path = "/mnt/c/Users/rasmu/Desktop/Bioinformatics/MSc/Thesis/img/svm_linear.png"
plt.savefig(save_path, dpi=300)
plt.show()

# Generate synthetic data for non-linear separable case
X, y = make_circles(n_samples=100, factor=0.5, noise=0.1)

# Train SVM with RBF kernel
svc_rbf = SVC(kernel='rbf', gamma='auto')
svc_rbf.fit(X, y)

# Plotting decision boundary
fig, ax = plt.subplots()
plot_svm_decision_boundary(svc_rbf, X, y, ax, 'RBF Kernel SVM')
save_path = "/mnt/c/Users/rasmu/Desktop/Bioinformatics/MSc/Thesis/img/svm_nonlinear.png"
plt.savefig(save_path, dpi=300)
plt.show()

