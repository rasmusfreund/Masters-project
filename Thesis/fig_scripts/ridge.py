import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

plt.rcParams.update({'font.size': 14})

np.random.seed(0)
n_samples = 10
X = np.linspace(0, 10, n_samples).reshape(-1, 1)
y = 10 * np.sin(X).ravel() + np.random.normal(0, 1, n_samples)

poly = PolynomialFeatures(degree=9)
X_poly = poly.fit_transform(X)

scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

lambdas = np.logspace(-4, 4, 100)

coefficients = []
for alpha in lambdas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_poly_scaled, y)
    coefficients.append(ridge.coef_)
coefficients = np.array(coefficients)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,12))
colors = plt.cm.tab20b(np.linspace(0, 1, X_poly.shape[1]))
for i in range(X_poly.shape[1]):
    ax1.plot(lambdas, coefficients[:, i], label=f'Feature {i+1}', color=colors[i % len(colors)])
ax1.set_xscale('log')
ax1.set_xlabel(r'$\lambda$')
ax1.set_ylabel('Coefficient value')
ax1.set_title('Ridge Regression Coefficients as a Function of Regularization')
ax1.axhline(0, color='black', linestyle='--', linewidth=1)

lambdas_to_plot = [0, 0.1, 1, 10, 100]
X_fit = np.linspace(0, 10, 100).reshape(-1, 1)
X_fit_poly = poly.transform(X_fit)
for idx, alpha in enumerate(lambdas_to_plot):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_poly_scaled, y)
    y_fit = ridge.predict(scaler.transform(X_fit_poly))
    ax2.plot(X_fit, y_fit, label=r'$\lambda$' + f' = {alpha}', color=colors[idx*3 % len(colors)])
ax2.scatter(X, y, color='black', label='Data points')
ax2.set_xlabel('Feature Value')
ax2.set_ylabel('Target Value')
ax2.set_title('Effect of Increasing Regularization on Fitted Line')
ax2.legend()


save_path = "/mnt/c/Users/rasmu/Desktop/Bioinformatics/MSc/Thesis/img/ridge_lambda_effect.png"

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()