import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Step 1: Load and preprocess data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train an Elastic Net model
alpha = 0.5
l1_ratio = 0.5
elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
elastic_net.fit(X_train, y_train)

# Step 3: Visualize the model's coefficients
coef = pd.Series(elastic_net.coef_, index=X.columns)

plt.figure(figsize=(10, 6))
coef.sort_values().plot(kind='barh')
plt.title('Elastic Net Coefficients')
plt.show()

# Step 4: Plot predicted vs. actual values
y_pred = elastic_net.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()

# Step 5: Evaluate performance metrics over different alpha values
alphas = np.logspace(-4, 0, 50)
mse = []
r2 = []

for alpha in alphas:
    elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    scores = cross_val_score(elastic_net, X, y, cv=5, scoring='neg_mean_squared_error')
    mse.append(-scores.mean())
    elastic_net.fit(X_train, y_train)
    y_pred = elastic_net.predict(X_test)
    r2.append(r2_score(y_test, y_pred))

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(alphas, mse, marker='o', linestyle='--')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Alpha')

plt.subplot(1, 2, 2)
plt.plot(alphas, r2, marker='o', linestyle='--')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R-squared')
plt.title('R-squared vs Alpha')

plt.tight_layout()
plt.show()
