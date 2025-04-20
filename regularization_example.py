import numpy as np
from sklearn.linear_model import Lasso, Ridge
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 3 * X.squeeze() + np.random.randn(100) * 0.5

# Create models with different regularization
lasso = Lasso(alpha=1.0)  # L1 regularization
ridge = Ridge(alpha=1.0)  # L2 regularization

# Fit models
lasso.fit(X, y)
ridge.fit(X, y)

# Generate points for plotting
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
y_lasso = lasso.predict(X_test)
y_ridge = ridge.predict(X_test)

# Plot results
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
plt.plot(X_test, y_lasso, color='red', label='Lasso (L1)')
plt.plot(X_test, y_ridge, color='green', label='Ridge (L2)')
plt.title('L1 vs L2 Regularization')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

print("Lasso coefficient:", lasso.coef_[0])
print("Ridge coefficient:", ridge.coef_[0]) 