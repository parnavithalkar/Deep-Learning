import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.3])

# Mean Squared Error (MSE) Loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Binary Cross-Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Small constant to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Calculate losses
mse = mse_loss(y_true, y_pred)
bce = binary_cross_entropy(y_true, y_pred)

print("Mean Squared Error Loss:", mse)
print("Binary Cross-Entropy Loss:", bce)

# Visualize different loss functions
y_true_sample = 1
y_pred_range = np.linspace(0, 1, 100)

# Calculate losses for different predictions
mse_values = [(y_true_sample - y_pred) ** 2 for y_pred in y_pred_range]
bce_values = [-y_true_sample * np.log(max(y_pred, 1e-15)) - 
              (1 - y_true_sample) * np.log(max(1 - y_pred, 1e-15)) 
              for y_pred in y_pred_range]

# Plot the loss functions
plt.figure(figsize=(10, 5))
plt.plot(y_pred_range, mse_values, label='MSE Loss')
plt.plot(y_pred_range, bce_values, label='BCE Loss')
plt.title('Loss Functions Comparison')
plt.xlabel('Predicted Value')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show() 