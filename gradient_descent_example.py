import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return x**2 + 2  # Simple quadratic function f(x) = x^2 + 2

def gradient(x):
    return 2*x  # Derivative of x^2 + 2

# Gradient descent parameters
learning_rate = 0.1
num_iterations = 50
x = 10.0  # Starting point

# Lists to store values for plotting
x_history = [x]
y_history = [objective_function(x)]

# Gradient descent loop
for i in range(num_iterations):
    grad = gradient(x)
    x = x - learning_rate * grad
    x_history.append(x)
    y_history.append(objective_function(x))

# Plot the results
x_range = np.linspace(-10, 10, 100)
y_range = objective_function(x_range)

plt.plot(x_range, y_range, 'b-', label='Objective function')
plt.plot(x_history, y_history, 'ro-', label='Gradient descent path')
plt.title('Gradient Descent Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Minimum found at x = {x:.4f}, f(x) = {objective_function(x):.4f}") 