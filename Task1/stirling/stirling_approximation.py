import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import gamma, factorial
import os

# Set plot style for better visualization
plt.style.use('ggplot')

# Create output directory if it doesn't exist
os.makedirs('../../output/Task1/stirling', exist_ok=True)

# Function to calculate Stirling's approximation
def stirling_approximation(n):
    """
    Stirling's approximation: log(n!) ≈ n*log(n) - n + 0.5*log(2*pi*n)
    """
    return n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n)

# Function to calculate factorial using math.factorial
def exact_log_factorial(n):
    """
    Calculate the exact log(n!) using math.factorial for small n
    or scipy.special.gamma for larger n
    """
    if n <= 170:  # Limit for factorial in most systems
        return np.log(math.factorial(n))
    else:
        return np.log(gamma(n + 1))

# Create a range of N values
n_values = np.arange(1, 11)
n_values_extended = np.linspace(1, 10, 1000)  # For smooth curve

# Calculate exact log(n!) and Stirling's approximation
exact_values = [exact_log_factorial(n) for n in n_values]
stirling_values = [stirling_approximation(n) for n in n_values]
stirling_smooth = [stirling_approximation(n) for n in n_values_extended]
gamma_smooth = [np.log(gamma(n + 1)) for n in n_values_extended]

# Calculate the difference
difference = [stirling - exact for stirling, exact in zip(stirling_values, exact_values)]
difference_smooth = [stirling_approximation(n) - np.log(gamma(n + 1)) for n in n_values_extended]

# Create a 2x1 plot
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Factorial and Stirling's approximation
axs[0].scatter(n_values, exact_values, color='blue', s=100, label='Exact log(n!)')
axs[0].plot(n_values_extended, stirling_smooth, 'r-', linewidth=2, label="Stirling's Approximation")
axs[0].plot(n_values_extended, gamma_smooth, 'g--', linewidth=2, label='log(Γ(n+1))')
axs[0].set_title("Comparison of log(n!) and Stirling's Approximation", fontsize=16)
axs[0].set_xlabel('n', fontsize=14)
axs[0].set_ylabel('log(n!)', fontsize=14)
axs[0].legend(fontsize=12)
axs[0].grid(True)

# Plot 2: Difference between Stirling's approximation and exact factorial
axs[1].plot(n_values_extended, difference_smooth, 'b-', linewidth=2)
axs[1].scatter(n_values, difference, color='red', s=100)
axs[1].set_title("Difference: Stirling's Approximation - log(Γ(n+1))", fontsize=16)
axs[1].set_xlabel('n', fontsize=14)
axs[1].set_ylabel('Difference', fontsize=14)
axs[1].grid(True)

# Add a horizontal line at y=0 for reference
axs[1].axhline(y=0, color='k', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('../../output/Task1/stirling/stirling_approximation.png', dpi=300)
plt.close()

# Now let's verify the Stirling's approximation for larger values
n_large = np.arange(10, 101, 10)
exact_large = [np.log(gamma(n + 1)) for n in n_large]
stirling_large = [stirling_approximation(n) for n in n_large]
relative_error = [(s - e) / e * 100 for s, e in zip(stirling_large, exact_large)]

plt.figure(figsize=(12, 6))
plt.plot(n_large, relative_error, 'bo-', linewidth=2)
plt.title("Relative Error of Stirling's Approximation (%)", fontsize=16)
plt.xlabel('n', fontsize=14)
plt.ylabel('Relative Error (%)', fontsize=14)
plt.grid(True)
plt.savefig('../../output/Task1/stirling/stirling_relative_error.png', dpi=300)
plt.close()

print("Stirling's approximation analysis completed. Results saved in output/Task1/stirling/") 