import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.optimize import minimize
from scipy.stats import norm

# Set plot style for better visualization
plt.style.use('ggplot')

# Create output directory if it doesn't exist
os.makedirs('../../output/Task2/vacuum', exist_ok=True)

# Load the vacuum decay dataset
with open('Vacuum_decay_dataset.json', 'r') as f:
    vacuum_data = json.load(f)

# Filter data to include only x >= 1 as specified in the problem
vacuum_data = np.array([x for x in vacuum_data if x >= 1])

# Plot histogram of the data
plt.figure(figsize=(12, 8))
hist, bins, _ = plt.hist(vacuum_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
plt.title('Vacuum Decay Distance Distribution', fontsize=16)
plt.xlabel('Distance (x)', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.grid(True)
plt.savefig('../../output/Task2/vacuum/vacuum_decay_histogram.png', dpi=300)
plt.close()

# Define the exponential probability density function
def exponential_pdf(x, lambda_):
    """
    Exponential PDF with normalization for x >= 1
    P(x|λ) = (1/λ) * exp(-x/λ) / Z(λ)
    where Z(λ) = exp(-1/λ)
    """
    return (1/lambda_) * np.exp(-x/lambda_) / np.exp(-1/lambda_)

# Define the negative log-likelihood function for the exponential distribution
def neg_log_likelihood_exp(lambda_, data):
    """
    Negative log-likelihood for the exponential distribution
    """
    if lambda_ <= 0:
        return np.inf
    
    # Calculate log-likelihood
    log_likelihood = np.sum(np.log(exponential_pdf(data, lambda_)))
    
    return -log_likelihood

# Fit the exponential distribution to the data
initial_guess = [np.mean(vacuum_data)]
result = minimize(neg_log_likelihood_exp, initial_guess, args=(vacuum_data,), method='Nelder-Mead')
lambda_fit = result.x[0]

# Calculate the Fisher Information Matrix
def fisher_information(lambda_):
    """
    Fisher Information Matrix for the exponential distribution
    """
    # For exponential distribution, Fisher information is 1/λ²
    return 1 / (lambda_**2)

# Calculate the variance of the estimator
variance = 1 / (fisher_information(lambda_fit) * len(vacuum_data))

# Plot the fitted exponential distribution
x_values = np.linspace(min(vacuum_data), max(vacuum_data), 1000)
y_values = exponential_pdf(x_values, lambda_fit)

plt.figure(figsize=(12, 8))
plt.hist(vacuum_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True, label='Data')
plt.plot(x_values, y_values, 'r-', linewidth=2, label=f'Exponential Fit (λ = {lambda_fit:.4f})')
plt.title('Vacuum Decay: Data and Exponential Fit', fontsize=16)
plt.xlabel('Distance (x)', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('../../output/Task2/vacuum/vacuum_decay_exponential_fit.png', dpi=300)
plt.close()

# Calculate the confidence interval (95%)
confidence_level = 0.95
z_score = norm.ppf((1 + confidence_level) / 2)
margin_of_error = z_score * np.sqrt(variance)
confidence_interval = (lambda_fit - margin_of_error, lambda_fit + margin_of_error)

# Create a summary plot with results
plt.figure(figsize=(12, 8))
plt.hist(vacuum_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True, label='Data')
plt.plot(x_values, y_values, 'r-', linewidth=2, label=f'Exponential Fit (λ = {lambda_fit:.4f})')

# Add text box with results
textstr = '\n'.join((
    f'Decay Constant (λ): {lambda_fit:.4f}',
    f'Variance: {variance:.6f}',
    f'95% Confidence Interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})',
    f'Number of Data Points: {len(vacuum_data)}'
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.title('Vacuum Decay Analysis Results', fontsize=16)
plt.xlabel('Distance (x)', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('../../output/Task2/vacuum/vacuum_decay_results.png', dpi=300)
plt.close()

# Save the results to a text file
with open('../../output/Task2/vacuum/vacuum_decay_results.txt', 'w') as f:
    f.write(f"Vacuum Decay Analysis Results\n")
    f.write(f"============================\n\n")
    f.write(f"Decay Constant (λ): {lambda_fit:.6f}\n")
    f.write(f"Variance: {variance:.8f}\n")
    f.write(f"95% Confidence Interval: ({confidence_interval[0]:.6f}, {confidence_interval[1]:.6f})\n")
    f.write(f"Number of Data Points: {len(vacuum_data)}\n")
    f.write(f"Fisher Information: {fisher_information(lambda_fit):.6f}\n")
    f.write(f"Negative Log-Likelihood: {neg_log_likelihood_exp(lambda_fit, vacuum_data):.6f}\n")

print("Vacuum decay analysis completed. Results saved in output/Task2/vacuum/") 