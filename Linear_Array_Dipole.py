import numpy as np
import matplotlib.pyplot as plt

# Constants
lambda_val = 1  # For simplicity, let's normalize to lambda = 1 (you can change this)
spacing_start = 0.1 * lambda_val
spacing_end = 2.0 * lambda_val
num_spacing_points = 100
spacings = np.linspace(spacing_start, spacing_end, num_spacing_points)

num_elements_list = [2, 4, 8, 16]  # Array sizes to analyze

# Function to calculate directivity of a uniform linear array
def array_directivity(n_elements, spacing_lambda):
    """
    Calculates the maximum directivity of a uniform linear array.

    Args:
        n_elements (int): Number of elements in the array.
        spacing_lambda (float): Spacing between elements in wavelengths.

    Returns:
        float: Maximum directivity of the array.
    """
    directivity = n_elements * spacing_lambda
    return directivity

# Calculate directivity for each array size and spacing
directivity_data = {}
for n_elements in num_elements_list:
    directivity_values = [array_directivity(n_elements, d / lambda_val) for d in spacings]
    directivity_data[n_elements] = directivity_values

# Plotting the directivity curves
plt.figure(figsize=(10, 6))
for n_elements, directivity_values in directivity_data.items():
    plt.plot(spacings / lambda_val, directivity_values, label=f'N = {n_elements}')

plt.xlabel('Spacing (λ)')
plt.ylabel('Directivity')
plt.title('Directivity of Uniform Linear Array vs. Spacing')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Summarize the observations
print("\nObservations on Directivity of Uniform Linear Arrays:\n")

print("- Directivity generally increases with the number of elements in the array.")
print("- For a fixed number of elements, directivity varies with element spacing.")
print("- At smaller spacings (around 0.5 lambda), directivity is lower.")
print("- As spacing increases (up to a certain point), directivity tends to increase.")
print("- When spacing exceeds lambda, grating lobes appear, which reduces the maximum directivity and affects the main lobe.")
print("- The optimal spacing for maximum directivity depends on the number of elements.")