import numpy as np
import matplotlib.pyplot as plt
from scipy.special import chebyt

# Try to set an interactive backend for displaying plots
try:
    plt.switch_backend('TkAgg')  # Use TkAgg for interactive rendering
except ImportError:
    print("Warning: TkAgg backend unavailable. Install Tkinter (e.g., pip install tk). Falling back to default backend.")

# Constants
N = 10  # Number of elements
lambda_val = 1.0  # Normalized wavelength (m)
d = 0.5 * lambda_val  # Element spacing (m)
k = 2 * np.pi / lambda_val  # Wave number
theta = np.linspace(0, np.pi, 1000)  # Angles in radians
psi = k * d * np.cos(theta)  # Phase term

# Side-lobe levels in dB (convert to linear scale)
sll_dB = [20, 30, 40]  # Side-lobe levels in dB
R = [10**(sll / 20) for sll in sll_dB]  # Linear scale (R = 10^(SLL/20))

# Function to compute Dolph-Chebyshev weights
def dolph_chebyshev_weights(N, R):
    """
    Compute Dolph-Chebyshev weights for a given side-lobe level.
    Args:
        N: Number of elements
        R: Side-lobe amplitude ratio (linear)
    Returns:
        weights: Array of weights
    """
    # Compute z0 for Chebyshev polynomial
    z0 = (R + np.sqrt(R**2 - 1))**(1/(N-1))
    # Generate Chebyshev polynomial of degree N-1
    T = chebyt(N-1)
    # Evaluate polynomial at points to get weights
    m = np.arange(N)
    weights = np.zeros(N, dtype=complex)
    for n in range(N):
        # Sum over m for each element
        sum_term = 0
        for m_idx in range(N):
            theta_m = (2 * m_idx + 1) * np.pi / (2 * N)
            sum_term += T(np.cos(theta_m)) * np.cos((n - (N-1)/2) * theta_m)
        weights[n] = sum_term / N
    # Normalize weights
    weights /= np.max(np.abs(weights))
    return weights.real

# Function to compute array factor
def array_factor(weights, psi):
    """
    Compute array factor for given weights.
    Args:
        weights: Array weights
        psi: Phase term (k * d * cos(theta))
    Returns:
        af: Array factor
    """
    N = len(weights)
    af = np.zeros(len(psi), dtype=complex)
    for n in range(N):
        af += weights[n] * np.exp(1j * n * psi)
    return np.abs(af)

# Compute weights and array factors
array_factors = {}
weights_dict = {}

# Dolph-Chebyshev for each side-lobe level
for sll, R_val in zip(sll_dB, R):
    weights = dolph_chebyshev_weights(N, R_val)
    weights_dict[f'Chebyshev_{sll}dB'] = weights
    array_factors[f'Chebyshev_{sll}dB'] = array_factor(weights, psi)

# Uniform taper
weights_uniform = np.ones(N)
weights_dict['Uniform'] = weights_uniform
array_factors['Uniform'] = array_factor(weights_uniform, psi)

# Normalize array factors
for key in array_factors:
    array_factors[key] /= np.max(array_factors[key])

# Compute side-lobe levels and beamwidths
results = {}
for key, af in array_factors.items():
    # Convert to dB
    af_dB = 20 * np.log10(af + 1e-10)  # Add small value to avoid log(0)
    # Find main lobe peak
    max_dB = np.max(af_dB)
    # Find side-lobe level (max level outside main lobe)
    main_lobe_idx = np.argmax(af_dB)
    # Consider regions away from main lobe (e.g., > 20 degrees from 90°)
    theta_deg = 180 * theta / np.pi
    side_lobe_region = (theta_deg < 70) | (theta_deg > 110)
    sll_dB_measured = np.max(af_dB[side_lobe_region])
    # Compute half-power beamwidth
    half_power_dB = max_dB - 3  # 3 dB down from peak
    # Find indices where AF is above half-power level
    above_half_power = af_dB >= half_power_dB
    # Find angles corresponding to half-power points
    half_power_indices = np.where(above_half_power)[0]
    if len(half_power_indices) > 1:
        theta1 = theta[half_power_indices[0]]
        theta2 = theta[half_power_indices[-1]]
        hpbw = 180 * abs(theta2 - theta1) / np.pi  # Convert to degrees
    else:
        hpbw = np.nan
    results[key] = {'SLL_dB': sll_dB_measured, 'HPBW_deg': hpbw}

# Plotting the array factors
plt.figure(figsize=(12, 7))
for key, af in array_factors.items():
    plt.plot(180 * theta / np.pi, 20 * np.log10(af + 1e-10), label=key, linewidth=2)
plt.xlabel('Angle (degrees)')
plt.ylabel('Array Factor (dB)')
plt.title('Dolph-Chebyshev vs. Uniform Array Factor (N=10, d=0.5 lambda)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.ylim(-60, 0)  # Limit y-axis for clarity

# Save plot
plt.savefig('dolph_chebyshev_array_factor.png')

# Display plot
plt.show(block=True)
