import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sici        # Sine and cosine integrals
from scipy.integrate import quad      # For numerical integration

# Constants
C = 0.5772156649                  # Euler-Mascheroni constant
SPEED_OF_LIGHT = 3e8              # Speed of light (m/s)
FREQUENCY = 300e6                 # Frequency in Hz (arbitrary reference)
WAVELENGTH = SPEED_OF_LIGHT / FREQUENCY  # Wavelength (λ)
FREE_SPACE_IMPEDANCE = 120 * np.pi     # Characteristic impedance of free space
WAVENUMBER = 2 * np.pi / WAVELENGTH    # k = 2π/λ
WIRE_RADIUS = 0.001 * WAVELENGTH       # Thin wire approximation

# Define dipole lengths from 0.1λ to 2.5λ
LENGTH_OVER_WAVELENGTH = np.arange(0.1, 2.51, 0.05)
DIPOLE_LENGTH = LENGTH_OVER_WAVELENGTH * WAVELENGTH

def calculate_impedance(dipole_length, wavenumber, wire_radius, free_space_impedance, euler_constant):
    """
    Calculates the input impedance (R + jX) of a thin-wire dipole using analytical formulas.
    Based on electromagnetic theory involving sine and cosine integrals.
    
    Parameters:
    - dipole_length: Physical length of the dipole
    - wavenumber: k = 2π/λ
    - wire_radius: Radius of dipole conductor
    - free_space_impedance: Z0 = 120π Ω
    - euler_constant: Euler–Mascheroni constant (~0.5772)

    Returns:
    - resistance: Real part of impedance (Ω)
    - reactance: Imaginary part of impedance (Ω)
    """
    kL = wavenumber * dipole_length
    si_kL, ci_kL = sici(kL)
    si_2kL, ci_2kL = sici(2 * kL)
    si_2ka2_L, ci_2ka2_L = sici(2 * wavenumber * wire_radius**2 / dipole_length)

    # Resistance formula derived from EM theory
    resistance = (free_space_impedance / (2 * np.pi)) * (
        euler_constant + np.log(kL) - ci_kL +
        0.5 * np.sin(kL) * (si_2kL - 2 * si_kL) +
        0.5 * np.cos(kL) * (euler_constant + np.log(kL / 2) + ci_2kL - 2 * ci_kL)
    )

    # Reactance formula derived from EM theory
    reactance = (free_space_impedance / (4 * np.pi)) * (
        2 * si_kL +
        np.cos(kL) * (2 * si_kL - si_2kL) -
        np.sin(kL) * (2 * ci_kL - ci_2kL - ci_2ka2_L)
    )

    return resistance, reactance


def antenna_pattern(theta, kL):
    """
    Computes normalized radiation intensity of a dipole at angle theta.
    
    Parameters:
    - theta: Observation angle (radians)
    - kL: Product of wavenumber and dipole length
    
    Returns:
    - Pattern intensity proportional to power density
    """
    sin_theta = np.sin(theta)
    if np.isscalar(theta):
        if abs(sin_theta) < 1e-10:
            return 0.0
        return ((np.cos(kL / 2 * np.cos(theta)) - np.cos(kL / 2)) / sin_theta) ** 2
    else:
        near_zero_mask = np.abs(sin_theta) < 1e-10
        pattern = np.zeros_like(theta, dtype=float)
        kL_broadcast = np.broadcast_to(kL, theta.shape)
        pattern[~near_zero_mask] = ((np.cos(kL_broadcast[~near_zero_mask] / 2 * np.cos(theta[~near_zero_mask])) - np.cos(kL_broadcast[~near_zero_mask] / 2)) / sin_theta[~near_zero_mask]) ** 2
        return pattern


def calculate_directivity(dipole_length, wavenumber):
    """
    Calculates maximum directivity of a dipole by numerically integrating its radiation pattern.
    
    Parameters:
    - dipole_length: Physical length of the dipole
    - wavenumber: k = 2π/λ

    Returns:
    - directivity: Max directivity value
    """
    directivity = np.zeros_like(dipole_length)
    theta_vals = np.linspace(0, np.pi, 1000)

    for i, length in enumerate(dipole_length):
        kL = wavenumber * length
        intensity = antenna_pattern(theta_vals, kL)
        max_intensity = np.max(intensity)
        integral, _ = quad(lambda theta: antenna_pattern(theta, kL) * np.sin(theta), 0, np.pi, epsabs=1e-8)
        directivity[i] = 2 * max_intensity / integral if integral > 0 else 1.5

    return directivity


if __name__ == "__main__":
    # Compute impedance and directivity values
    resistance, reactance = calculate_impedance(DIPOLE_LENGTH, WAVENUMBER, WIRE_RADIUS, FREE_SPACE_IMPEDANCE, C)
    directivity_max = calculate_directivity(DIPOLE_LENGTH, WAVENUMBER)

    # Detect resonant lengths (where reactance ~ 0)
    resonance_threshold = 10  # Ohms
    resonant_indices = np.where(np.abs(reactance) < resonance_threshold)[0]
    resonant_lengths = LENGTH_OVER_WAVELENGTH[resonant_indices]

    print("Resonant Lengths (where X_in = 0):")
    for l in resonant_lengths:
        print(f"  - {l:.2f} λ")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot input impedance (Resistance and Reactance)
    ax1.plot(LENGTH_OVER_WAVELENGTH, resistance, label='Resistance $R_{in}$ (Ω)', color='blue', linewidth=2)
    ax1.plot(LENGTH_OVER_WAVELENGTH, reactance, label='Reactance $X_{in}$ (Ω)', color='red', linestyle='--', linewidth=2)
    ax1.scatter(resonant_lengths, resistance[resonant_indices],
                color='green', s=100, zorder=5, edgecolor='black',
                label=r'Resonant Points ($X_{in} \approx 0$)')
    ax1.set_xlabel('Length $L/\\lambda$')
    ax1.set_ylabel('Impedance (Ω)')
    ax1.set_title('Dipole Input Impedance vs. Length')
    ax1.legend()
    ax1.grid(True)

    # Plot maximum directivity in dB
    ax2.plot(LENGTH_OVER_WAVELENGTH, 10 * np.log10(directivity_max), label='Directivity (dB)', color='green')
    ax2.set_xlabel('Length $L/\\lambda$')
    ax2.set_ylabel('Directivity (dB)')
    ax2.set_title('Dipole Maximum Directivity vs. Length')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()