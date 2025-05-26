import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sici  # Sine and cosine integrals

# Constants
C = 0.5772156649  # Euler-Mascheroni constant
WIRE_RADIUS = 1e-5  # Wire radius in wavelengths

# Define dipole lengths from 0.1λ to 2.5λ
LENGTH_OVER_WAVELENGTH = np.linspace(0.1, 2.5, 1000)

def calculate_impedance(L_over_lam, a_over_lam):
    """
    Calculates the input impedance (Rr, Rin, Xin) of a thin-wire dipole.
    Matches the formulas from the first code.
    """
    kL = 2 * np.pi * L_over_lam
    si_kL, ci_kL = sici(kL)
    si_2kL, ci_2kL = sici(2 * kL)
    si_4kL, ci_4kL = sici(4 * kL)
    ci_2ka2_L = sici(2 * kL * (a_over_lam / L_over_lam)**2)[1]  # Only need Ci

    # Radiation resistance (Rr)
    term1 = C + np.log(kL) - ci_kL
    term2 = 0.5 * np.sin(kL) * (si_2kL - 2 * si_kL)
    term3 = 0.5 * np.cos(kL) * (C + np.log(kL / 2) + ci_2kL - 2 * ci_kL)
    Rr = 60 * (term1 + term2 + term3)
    # Input resistance (Rin)
    Rin = Rr / (np.sin(kL / 2)**2) if np.sin(kL / 2) != 0 else Rr
    # Reactance (Xin)
    term_x1 = 2 * si_kL
    term_x2 = np.cos(kL) * (2 * si_kL - si_2kL)
    term_x3 = np.sin(kL) * (2 * ci_kL - ci_2kL - ci_2ka2_L)
    Xin = 30 * (term_x1 + term_x2 - term_x3)

    return Rr, Rin, Xin

def antenna_pattern(theta, kL):
    """
    Computes normalized radiation intensity of a dipole at angle theta.
    Matches the first code's pattern function.
    """
    sin_theta = np.sin(theta)
    if np.isscalar(theta):
        if abs(sin_theta) < 1e-10:
            return 0.0
        return ((np.cos(kL / 2 * np.cos(theta)) - np.cos(kL / 2)) / sin_theta) ** 2
    else:
        near_zero_mask = np.abs(sin_theta) < 1e-10
        pattern = np.zeros_like(theta, dtype=float)
        pattern[~near_zero_mask] = ((np.cos(kL / 2 * np.cos(theta[~near_zero_mask])) - np.cos(kL / 2)) / sin_theta[~near_zero_mask]) ** 2
        return pattern

def calculate_directivity(L_over_lam, Rr):
    """
    Calculates maximum directivity using the simplified formula from the first code.
    """
    Q = Rr / 60 if Rr != 0 else 1
    kL = 2 * np.pi * L_over_lam
    theta_vals = np.linspace(0.001, np.pi, 100)
    F = antenna_pattern(theta_vals, kL).max()
    return 2 * F / Q if Q != 0 else 0

if __name__ == "__main__":
    # Compute impedance and directivity
    Rr, Rin, Xin, D = [], [], [], []
    for l in LENGTH_OVER_WAVELENGTH:
        Rr_val, Rin_val, Xin_val = calculate_impedance(l, WIRE_RADIUS)
        Rr.append(Rr_val)
        Rin.append(Rin_val)
        Xin.append(Xin_val)
        D.append(calculate_directivity(l, Rr_val))

    # Plot
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 6)
    ax1.set_ylim(-1000, 1000)
    ax1.plot(LENGTH_OVER_WAVELENGTH, Rin, label='Input Resistance (R)')
    ax1.plot(LENGTH_OVER_WAVELENGTH, Xin, label='Input Reactance (Xin)')
    ax1.plot(LENGTH_OVER_WAVELENGTH, Rr, label='Radiation Resistance(r)')
    ax1.set_xlabel('Dipole Length')
    ax1.set_ylabel('Impedance')
    ax1.set_title('Input Impedance & Maximum Directivity vs Dipole Length')
    ax1.legend()
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.plot(LENGTH_OVER_WAVELENGTH, D, "r", label='Maximum Directivity')
    ax2.set_ylabel('Directivity')
    ax2.legend()
    plt.show()
