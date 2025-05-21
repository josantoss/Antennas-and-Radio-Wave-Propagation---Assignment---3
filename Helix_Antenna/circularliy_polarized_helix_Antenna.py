'''import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# Constants
freq = 600e6          # Frequency in Hz (600 MHz)
c = 3e8               # Speed of light in m/s
lam = c / freq        # Wavelength λ

# ————————————————————————————————
# Axial Mode Helix Parameters
a_axial = lam / (2 * np.pi)       # Radius (m): Circumference ≈ λ
pitch_axial = 0.25 * lam         # Pitch per turn (m)
num_turns_axial = 10              # Number of turns (standard for axial mode)
length_axial = num_turns_axial * pitch_axial  # Total length

# ————————————————————————————————
# Normal Mode Helix Parameters
a_normal = 0.01 * lam            # Small radius compared to axial
pitch_normal = 0.05 * lam        # Very tight winding
num_turns_normal = 3             # Fewer turns
length_normal = num_turns_normal * pitch_normal

# ————————————————————————————————
# Plotting Both Helices Side-by-Side

def plot_helix(ax, radius, pitch, turns, title):
    """
    Draws a helix on a given 3D axis.
    
    Parameters:
    - ax: Matplotlib 3D subplot
    - radius: Radius of helix (m)
    - pitch: Distance between turns (m)
    - turns: Number of full rotations
    - title: Title for the subplot
    """
    t = np.linspace(0, 2 * np.pi * turns, 1000)
    z = t * (pitch / (2 * np.pi))  # Linear increase along Z-axis
    x = radius * np.cos(t)
    y = radius * np.sin(t)

    ax.plot(x, y, z, color='blue', linewidth=2)
    ax.set_title(title, fontsize=12, pad=20)  # Add space above title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_box_aspect([1, 1, 3])  # Stretch Z-axis for better visibility


# Create figure with increased width and padding
fig = plt.figure(figsize=(16, 7))  # Wider figure to avoid title overlap

# Subplot 1: Axial Mode Helix
ax1 = fig.add_subplot(121, projection='3d')
plot_helix(ax1, a_axial, pitch_axial, num_turns_axial, "Axial Mode\nHelix")

# Subplot 2: Normal Mode Helix
ax2 = fig.add_subplot(122, projection='3d')
plot_helix(ax2, a_normal, pitch_normal, num_turns_normal, "Normal Mode\nHelix")

# Adjust layout with extra padding
plt.subplots_adjust(top=0.85, bottom=0.1, left=0.05, right=0.95, wspace=0.3)
plt.suptitle("Helix Antenna Geometry Comparison", fontsize=16, y=0.95)

# Display final plot
plt.show()'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Try to set an interactive backend for displaying plots
try:
    plt.switch_backend('TkAgg')  # Use TkAgg for interactive rendering
except ImportError:
    print("Warning: TkAgg backend unavailable. Install Tkinter or try another backend (e.g., pip install tk). Falling back to default backend.")
    # Default backend may still work in some VS Code setups

# Constants
freq = 600e6          # Frequency in Hz (600 MHz)
c = 3e8               # Speed of light in m/s
lam = c / freq        # Wavelength (m)

# ————————————————————————————————
# Design Equations for Helix Antennas

def design_axial_helix(freq, c, num_turns=10):
    """
    Design parameters for Axial Mode Helix.
    Returns: radius, pitch, length, circumference-to-wavelength ratio.
    """
    lam = c / freq
    C_lambda = 0.75 * lam  # Circumference ~ 0.75*lambda to 1.33*lambda for axial mode
    radius = C_lambda / (2 * np.pi)  # Radius = C / (2π)
    pitch_angle = np.arctan(0.25)  # Pitch angle ~ 12-14° (tan α = S/C)
    pitch = 0.25 * lam  # Pitch S = 0.25*lambda (standard for axial mode)
    length = num_turns * pitch
    return radius, pitch, length, C_lambda / lam

def design_normal_helix(freq, c, num_turns=3):
    """
    Design parameters for Normal Mode Helix.
    Returns: radius, pitch, length, circumference-to-wavelength ratio.
    """
    lam = c / freq
    radius = 0.01 * lam  # Small radius: C << lambda
    pitch = 0.05 * lam   # Tight pitch: S << lambda
    length = num_turns * pitch
    C_lambda = 2 * np.pi * radius
    return radius, pitch, length, C_lambda / lam

# Calculate parameters
a_axial, pitch_axial, length_axial, C_lambda_axial = design_axial_helix(freq, c)
a_normal, pitch_normal, length_normal, C_lambda_normal = design_normal_helix(freq, c)

# ————————————————————————————————
# Radiation Pattern (Analytical Approximation)

def axial_mode_pattern(theta, num_turns, pitch, lam):
    """
    Approximate radiation pattern for axial mode helix (along Z-axis).
    Theta in radians.
    """
    k = 2 * np.pi / lam
    S = pitch
    N = num_turns
    # Simplified pattern: E ~ sin(θ) * array factor
    array_factor = np.sin(N * k * S * np.cos(theta) / 2) / np.sin(k * S * np.cos(theta) / 2)
    pattern = np.abs(np.sin(theta) * array_factor)
    return pattern / np.max(pattern)  # Normalize

def normal_mode_pattern(theta, phi, radius, lam):
    """
    Approximate radiation pattern for normal mode helix (omnidirectional in XY-plane).
    """
    k = 2 * np.pi / lam
    # Normal mode: dipole-like along helix axis, omnidirectional in XY
    pattern = np.cos(theta)  # Dipole-like pattern
    return pattern / np.max(pattern)  # Normalize

# Generate angles for patterns
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
Theta, Phi = np.meshgrid(theta, phi)

# Calculate patterns
axial_pattern = axial_mode_pattern(theta, num_turns=10, pitch=pitch_axial, lam=lam)
normal_pattern = normal_mode_pattern(theta, phi, radius=a_normal, lam=lam)

# ————————————————————————————————
# Axial Ratio (Simplified Estimation)

def axial_ratio_axial(C_lambda, pitch, num_turns):
    """
    Estimate axial ratio for axial mode helix.
    """
    # Ideal axial ratio ~ 1 for circular polarization when C ≈ lambda and pitch angle ~ 12-14°
    AR = (2 * num_turns + 1) / (2 * num_turns)  # Simplified model
    return AR

def axial_ratio_normal():
    """
    Axial ratio for normal mode (linear polarization).
    """
    return np.inf  # Linear polarization (not circular)

AR_axial = axial_ratio_axial(C_lambda_axial, pitch_axial, num_turns=10)
AR_normal = axial_ratio_normal()

# ————————————————————————————————
# Bandwidth Estimation
def bandwidth_axial(C_lambda, num_turns):
    """
    Estimate bandwidth for axial mode helix.
    """
    # Axial mode: ~50% bandwidth around center frequency
    return 0.5 * freq

def bandwidth_normal():
    """
    Estimate bandwidth for normal mode helix.
    """
    # Normal mode: narrow bandwidth, ~10% of center frequency
    return 0.1 * freq

bw_axial = bandwidth_axial(C_lambda_axial, num_turns=10)
bw_normal = bandwidth_normal()

# ————————————————————————————————
# Plotting Functions

def plot_helix(ax, radius, pitch, turns, title):
    """
    Draws a helix on a given 3D axis.
    """
    t = np.linspace(0, 2 * np.pi * turns, 1000)
    z = t * (pitch / (2 * np.pi))  # Linear increase along Z-axis
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    ax.plot(x, y, z, color='blue', linewidth=2)
    ax.set_title(title, fontsize=12, pad=20)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_box_aspect([1, 1, 3])

def plot_radiation_pattern(ax, pattern, theta, title):
    """
    Plot 2D radiation pattern (elevation).
    """
    ax.plot(np.degrees(theta), pattern, color='red', linewidth=2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Normalized Gain')
    ax.grid(True)

# ————————————————————————————————
# Generate Plots
fig = plt.figure(figsize=(18, 10))

# Helix Geometry: Axial Mode
ax1 = fig.add_subplot(231, projection='3d')
plot_helix(ax1, a_axial, pitch_axial, 10, title="Axial Mode Helix")

# Helix Geometry: Normal Mode
ax2 = fig.add_subplot(232, projection='3d')
plot_helix(ax2, a_normal, pitch_normal, 3, title="Normal Mode Helix")

# Radiation Pattern: Axial Mode
ax3 = fig.add_subplot(233)
plot_radiation_pattern(ax3, axial_pattern, theta, "Axial Mode Radiation Pattern")

# Radiation Pattern: Normal Mode
ax4 = fig.add_subplot(234)
plot_radiation_pattern(ax4, normal_pattern, theta, "Normal Mode Radiation Pattern")

# Adjust layout
plt.subplots_adjust(top=0.85, bottom=0.1, left=0.05, right=0.95, wspace=0.4, hspace=0.4)
plt.suptitle("Helix Antenna Design and Radiation Patterns @ 600 MHz", fontsize=16, y=0.95)

# Save plot
plt.savefig('helix_antenna_design.png')

# Display the plot
plt.show(block=True)  


# ————————————————————————————————
# This part of code will generate result report.
report = f"""
# Helix Antenna Design Report @ 600 MHz

## Axial Mode Helix
- **Radius**: {a_axial:.4f} m
- **Pitch**: {pitch_axial:.4f} m
- **Number of Turns**: 10
- **Length**: {length_axial:.4f} m
- **Circumference/Wavelength (C/lambda)**: {C_lambda_axial:.4f}
- **Axial Ratio**: {AR_axial:.2f} (Circular Polarization)
- **Bandwidth**: ±{bw_axial/1e6:.2f} MHz
- **Radiation Pattern**: Directive along helix axis (Z-axis)

## Normal Mode Helix
- **Radius**: {a_normal:.4f} m
- **Pitch**: {pitch_normal:.4f} m
- **Number of Turns**: 3
- **Length**: {length_normal:.4f} m
- **Circumference/Wavelength (C/lambda)**: {C_lambda_normal:.4f}
- **Axial Ratio**: {AR_normal} (Linear Polarization)
- **Bandwidth**: ±{bw_normal/1e6:.2f} MHz
- **Radiation Pattern**: Omnidirectional in XY-plane, dipole-like

## Notes
- Axial mode provides circular polarization with high gain, suitable for satellite communication.
- Normal mode is linearly polarized, compact, and suitable for short-range applications.
- Radiation patterns are analytical approximations; for precise results, use NEC simulation.
- Plot saved as 'helix_antenna_design.png' and displayed.
"""
print(report)

# Save report to file with UTF-8 encoding to handle special characters
with open('helix_antenna_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)