import numpy as np
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
plt.show()