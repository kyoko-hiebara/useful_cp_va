import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better visualization
sns.set(style='whitegrid', context='talk')

def parse_extxyz(filename):
    """
    Parse an extxyz file to extract energy and compute the RMS of forces
    from each configuration.
    """
    energies = []
    forces_rms = []
    
    with open(filename, 'r') as f:
        while True:
            header = f.readline()
            if not header:
                break  # End of file
            try:
                natoms = int(header.strip())
            except ValueError:
                break
            # Extract energy from the comment line using regex
            comment = f.readline().strip()
            energy_match = re.search(r'energy\s*=\s*([-\d.]+)', comment)
            if energy_match:
                energy = float(energy_match.group(1))
            else:
                energy = None
            energies.append(energy)
            
            # Read each atom's data (assuming last three columns are force components)
            forces = []
            for _ in range(natoms):
                line = f.readline().strip().split()
                if len(line) < 7:
                    continue  # Skip incomplete lines
                fx, fy, fz = map(float, line[-3:])
                forces.append([fx, fy, fz])
            forces = np.array(forces)
            if forces.size > 0:
                # Calculate RMS of forces for this configuration
                rms = np.sqrt(np.mean(np.sum(forces**2, axis=1)))
            else:
                rms = None
            forces_rms.append(rms)
    
    return np.array(energies), np.array(forces_rms)

# Replace filenames with your actual file names
dft_energy, dft_force = parse_extxyz('dft.extxyz')
mlff_energy, mlff_force = parse_extxyz('mlff.extxyz')

# Calculate energy difference and RMS force difference
energy_diff = mlff_energy - dft_energy
force_diff = mlff_force - dft_force

# Create a single figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(8, 14))

# 1. Energy parity plot (DFT vs MLFF)
axes[0].scatter(dft_energy, mlff_energy, s=60, edgecolor='k', label='Data Points')
min_val = min(np.min(dft_energy), np.min(mlff_energy))
max_val = max(np.max(dft_energy), np.max(mlff_energy))
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal: y=x')
axes[0].set_xlabel('DFT Energy')
axes[0].set_ylabel('MLFF Energy')
axes[0].set_title('Energy Parity Plot')
axes[0].legend()

# 2. Energy error plot (MLFF - DFT)
axes[1].plot(energy_diff, marker='o', lw=2, label='Energy Error')
axes[1].set_xlabel('Configuration Index')
axes[1].set_ylabel('Energy Difference (MLFF - DFT)')
axes[1].set_title('Energy Error')
axes[1].legend()

# 3. RMS force error plot (MLFF - DFT)
axes[2].plot(force_diff, marker='o', lw=2, color='purple', label='Force RMS Error')
axes[2].set_xlabel('Configuration Index')
axes[2].set_ylabel('RMS Force Difference (MLFF - DFT)')
axes[2].set_title('RMS Force Error')
axes[2].legend()

plt.tight_layout()
plt.savefig('comparison_plots.png', dpi=300)
plt.show()
