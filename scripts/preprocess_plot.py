import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


# Load the original data
input_path = 'ff3751e9-a2c0-4902-9d8d-43dd5f7b26a6.txt'
data = pd.read_csv(input_path, delim_whitespace=True, skiprows=3, names=['mass', 'int'])

# Step 1: Log-transform the intensity values
data['int_log'] = np.log(data['int'] + 1)

# Step 2: First LOWESS smoothing with span 0.0008
spl1 = UnivariateSpline(data['mass'], data['int_log'], s=0.0008 * len(data))
data['int_smoothed1'] = spl1(data['mass'])

# Step 3: Second LOWESS smoothing with span 0.3
spl2 = UnivariateSpline(data['mass'], data['int_smoothed1'], s=0.3 * len(data))
data['int_normalized'] = data['int_smoothed1'] / spl2(data['mass'])

# Step 4: Rescaling to range [-1, 1]
int_min, int_max = data['int_normalized'].min(), data['int_normalized'].max()
data['int_rescaled'] = 2 * (data['int_normalized'] - int_min) / (int_max - int_min) - 1

# Step 5: Mapping to new scale and final LOWESS smoothing with span 0.0003
mass_new = np.arange(2000, 20001, 0.5)
spl3 = UnivariateSpline(data['mass'], data['int_rescaled'], s=0.0003 * len(data))
int_smoothed_final = spl3(mass_new)

# Filter the data to focus on the range from 2000 to 10000 Da
data_filtered = data[(data['mass'] >= 2000) & (data['mass'] <= 10000)]

# Create the figure with adjusted layout and smaller x-axis range
fig = plt.figure(figsize=(12, 18))
grid = plt.GridSpec(3, 3, height_ratios=[3, 2, 3], hspace=0.25)

# Original spectrum
ax0 = fig.add_subplot(grid[0, :])
ax0.plot(data['mass'], data['int'], label='Original')
ax0.set_title('Original Mass Spectrum')
ax0.set_xlabel('m/z')
ax0.set_ylabel('Intensity')

# Intermediate steps
ax1 = fig.add_subplot(grid[1, 0])
ax1.plot(data_filtered['mass'], data_filtered['int_log'], label='Log-Transformed')
ax1.set_title('Log-Transformed Mass Spectrum')
ax1.set_xlabel('m/z')
ax1.set_ylabel('Log-Intensity')

ax2 = fig.add_subplot(grid[1, 1])
ax2.plot(data_filtered['mass'], data_filtered['int_smoothed1'], label='First LOWESS Smoothing')
ax2.set_title('First LOWESS Smoothing')
ax2.set_xlabel('m/z')
ax2.set_ylabel('Smoothed Intensity')

ax3 = fig.add_subplot(grid[1, 2])
ax3.plot(data_filtered['mass'], data_filtered['int_normalized'], label='Second LOWESS & Normalized')
ax3.set_title('Second LOWESS\nSmoothing & Normalization')
ax3.set_xlabel('m/z')
ax3.set_ylabel('Normalized Intensity')

# Final rescaled spectrum
ax4 = fig.add_subplot(grid[2, :])
ax4.plot(mass_new, int_smoothed_final, label='Final Rescaled & Smoothed')
ax4.set_title('Final Rescaled & Smoothed Mass Spectrum')
ax4.set_xlabel('m/z')
ax4.set_ylabel('Rescaled Intensity')

plt.tight_layout()
plt.savefig('preprocessing_steps_combined.png', dpi=300)
plt.show()
