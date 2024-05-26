import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the figure and axis
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Define the size of the grids
input_size = 6
kernel_size = 3
feature_map_size = input_size - kernel_size + 1
square_size = 2 

# Draw the input grid
for i in range(input_size):
    for j in range(input_size):
        rect = patches.Rectangle((j * square_size, input_size * square_size - (i + 1) * square_size), square_size, square_size, fill=None, edgecolor='black')
        ax.add_patch(rect)

# Highlight a region of the input grid
for i in range(2, 2 + kernel_size):
    for j in range(2, 2 + kernel_size):
        rect = patches.Rectangle((j * square_size, input_size * square_size - (i + 1) * square_size), square_size, square_size, fill=True, edgecolor='black', facecolor='blue', alpha=0.3)
        ax.add_patch(rect)

# Draw the feature map grid
for i in range(feature_map_size):
    for j in range(feature_map_size):
        rect = patches.Rectangle((j * square_size + input_size * square_size + 2 * square_size, feature_map_size * square_size - (i + 1) * square_size), square_size, square_size, fill=None, edgecolor='black')
        ax.add_patch(rect)

# Highlight a region of the feature map grid
rect = patches.Rectangle((2 * square_size + input_size * square_size + 2 * square_size, feature_map_size * square_size - 2 * square_size - square_size), square_size, square_size, fill=True, edgecolor='black', facecolor='red', alpha=0.3)
ax.add_patch(rect)

# Draw the arrows
for i in range(kernel_size):
    for j in range(kernel_size):
        ax.arrow(2.5 * square_size + j * square_size, input_size * square_size - 2.5 * square_size - i * square_size, 16 - (j * 2), -4 + (2 * i), head_width=0.1 * square_size, head_length=0.1 * square_size, fc='red', ec='red', alpha=0.5)

# Labels
ax.text(3 * square_size, -0.5 * square_size, 'Input', fontsize=16, ha='center')
ax.text(3 * square_size + input_size * square_size + square_size, -0.5 * square_size, 'Feature Map', fontsize=16, ha='center')
ax.text(3.5 * square_size, 4.1 * square_size, 'Kernel', fontsize=16, ha='center')

ax.set_xlim(0, (input_size + feature_map_size + 4) * (0.9 * square_size))
ax.set_ylim(-2 * square_size, input_size * square_size + 2 * square_size)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()

# Save the figure
save_path = "/mnt/c/Users/rasmu/Desktop/Bioinformatics/MSc/Thesis/img/receptive_field.png"
plt.savefig(save_path, dpi=300)
plt.show()
