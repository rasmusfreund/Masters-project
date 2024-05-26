import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from scipy.ndimage import convolve

# Load an example image from skimage
#image = color.rgb2gray(data.brick())

caller = getattr(data, 'brick')
image = caller()


# Define kernels for edge detection
vertical_kernel = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])

horizontal_kernel = np.array([[1, 2, 1],
                              [0, 0, 0],
                              [-1, -2, -1]])

# Apply convolution
vertical_edges = convolve(image, vertical_kernel)
horizontal_edges = convolve(image, horizontal_kernel)

# Plotting the results
fig, axs = plt.subplots(1, 3, figsize=(18, 7))

# Original image
axs[0].imshow(image, cmap='gray')
axs[0].set_title("Input Image", fontsize=22)
axs[0].axis('off')

# Vertical edge detection
axs[1].imshow(vertical_edges, cmap='gray')
axs[1].set_title("Vertical Edges", fontsize=22)
axs[1].axis('off')

# Horizontal edge detection
axs[2].imshow(horizontal_edges, cmap='gray')
axs[2].set_title("Horizontal Edges", fontsize=22)
axs[2].axis('off')

plt.tight_layout()

save_path = "/mnt/c/Users/rasmu/Desktop/Bioinformatics/MSc/Thesis/img/kernels.png"
plt.savefig(save_path, dpi=300)
plt.show()
