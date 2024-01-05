import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data, img_as_float
from skimage import exposure

# Load an example image from skimage's data module
image = img_as_float(data.coffee())

# Normalization (Min-Max Scaling)
def normalize(image):
    return (image - image.min()) / (image.ptp())

# Standardization (Z-Score Normalization)
def standardize(image):
    return (image - image.mean()) / image.std()

# Apply normalization and standardization
normalized_image = normalize(image)
standardized_image = standardize(image)

# Visualize the original, normalized, and standardized images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title('Original Image')

axes[1].imshow(normalized_image)
axes[1].set_title('Normalized Image')

# Standardized images can have negative values and a colormap can be used to visualize it
axes[2].imshow(standardized_image, cmap='gray')
axes[2].set_title('Standardized Image')

# Remove axis ticks
for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
