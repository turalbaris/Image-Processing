import numpy as np
import matplotlib.pyplot as plt
import cv2

# Image loading and conversion to grayscale
image = cv2.imread('./Bit Plane Slicing/image_1.png', cv2.IMREAD_GRAYSCALE)

# Function to perform bit plane slicing
def bit_plane_slicing(image):
    # Convert the image to a numpy array and get the shape
    rows, cols = image.shape
    # Create an array to store the 8 bit planes
    planes = [np.zeros((rows, cols), dtype=np.uint8) for _ in range(8)]

    # Extract each bit plane
    for i in range(8):
        # Shift right by i bits and bitwise AND with 1 to extract the ith bit plane
        planes[i] = (image >> i) & 1
        # Scale the bit plane for visualization (0 or 1 to 0 or 255)
        planes[i] *= 255

    return planes

# Apply the function
bit_planes = bit_plane_slicing(image)

# Plotting the original image and the bit planes
plt.figure(figsize=(12, 8))
plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

for i, plane in enumerate(bit_planes):
    plt.subplot(3, 3, i+2)
    plt.imshow(plane, cmap='gray')
    plt.title(f'Bit plane {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()
