import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('./Sobel Filter/image_1.png', cv2.IMREAD_GRAYSCALE)

# Apply Sobel Filters
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in the X direction
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in the Y direction

# Take the absolute values to enhance edges and combine them
sobel_combined = np.sqrt(np.square(sobel_x) + np.square(sobel_y))


# Display the results
plt.figure(figsize=(20, 10))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 3, 2), plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel X')
plt.subplot(1, 3, 3), plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel Y')
plt.show()
