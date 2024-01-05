import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('./Laplacian Filter/image_1.png')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply the Laplacian filter
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

# Convert Laplacian to absolute values and convert to uint8
abs_laplacian = np.uint8(np.absolute(laplacian))

# Apply Gaussian blur to the Laplacian
blurred_laplacian = cv2.GaussianBlur(abs_laplacian, (5, 5), 0)

# Save the image 
cv2.imwrite('./Laplacian Filter/abs_laplacian_image_1.png', abs_laplacian)
cv2.imwrite('./Laplacian Filter/blurred_laplacian_image_1.png', blurred_laplacian)

# Display images using matplotlib
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(abs_laplacian, cmap='gray')
plt.title('Laplacian Filtered Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.imshow(blurred_laplacian, cmap='gray')
plt.title('Blurred Laplacian Image'), plt.xticks([]), plt.yticks([])

plt.show()
