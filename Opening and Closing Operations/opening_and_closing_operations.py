import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('./Opening and Closing Operations/image_2.png', 0)  # Load as grayscale

# Create a structuring element
kernel = np.ones((5, 5), np.uint8)

# Perform Opening operation
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Perform Closing operation
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Display the results
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(opening, cmap='gray'), plt.title('Opening Operation')
plt.subplot(133), plt.imshow(closing, cmap='gray'), plt.title('Closing Operation')
plt.show()
