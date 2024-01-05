import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('./Smoothing - Sharpening/image_1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply Gaussian Blur for smoothing
smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

# Define and apply a sharpening kernel for sharpening
sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)


# Save the image 
cv2.imwrite('./Smoothing - Sharpening/smoothed_image_1.png', smoothed_image)
cv2.imwrite('./Smoothing - Sharpening/sharpened_image_1.png', sharpened_image)

# Display the original, smoothed, and sharpened images
plt.figure(figsize=(15, 15))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(smoothed_image)
plt.title('Smoothed Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sharpened_image)
plt.title('Sharpened Image')
plt.axis('off')

plt.show()
