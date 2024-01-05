import cv2
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('./Gaussian Filter/image_1.png')  # Image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Fix color format

# Apply Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)  # (5, 5) kernel size, 0 sigma

# Save the image 
cv2.imwrite('./Gaussian Filter/gaussian_blured_image_1.png', gaussian_blur)

# Display the Original and Processed Images
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image), plt.title('Original Image')
plt.subplot(122), plt.imshow(gaussian_blur), plt.title('Gaussian Blurred Image')
plt.show()


