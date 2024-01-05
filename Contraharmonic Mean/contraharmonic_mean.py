import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

def contraharmonic_mean_filter(image, kernel_size, q):
    """
    Apply the contraharmonic mean filter to an image.
    
    This filter is useful in image processing for noise reduction,
    especially in cases where the noise is of a particular type (salt-and-pepper noise).
    It calculates the ratio of the sum of the pixels raised to the power of q to the sum of the pixels raised to the power of q-1.
    
    :param image: Input image array (should be in grayscale).
    :param kernel_size: Size of the kernel (must be odd).
    :param q: Order of the filter. Positive q values target pepper noise, negative q values target salt noise.
    :return: Filtered image.
    """
    # Convert the image to float for accurate computation
    image = image.astype(float)

    # Apply the contraharmonic mean filter
    numerator = ndimage.generic_filter(image, lambda x: (x**q).sum(), size=kernel_size)
    denominator = ndimage.generic_filter(image, lambda x: (x**(q-1)).sum(), size=kernel_size)

    # Avoid division by zero
    result = np.where(denominator != 0, numerator / denominator, 0)

    return result.astype(np.uint8)

# Example usage of the function
# Load an example image
image_path = './Contraharmonic Mean/image_1.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Apply contraharmonic mean filter
q_value = 1.5  # Adjust q value based on the type of noise
kernel_size = 3  # A typical choice for kernel size
filtered_image = contraharmonic_mean_filter(image, kernel_size, q_value)

# Save the image 
cv2.imwrite('./Contraharmonic Mean/filtered_image.png', filtered_image)

# Display the original and filtered images
plt.figure(figsize=(15, 5))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.axis('off')
plt.subplot(122), plt.imshow(filtered_image, cmap='gray'), plt.title('Filtered Image'), plt.axis('off')
plt.show()