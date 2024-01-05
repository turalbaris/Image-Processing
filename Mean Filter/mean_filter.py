import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply a mean filter to an image
def apply_mean_filter(image_path, kernel_size):

    # Load the image in grayscale
    image = cv2.imread(image_path, 0)  # 0 to read image in grayscale mode

    # Define the kernel - all elements are 1
    kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size**2
    
    # Apply the mean filter using the filter2D function
    mean_filtered_image = cv2.filter2D(image, -1, kernel)
    
    # Display the original and filtered images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.axis('off')
    plt.subplot(1, 2, 2), plt.imshow(mean_filtered_image, cmap='gray'), plt.title('Mean Filtered Image'), plt.axis('off')
    plt.show()


# Apply a 5x5 mean filter to the uploaded image
apply_mean_filter('./Mean Filter/image_1.png', 5)

