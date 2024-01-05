import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, amount):
    # Create a copy to avoid modifying the original image
    noisy_image = np.copy(image)

    # Calculate number of pixels to add noise to
    num_salt = np.ceil(amount * image.size * 0.5).astype(int)
    num_pepper = np.ceil(amount * image.size * 0.5).astype(int)

    # Add salt noise (white pixels)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[tuple(coords)] = 255

    # Add pepper noise (black pixels)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[tuple(coords)] = 0

    return noisy_image

def apply_median_filter(image, kernel_size):
    # Apply median filter and return the filtered image
    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image

# Load an example image in grayscale
image_path = './Salt and pepper/image_2.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Add salt and pepper noise to the image
amount_of_noise = 0.05  # 5% of the pixels will be noisy
noisy_image = add_salt_and_pepper_noise(image, amount_of_noise)

# Apply a median filter to the noisy image
kernel_size = 5  # Size of the kernel (must be odd)
filtered_image = apply_median_filter(noisy_image, kernel_size)

# Display the images
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.axis('off')
plt.subplot(132), plt.imshow(noisy_image, cmap='gray'), plt.title('Noisy Image'), plt.axis('off')
plt.subplot(133), plt.imshow(filtered_image, cmap='gray'), plt.title('Filtered Image (Median Filter)'), plt.axis('off')
plt.show()
