import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, amount, pepper_fraction):
    noisy_image = np.copy(image)

    num_salt = np.ceil(amount * image.size * (1.0 - pepper_fraction)).astype(int)
    num_pepper = np.ceil(amount * image.size * pepper_fraction).astype(int)

    # Salt noise (white pixels)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[tuple(salt_coords)] = 255
    salt_coords = list(zip(salt_coords[0], salt_coords[1]))

    # Pepper noise (black pixels)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    pepper_coords = list(zip(pepper_coords[0], pepper_coords[1]))

    return noisy_image, salt_coords, pepper_coords

def enlarge_points(image, coords, color):
    for x, y in coords:
        max_x, max_y = image.shape
        # Check and change the surrounding pixels
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < max_x and 0 <= new_y < max_y:
                    image[new_x, new_y] = color
    return image

# Load an example image in grayscale
image_path = './Salt and pepper/image_1.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Add salt and pepper noise to the image
amount_of_noise = 0.002  # 0.5% of the pixels will be noisy
pepper_fraction = 0.5  # Fraction of noise that is pepper
noisy_image, salt_coords, pepper_coords = add_salt_and_pepper_noise(image, amount_of_noise, pepper_fraction)

# Enlarge salt and pepper points
noisy_image = enlarge_points(noisy_image, pepper_coords, 0)  # For pepper, color is black (0)
noisy_image = enlarge_points(noisy_image, salt_coords, 255)  # For salt, color is white (255)

# Save the image 
#cv2.imwrite('filtered_image.png', noisy_image)

# Display the images
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.axis('off')
plt.subplot(132), plt.imshow(noisy_image, cmap='gray'), plt.title('Noisy Image'), plt.axis('off')
plt.show()
