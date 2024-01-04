from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(img, lower_percentile, upper_percentile):

    # Convert image to numpy array
    img_array = np.asarray(img)

    # Calculate lower and upper bounds for contrast stretching
    lower_bound = np.percentile(img_array, lower_percentile)
    upper_bound = np.percentile(img_array, upper_percentile)

    # Stretch the contrast
    stretched_img = np.clip((img_array - lower_bound) / (upper_bound - lower_bound) * 255, 0, 255).astype(np.uint8)

    # Convert back to image
    stretched_img = Image.fromarray(stretched_img)

    return stretched_img


img_path = './Contrast Stretching/before_contrast_stretching.png'
original_img = Image.open(img_path)

# Perform contrast stretching
stretched_img = contrast_stretching(original_img, 5, 95)

# Display the original and stretched images for comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(stretched_img, cmap='gray')
plt.title('Contrast Stretched Image')
plt.axis('off')

plt.show()
