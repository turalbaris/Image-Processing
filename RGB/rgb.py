import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

# Load the image
def load_image(image_path):
    image = Image.open(image_path)
    return np.array(image)

# Split RGB channels
def split_rgb_channels(image):
    red_channel = image.copy()
    green_channel = image.copy()
    blue_channel = image.copy()

    # Zero out channels other than red
    red_channel[:, :, 1] = 0  # green set to zero
    red_channel[:, :, 2] = 0  # blue set to zero

    # Zero out channels other than green
    green_channel[:, :, 0] = 0  # red set to zero
    green_channel[:, :, 2] = 0  # blue set to zero

    # Zero out channels other than blue
    blue_channel[:, :, 0] = 0  # red set to zero
    blue_channel[:, :, 1] = 0  # green set to zero

    return red_channel, green_channel, blue_channel

# Display the images
def show_channels(red, green, blue):
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(red)
    plt.title("Red Channel")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(green)
    plt.title("Green Channel")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(blue)
    plt.title("Blue Channel")
    plt.axis('off')

    plt.show()

# Main function
def main():
    image_path = './RGB/image_1.png'
    image = load_image(image_path)
    red, green, blue = split_rgb_channels(image)
    
    # Save the image 
    cv2.imwrite('./RGB/red_image_1.png', red)
    cv2.imwrite('./RGB/green_image_1.png', green)
    cv2.imwrite('./RGB/blue_image_1.png', blue)

    show_channels(red, green, blue)

if __name__ == "__main__":
    main()