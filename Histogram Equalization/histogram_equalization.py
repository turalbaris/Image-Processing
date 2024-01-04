import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram_equalization(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply histogram equalization
    equ = cv2.equalizeHist(img)

    # Display the original and equalized images side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(equ, cmap='gray')
    plt.title('Histogram Equalized Image')
    plt.xticks([])
    plt.yticks([])

    plt.show()

image_path = './Histogram Equalization/image_1.png'
histogram_equalization(image_path)
