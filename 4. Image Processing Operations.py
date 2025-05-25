import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            self.image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

    def process_and_display(self):
        # Histogram equalization
        eq_img = cv2.equalizeHist(self.image)
        # Thresholding
        _, thresh_img = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        # Edge detection
        edges = cv2.Canny(self.image, 50, 150)
        # Data augmentation
        rows, cols = self.image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
        rotated = cv2.warpAffine(self.image, M, (cols, rows))
        flipped = cv2.flip(self.image, 1)
        noise = np.random.normal(0, 25, self.image.shape)
        noisy = np.clip(self.image + noise, 0, 255).astype(np.uint8)
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        erosion = cv2.erode(self.image, kernel, iterations=1)
        dilation = cv2.dilate(self.image, kernel, iterations=1)
        opening = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)

        # Display results
        plt.figure(figsize=(20, 15))
        titles = ['Original', 'Histogram Equalized', 'Thresholded', 'Edge Detection', 'Rotated', 'Flipped', 'Noisy', 'Erosion', 'Dilation', 'Opening', 'Closing']
        images = [self.image, eq_img, thresh_img, edges, rotated, flipped, noisy, erosion, dilation, opening, closing]

        for i, (img, title) in enumerate(zip(images, titles), 1):
            plt.subplot(4, 4, i)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

# Example usage
processor = ImageProcessor('sample_image.jpg')
processor.process_and_display()
