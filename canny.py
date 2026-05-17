import cv2
import numpy as np
import matplotlib.pyplot as plt

img_gray = cv2.imread('car.jpg', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img_gray, 50, 100)
edges_strong = cv2.Canny(img_gray, 100, 200)
edges_strongest = cv2.Canny(img_gray, 200, 300)

blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
edges_blurred = cv2.Canny(blurred, 100, 200)

v = np.median(blurred)
lower = int(max(0, 0.66 * v))
upper = int(min(255, 1.33 * v))
auto_edges = cv2.Canny(blurred, lower, upper)

plt.figure(figsize=(10, 10))
plt.subplot(4, 2, 1), plt.imshow(img_gray, cmap='gray'), plt.title('Original Grayscale'), plt.axis('off')
plt.subplot(4, 2, 2), plt.imshow(edges, cmap='gray'), plt.title('Canny Edges'), plt.axis('off')
plt.subplot(4, 2, 3), plt.imshow(edges_strong, cmap='gray'), plt.title('Strong Canny Edges'), plt.axis('off')
plt.subplot(4, 2, 4), plt.imshow(edges_strongest, cmap='gray'), plt.title('Strongest Canny Edges'), plt.axis('off')
plt.subplot(4, 2, 5), plt.imshow(blurred, cmap='gray'), plt.title('Blurred Image'), plt.axis('off')
plt.subplot(4, 2, 6), plt.imshow(auto_edges, cmap='gray'), plt.title('Auto Canny Edges'), plt.axis('off')
plt.subplot(4, 2, 7), plt.imshow(edges_blurred, cmap='gray'), plt.title('Canny on Blurred Image'), plt.axis('off')

plt.tight_layout()
plt.show()