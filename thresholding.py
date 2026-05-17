import cv2
import numpy as np
import matplotlib.pyplot as plt

img_gray = cv2.imread('book page.jpg', cv2.IMREAD_GRAYSCALE)
_,simple = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)

ret, otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu's threshold: {ret}")

adaptive_mean = cv2.adaptiveThreshold(
    img_gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11,
    2    
)
adaptive_gauss = cv2.adaptiveThreshold(
    img_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

plt.figure(figsize=(10, 8))
plt.subplot(2, 3, 1), plt.imshow(img_gray, cmap='gray'), plt.title('Original Grayscale'), plt.axis('off')
plt.subplot(2, 3, 2), plt.imshow(simple, cmap='gray'), plt.title('Simple Thresholding'), plt.axis('off')
plt.subplot(2, 3, 3), plt.imshow(otsu, cmap='gray'), plt.title("Otsu's Thresholding"), plt.axis('off')
plt.subplot(2, 3, 4), plt.imshow(adaptive_mean, cmap='gray'), plt.title('Adaptive Mean Thresholding'), plt.axis('off')
plt.subplot(2, 3, 5), plt.imshow(adaptive_gauss, cmap='gray'), plt.title('Adaptive Gaussian Thresholding'), plt.axis('off')
plt.subplot(2, 3, 6), plt.hist(img_gray.ravel(), 256, [0,256])
plt.title('Histogram'), plt.axvline(ret, color='r')
plt.tight_layout() 
plt.show()