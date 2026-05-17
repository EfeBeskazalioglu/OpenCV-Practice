import cv2
import numpy as np
import matplotlib.pyplot as plt


LOWER_GREEN = np.array([20, 50, 50])
UPPER_GREEN = np.array([100, 255, 255])
KERNEL_SIZE = 15


img = cv2.imread('green_apple.jpg')
if img is None:
    raise FileNotFoundError("green_apple.jpg bulunamadı!")


img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(img_hsv, LOWER_GREEN, UPPER_GREEN)


kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)


result = cv2.bitwise_and(img, img, mask=clean_mask)


fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Raw Mask')

axes[2].imshow(clean_mask, cmap='gray')
axes[2].set_title('Cleaned Mask')

axes[3].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[3].set_title('Final Result')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()