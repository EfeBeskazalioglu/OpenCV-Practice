import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('green_apple.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_green = np.array([20, 50, 50])
upper_green = np.array([100, 255, 255])

mask = cv2.inRange(img_hsv, lower_green, upper_green)

result = cv2.bitwise_and(img, img, mask=mask)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Mask')
axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[2].set_title('Result')
plt.show()