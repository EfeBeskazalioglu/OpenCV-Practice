import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('green_apple.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_green = np.array([20, 50, 50])
upper_green = np.array([100, 255, 255])

mask = cv2.inRange(img_hsv, lower_green, upper_green)

result = cv2.bitwise_and(img, img, mask=mask)

kernel = np.ones((15, 15), np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

clean_result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
clean_result = cv2.morphologyEx(clean_result, cv2.MORPH_OPEN, kernel)


fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Mask')
axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[2].set_title('Result')
axes[3].imshow(clean_result, cmap='gray')
axes[3].set_title('Cleaned Mask')

plt.show()
