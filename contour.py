import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("coins.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
blurred_img = cv2.GaussianBlur(img_gray,(7,7),0,0)

thresh = cv2.adaptiveThreshold(
    blurred_img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=51,
    C=10
)

kernel = np.ones((26,26),np.uint8)

thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=1)
thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=1)
contours , hierarchy =cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


cv2.drawContours(img, contours, -1, (0,255,0), 2)
print(f"Num of Coins: {len(contours)}")
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()