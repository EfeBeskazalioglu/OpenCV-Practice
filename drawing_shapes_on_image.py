import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('puppy.jpg')
print(img.shape)
x_middle = img.shape[1]//2
y_middle = img.shape[0]//2

img_changed = img.copy()
cv2.line(img_changed, (0,0), (img.shape[1], img.shape[0]), (255,0,0), 10)
cv2.circle(img_changed, (x_middle,y_middle), 50, (0,255,0), -1)
right_top_x = img.shape[1] - 200
cv2.rectangle(img_changed, (right_top_x, 0), (img.shape[1], 150), (0,0,255), 5)
cv2.putText(img_changed, 'OpenCV Tutorial', (50,img.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
plt.imshow(cv2.cvtColor(img_changed, cv2.COLOR_BGR2RGB))
plt.show()