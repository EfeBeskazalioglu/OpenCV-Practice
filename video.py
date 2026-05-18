import cv2
import numpy as np
cap = cv2.VideoCapture("samples_data_vtest.avi")

backSub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=True
)

if not cap.isOpened():
    print("Video açılamadı!")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti veya okunamadı")
        break

    fg_mask = backSub.apply(frame)

    kernel= np.ones((5,5),np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # blurred_frame = cv2.GaussianBlur(frame_gray,(5,5),0)
    # canny_frame = cv2.Canny(blurred_frame,100,200)

    contours,_ = cv2.findContours(fg_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    filtered = [c for c in contours if cv2.contourArea(c) > 500]

    for c in filtered:
        x, y, w ,h = cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.putText(frame, f"Nesneler: {len(filtered)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

   

    cv2.imshow("Original Video",frame)
    cv2.imshow("Foreground Mask", fg_mask)
    # cv2.imshow("Gray",frame_gray)    
    # cv2.imshow("Edges", canny_frame)

    # 'q' tuşuyla çık
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()