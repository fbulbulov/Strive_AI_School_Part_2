import cv2
import numpy as np

cap= cv2.VideoCapture('videos/video.avi')

detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
green = (0,255,0)

while True:
    ret, frame = cap.read()
    #height, width, _ = frame.shape
    #print (height, width)
    # Return the area of interest
    #roi = frame [340: 720, 500:800]

    #Detection
    mask=detector.apply(frame)
    #Fining the countours
    contours, _ =cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 90:
            #cv2.drawContours(frame, [contour], -1,green, 2)
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
            

    
    cv2.imshow("Frame", frame)

    cv2.imshow('Mask', mask)


    key=cv2.waitKey(30)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

