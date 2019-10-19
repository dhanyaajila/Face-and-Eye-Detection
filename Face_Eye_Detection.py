import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:/Users/ANKITA ADITYA/Anaconda3/envs/opencv/opencv-3.4.3/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/ANKITA ADITYA/Anaconda3/envs/opencv/opencv-3.4.3/data/haarcascades/haarcascade_eye.xml')
cap=cv2.VideoCapture(0)
while True:
    ret, img=cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)
        font=cv2.FONT_HERSHEY_SIMPLEX #optional -> if you want to put text inside the detected face.
        cv2.putText(img,'Ankita',((int)(x+w/2),(int)(y+h/2)),font,0.5,(0,0,255),1,cv2.LINE_AA) # This will write the text inside the rectangle.
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh),(255,0,0),3)
    cv2.imshow('img',img)
    k=cv2.waitKey(30) & 0xFF
    if k=='q':
        break
cap.release() 
cv2.destroyAllWindows()
//change done
