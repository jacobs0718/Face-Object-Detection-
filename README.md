# Face-Object-Detection-
Face/Object Detection using Python and OpenCV

# Requirement
1. OpenCV packages for Python.
2. Python 3.x (3.4+)
3. Numpy package (for example, using pip install numpy command).

# Cascade Detection in OpenCV
First we need to load the required XML classifiers.

```py
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('./cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./cascade/haarcascade_eye.xml')
```
Now we find the faces in the image. If faces are found, it returns the positions of detected faces as Rectangle(x,y,w,h). Once we get these locations, we can create a ROI for the face and apply eye detection on this ROI

```py
while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h) ,(255 ,0 ,0) ,2)
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = frame[y: y + h, x: x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        success, image = cap.read()
        crop_img = image[y: y + h, x: x + w]

        cv2.imwrite(f"./faces/frame{number_of_frame}.jpg", crop_img)
        number_of_frame += 1

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0,255,0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#Additional Resources
https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/

