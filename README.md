# Face-Object-Detection-
Face/Object Detection using Python and OpenCV

# Requirement
1. OpenCV packages for Python.
2. Python 3.x (3.4+)
3. Numpy package (for example, using pip install numpy command).

# Installation
1. Run pip install -r requirements.txt

# Cascade Detection in OpenCV
First we need to load the required XML classifiers.

```py

face_cascade = cv2.CascadeClassifier('./cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./cascade/haarcascade_eye.xml')

```
Now we find the faces in the image. If faces are found, it returns the positions of detected faces as Rectangle(x,y,w,h). Once we get these locations, we can create a ROI for the face and apply eye detection on this ROI. Finally, we are cropping face and saving in 'faces/' folder. 

```py

for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h) ,(255 ,0 ,0) ,2)
    roi_gray = gray[y: y + h, x: x + w]
    roi_color = frame[y: y + h, x: x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    success, image = cap.read()
    crop_img = image[y: y + h, x: x + w]

    cv2.imwrite(f"./faces/frame{number_of_frame}.jpg", crop_img)

```

# Additional Resources
https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/

