
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

txtfiles = []
for file in glob.glob("*.jpg"):
	txtfiles.append(file)

for ix in txtfiles:
	img = cv2.imread(ix, cv2.IMREAD_COLOR)
	imgtest1 = img.copy()
	imgtest = cv2.cvtColor(imgtest1, cv2.COLOR_BGR2GRAY)
	facecascade = cv2.CascadeClassifier(
		'D:\\KJ\\Nagesh\\Downloads\\face_recognition\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('D:\\KJ\\Nagesh\\Downloads\\face_recognition\\haarcascade_eye.xml')

faces = facecascade.detectMultiScale(imgtest, scaleFactor=1.2, minNeighbors=5)

print('TotalnumberofFacesfound', len(faces))

for (x, y, w, h) in faces:
		face_detect = cv2.rectangle(imgtest, (x, y), (x + w, y + h), (255, 0, 255), 2)
		roi_gray = imgtest[y:y + h, x:x + w]
		roi_color = imgtest[y:y + h, x:x + w]
		plt.imshow(face_detect)
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex, ey, ew, eh) in eyes:
			eye_detect = cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)
			plt.imshow(eye_detect)


