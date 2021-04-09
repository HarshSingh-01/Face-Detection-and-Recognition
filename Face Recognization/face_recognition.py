import os
import cv2
import numpy as np

haar_cascade = cv2.CascadeClassifier('Face Detection and Recognition\haar_face.xml')

DIR = r'Face Detection and Recognition\Faces\train'
people = []
for i in os.listdir(DIR):
    people.append(i)

#print(people)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('Face Detection and Recognition\Face Recognization/face_trained.yml')

img = cv2.imread(r'Face Detection and Recognition\Faces\val\ben_afflek/3.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv2.putText(img, str(people[label]), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv2.imshow('Detected Face', img)

cv2.waitKey(0)