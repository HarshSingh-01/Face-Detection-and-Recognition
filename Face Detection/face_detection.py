import cv2

img = cv2.imread('Face Detection and Recognition\Faces\lady.jpg')
cv2.imshow("Lady", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray People',gray)

haar_cascade = cv2.CascadeClassifier('Face Detection and Recognition\haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f'Nmber of faces = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv2.imshow('Detected Faces', img)

cv2.waitKey(0)