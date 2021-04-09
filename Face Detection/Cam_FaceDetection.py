import cv2

haar_cascade = cv2.CascadeClassifier('Face Detection and Recognition\haar_face.xml')

cam = cv2.VideoCapture(0) # For the local videos change the '0' with video path
while cam.isOpened():
    isTrue, frame = cam.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    for (x,y,w,h) in faces_rect:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(25) & 0xFF==ord('d'):
        break

cam.release()        
cv2.destroyAllWindows()