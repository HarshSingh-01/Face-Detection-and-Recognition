import os
import cv2
import numpy as np

DIR = r'Face Detection and Recognition\Faces\train'
people = []
for i in os.listdir(DIR):
    people.append(i)

#print(people)

haar_cascade = cv2.CascadeClassifier('Face Detection and Recognition\haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv2.imread(img_path)

            if img_array is None:
                continue

            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done -----------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the Reconizer on the features list and the labels list
face_recognizer.train(features, labels)

save_path = 'Face Detection and Recognition\Face Recognization'
face_recognizer.save(save_path + '/' + 'face_trained.yml')
np.save(save_path + '/' + 'features.npy', features)
np.save(save_path + '/' + 'labels.npy', labels)