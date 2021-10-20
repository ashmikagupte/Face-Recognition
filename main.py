import cv2
import face_recognition
import numpy
im = cv2.imread('Resources/diana.jpg')
img = cv2.resize(im, (500, 600))
img1 = face_recognition.load_image_file('Resources/diana.jpg')
loc = face_recognition.face_locations(img)
#print(loc)
cv2.rectangle(img, (236, 64), (390, 219), (0,0,255), 1)
cv2.putText(img, 'Diana', (236,240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1)
features = face_recognition.face_landmarks(img)
#print(features)
known = face_recognition.load_image_file('Resources/diana.jpg')
unknown = face_recognition.load_image_file('Resources/diana3.jfif')
known_en = face_recognition.face_encodings(known)[0]
unknown_en = face_recognition.face_encodings(unknown)[0]
results = face_recognition.compare_faces([known_en],unknown_en)
print(results)
cv2.imshow("Image", img)
cv2.waitKey(0)