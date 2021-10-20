import cv2
import face_recognition

mydict = {}

#Trump

trump = face_recognition.load_image_file('Resources/trump.jpg')
trump_en = face_recognition.face_encodings(trump)[0]

mydict['Donald Trump'] = trump_en

#Obama

obama = face_recognition.load_image_file('Resources/obama.jfif')
obama_en = face_recognition.face_encodings(obama)[0]

mydict['Barrack Obama'] = obama_en

#Modi

modi = face_recognition.load_image_file('Resources/modi.jfif')
modi_en = face_recognition.face_encodings(modi)[0]

mydict['Narendra Modi'] = modi_en

#Rahul

rahul = face_recognition.load_image_file('Resources/rahul.jfif')
rahul_en = face_recognition.face_encodings(rahul)[0]

mydict['Rahul Gandhi'] = rahul_en

#Test


img = cv2.imread('Resources/moditru.jpg')
#img = cv2.resize(im, (500, 500))

unknown = face_recognition.load_image_file('Resources/moditru.jpg')
unknown_en = face_recognition.face_encodings(unknown)[0]

for key,value in mydict.items():
    if face_recognition.compare_faces([value],unknown_en)==[True]:
        print(key)

cv2.imshow("Image", img)
cv2.waitKey(0)