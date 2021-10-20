import cv2
import face_recognition as fr
import numpy as np
import os
import shutil


def get_encoded_faces():
    encoded = {}
    for dirpath, dname, fname in os.walk('./faces'):
        for f in fname:
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jfif"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
    return encoded


def classify_faces(im):
    fac = get_encoded_faces()
    fac_encoded = list(fac.values())
    known_face_names = list(fac.keys())

    img = cv2.imread('Resources/' + im)
    face_location = fr.face_locations(img)
    unknown_encoding = fr.face_encodings(img, face_location)

    face_names = []
    for f_en in unknown_encoding:
        name = "Unknown"
        match = fr.compare_faces(fac_encoded, f_en)
        face_dist = fr.face_distance(fac_encoded, f_en)
        best_match = np.nanargmin(face_dist)
        if match[best_match]:
            name = known_face_names[best_match]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_location, face_names):
            # Rectangle around face
            cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (0, 0, 255), 1)

            # Label
            cv2.rectangle(img, (left - 20, bottom - 9), (right + 20, bottom + 20), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left - 20, bottom + 15), font, 0.5, (255, 255, 255), 1)

    while True:
        cv2.imshow('Video', img)
        cv2.waitKey(0)
        return face_names


def learning(image):
    c_face = classify_faces(image)
    if "Unknown" in c_face:
        print("Unknown face detected.")
        que = input("Would you like to add it the the database? \n Yes / No ?")
        if que == "yes" or que == 'Yes':
            img_name = input("Enter name for the detected face :")
            shutil.move('Resources/' + img_name, './faces')
            print(learning(image))
        else:
            print(c_face)
    else:
        for i in c_face:
            print(i)


# Main

x = input("Enter the name of image : ")
learning(x)