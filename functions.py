import cv2
import csv
import sys
import numpy as np
from PIL import Image
from tkinter import Tk, filedialog


#Pixels is the 48x48 list of pixels that make the images
# The function identifies the face and locates the landmarks
def facial_recognition_and_landmarks_locating(pixels,findRectangle=True,PrintFoto=True):
    image = np.array(pixels).reshape(48, 48).astype(np.uint8)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("lbfmodel.yaml")
    faces = face_classifier.detectMultiScale(image, scaleFactor = 1.1, minNeighbors = 5)
    if findRectangle:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
    if len(faces) == 0:
        print("No face detected.")
        return None
    elif len(faces) == 1:
        print("1 face detected.")
    else:
        print(f"{len(faces)} faces were detected.")
    
    _, punts = facemark.fit(image, faces)

    for punt in punts:
        for (x, y) in punt[0]:
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1) 
    if PrintFoto:
        cv2.namedWindow("Punts facials: ", cv2.WINDOW_NORMAL)
        cv2.imshow("Punts facials: ", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return punts


# The function cuts the face to focus better in it
def crop_face(path):
    image_path = path
    imatge = cv2.imread(image_path)

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("/home/polperez/Desktop/TR/lbfmodel.yaml")

    grey_scale = cv2.cvtColor(imatge, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey_scale, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No faces detected")
        return None

    print(f"{len(faces)} face(s) detected.")
    x, y, w, h = faces[0]  # take the first face

    foto = Image.open(image_path)
    crop_area = (x, y, x + w, y + h)
    imatge_retallada = foto.crop(crop_area)

    cropped_face = cv2.cvtColor(np.array(imatge_retallada), cv2.COLOR_RGB2GRAY)
    return cropped_face

"""
# Same as before but when analyzing from webcam
# grey = grayscale numpy array from webcam
def crop_face_from_webcam(grey):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("/home/polperez/Desktop/TR/lbfmodel.yaml")

    faces = face_classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No faces detected")
        return None

    print(f"{len(faces)} face(s) detected.")

    x, y, w, h = faces[0]

    imatge_retallada = grey[y:y+h, x:x+w]
    cropped_face = cv2.cvtColor(np.array(imatge_retallada), cv2.COLOR_RGB2GRAY)
    return cropped_face
"""


# Open a window to select the image from the files
def select_image():
    image_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if not image_path:
        print("No image selected")
        return None
    return image_path