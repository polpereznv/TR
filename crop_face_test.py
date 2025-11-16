"""
import cv2
import numpy as np
from PIL import Image
from functions import crop_face



pathh = "/home/polperez/Desktop/woman_face.jpeg"
cropped_face = crop_face(pathh)
if cropped_face is not None:
    cv2.imshow("foto prova", cropped_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""

import cv2
import numpy as np
from PIL import Image
from tkinter import Tk, filedialog

def crop_face():
    # Open file dialog
    image_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg")]
    )

    if not image_path:
        print("No image selected.")
        return None

    imatge = cv2.imread(image_path)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("/home/polperez/Desktop/lbfmodel.yaml")

    grey_scale = cv2.cvtColor(imatge, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey_scale, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No faces detected")
        return None

    print(f"{len(faces)} face(s) detected.")
    x, y, w, h = faces[0]

    foto = Image.open(image_path)
    crop_area = (x, y, x + w, y + h)
    imatge_retallada = foto.crop(crop_area)
    cropped_face = cv2.cvtColor(np.array(imatge_retallada), cv2.COLOR_RGB2GRAY)
    return cropped_face


# Main
cropped_face = crop_face()
if cropped_face is not None:
    cv2.imshow("Cropped Face", cropped_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
