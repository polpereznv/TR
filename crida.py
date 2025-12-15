#import pdb; pdb.set_trace()
import cv2

image_path = "/home/polperez/Desktop/TR/cara.jpeg"
imatge = cv2.imread(image_path)
#cv2.imshow("foto prova",imatge)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("/home/polperez/Desktop/TR/lbfmodel.yaml")
grey_scale = cv2.cvtColor(imatge, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(grey_scale, scaleFactor = 1.1, minNeighbors = 5)
if len(faces) == 0:
    print("No faces detected")
else:
    print(f"{len(faces)} face(s) detected.")
    for (x, y, w, h) in faces:
        cv2.rectangle(grey_scale, (x, y), (x + w, y + h), (0, 255, 0), 2)

from PIL import Image
import numpy as np

foto = Image.open(image_path)
crop_area = (x, y, x + w, y + h)
imatge_retallada = foto.crop(crop_area)
imatge_retallada.save("imatge_retallada.jpeg")

cropped_face = cv2.cvtColor(np.array(imatge_retallada), cv2.COLOR_BGR2GRAY)

cv2.imshow("foto prova",cropped_face)
cv2.waitKey(0)
cv2.destroyAllWindows()

