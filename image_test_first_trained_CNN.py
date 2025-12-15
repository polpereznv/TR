import cv2
import numpy as np
from tensorflow import keras
from functions import facial_recognition_and_landmarks_locating

model = keras.models.load_model("final_emotion_recognition_cnn.keras")

emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Cara provisional per provar si la CNN llegeix 
image_path = "/home/polperez/Desktop/TR/cara.jpeg"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found. Check path.")
    exit()

img_resized = cv2.resize(img, (48,48))

landmarks = facial_recognition_and_landmarks_locating(img_resized, findRectangle=False, PrintFoto=True)

if landmarks is not None:
    img_input = img_resized.astype("float32").reshape(1,48,48,1) / 255.0
    landmarks_input = np.array(landmarks).squeeze(axis=0).reshape(1, -1)
    landmarks_input[:, 0::2] /= 48.0
    landmarks_input[:, 1::2] /= 48.0

    prediction = model.predict([img_input, landmarks_input])
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    print(f"Predicted emotion: {emotion_dict[predicted_class]} ({confidence:.2f}%)")
else:
    print("No face detected in the image.")
