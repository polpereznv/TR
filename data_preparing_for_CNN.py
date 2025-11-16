#import pdb; pdb.set_trace()
import numpy as np
import cv2
import csv
import sys
from functions import facial_recognition_and_landmarks_locating
import tensorflow
from tensorflow.keras import layers, models
from tensorflow.keras.models import model_from_json
import keras
from functions import select_image
from functions import crop_face
import matplotlib.pyplot as plt
import time
from functions import crop_face_from_webcam
import pickle

whattodo = input("Train (T) or Test (R)  ")

if whattodo == "T":
    # Timer
    start_total_time = time.time()
    # Prepare lists
    pixels_byw = []
    landmarks_coordinates = []
    name_labels = []

    # Load the classifiers
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("lbfmodel.yaml")
    
    # Read the database training document
    with open("train.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)
        a = 0
        # Get the pixels and landmarks coordinates of every face in the database
        for line in file:
            a += 1
            print(a)
            labels, pixel_list = line.strip().split(",", 1)  # split into 2 parts
            pixels = pixel_list.split()
            pixels_array = np.array([int(x.replace('"','')) for x in pixels], dtype="float32")  
            label = int(labels)
            img = pixels_array.reshape(48,48)   # reshape flattened pixels
            coordinates = facial_recognition_and_landmarks_locating(img,findRectangle=False,PrintFoto=False)
            if coordinates is not None:
                pixels_byw.append(img)
                name_labels.append(label)
                landmarks_coordinates.append(coordinates)
            

        print(len(name_labels))
        #print(type(name_labels))


    # Here it goes the CNN

    # Convert list of pixels, landmarks and emotion labels to numpy.arrays
    training_images = np.array(pixels_byw, dtype="float32")
    training_landmarks = np.array(landmarks_coordinates, dtype="float32")
    training_labels = np.array(name_labels, dtype="int32")

    # Name the emotions 
    emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}
    # Count how many times each emotion was detected
    unique, counts = np.unique(training_labels, return_counts=True)

    for label, count in zip(unique, counts):
        emotion = emotion_labels.get(label, "Unknown")
        print(f"{emotion} ({label}) occurred {count} times")


    #Make sure it worked
    #print(type(training_images))
    #print(type(training_landmarks))
    #print(type(training_labels))

    # Normalization
    
    training_images = training_images.reshape(-1 , 48, 48, 1) / 255.0

    training_landmarks = training_landmarks.squeeze(axis=1)
    training_landmarks = training_landmarks.reshape(len(training_landmarks), -1)

    training_landmarks[:, 0::2] /= 48.0 # x
    training_landmarks[:, 1::2] /= 48.0 # y

    #Make sure it worked
    #print(training_landmarks)

    #num_classes = len(np.unique(training_labels)) #Això ho faré servir despres en entrenar tota la base de dades, ja que hi ha alguna emoció que no apareix en les primeres cares que agafo de mostra
    num_classes = 7
    #print(num_classes)

    # CNN
    
    image_input = layers.Input(shape=(48, 48, 1))

    x = layers.Conv2D(32, (3,3), activation='relu')(image_input)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Flatten()(x)

    x = layers.Dense(128, activation='relu')(x)

    landmark_input = layers.Input(shape=(136,))

    y = layers.Dense(128, activation='relu')(landmark_input)
    y = layers.Dense(64, activation='relu')(y)

    combined = layers.concatenate([x, y])

    z = layers.Dense(128, activation='relu')(combined)
    output = layers.Dense(num_classes, activation='softmax')(z)

    model = models.Model(inputs=[image_input, landmark_input], outputs=output)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    start_epochs_time = time.time()
    history = model.fit(
        [training_images, training_landmarks],
        training_labels,
        batch_size=32,
        epochs=20,
        validation_split=0.2
    )
    
    with open("final_model_history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    print("Training history saved to final_model_history.pkl")

    end_epochs_time = time.time()
    end_total_time = time.time()
    total_training_time = end_total_time - start_total_time
    total_epochs_time = end_epochs_time - start_epochs_time
    time_per_epoch = total_epochs_time / 20  # 20 epochs

    print(f"\nTotal training time: {total_training_time:.3f} seconds")
    print(f"\nmodel.fit() time: {total_epochs_time:.3f} seconds")
    print(f"\nAverage time per epoch: {time_per_epoch:.3f} seconds")

    model.save("final_emotion_recognition_cnn.keras")
    print("model saved")

    #Confusion matrix
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sn
    import pandas as pd
    from sklearn.metrics import confusion_matrix, classification_report

    # Step 1. Make predictions
    predictions = model.predict([training_images, training_landmarks])
    predicted_labels = np.argmax(predictions, axis=1)

    # Step 2. Build confusion matrix
    cm = confusion_matrix(training_labels, predicted_labels)

    # Step 3. Define your emotion labels
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    # Step 4. Ensure confusion matrix has full 7x7 shape
    num_classes = 7
    if cm.shape != (num_classes, num_classes):
        fixed_cm = np.zeros((num_classes, num_classes), dtype=int)
        fixed_cm[:cm.shape[0], :cm.shape[1]] = cm
        cm = fixed_cm

    # Step 5. Convert to DataFrame for plotting
    df_cm = pd.DataFrame(cm, index=emotion_labels, columns=emotion_labels)

    # Step 6. Plot the heatmap
    plt.figure(figsize=(8, 6))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix of CNN Emotion Classifier")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.show()

    # Step 7. Print the classification report
    print("\nClassification Report:")
    # Step 7 (fixed): Print classification report safely
    unique_labels = np.unique(training_labels)
    used_emotion_labels = [emotion_labels[i] for i in unique_labels]

    print("\nClassification Report:")
    print(classification_report(
        training_labels,
        predicted_labels,
        labels=unique_labels,
        target_names=used_emotion_labels
    ))
    from sklearn.metrics import roc_auc_score, RocCurveDisplay
    from sklearn.preprocessing import label_binarize

    # Binarize the labels for multi-class ROC
    y_true_bin = label_binarize(training_labels, classes=np.arange(num_classes))
    y_pred_bin = predictions  # model.predict output = probability for each class

    # Compute the mean AUC-ROC score (macro average)
    auc_score = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovr')
    print(f"\n Mean AUC-ROC score (macro-avg): {auc_score:.4f}")

    # OPTIONAL: Plot ROC curve for one example class (e.g., 'Happy')
    class_to_plot = 3  # change index if you want another class
    RocCurveDisplay.from_predictions(y_true_bin[:, class_to_plot], y_pred_bin[:, class_to_plot])
    plt.title(f"ROC Curve – {emotion_labels[class_to_plot]} (Class {class_to_plot})")
    plt.show()




    # Accuracy graph
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # Loss graph
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


elif whattodo == "R":
    
    model = keras.models.load_model("/home/polperez/Desktop/final_emotion_recognition_cnn.keras")
    model.compile()
    camera_or_saved = input("Foto from files (F) or take a picture (P)")
    if camera_or_saved == "F":
        path = select_image()
        cropped_face = crop_face(path)
        if cropped_face is not None:
            cv2.imshow("Cropped Face", cropped_face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        resized_cropped_face = cv2.resize(cropped_face, (48,48))
        cv2.imshow("resized_cropped_face", resized_cropped_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        landmarks = facial_recognition_and_landmarks_locating(resized_cropped_face, findRectangle=False, PrintFoto=True)
        
        emotion_dict = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral"
    }
        if landmarks is not None:
            img_input = resized_cropped_face.astype("float32").reshape(1,48,48,1) / 255.0
            landmarks_input = np.array(landmarks).squeeze(axis=0).reshape(1, -1)
            landmarks_input[:, 0::2] /= 48.0
            landmarks_input[:, 1::2] /= 48.0

            start_pred = time.time()  #  Start measuring prediction time
            prediction = model.predict([img_input, landmarks_input])
            end_pred = time.time()  #  End measurement

            prediction_time = end_pred - start_pred

            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            print(f"Predicted emotion: {emotion_dict[predicted_class]} ({confidence:.3f}%)")
            print(f" Time to predict this emotion: {prediction_time:.4f} seconds")

        else:
            print("No face detected in the image.")



    elif camera_or_saved == "P":
        # Load model
        model = keras.models.load_model("/home/polperez/Desktop/final_emotion_recognition_cnn.keras")
        model.compile()

        # Open webcam 
        cap = cv2.VideoCapture(0)  # 0 = default webcam
        if not cap.isOpened():
            print("Error: Cannot access camera.")
            sys.exit()

        print("Press SPACE to take a photo, or ESC to cancel.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            cv2.imshow("Camera - Press SPACE to capture", frame)
            key = cv2.waitKey(1)

            # ESC = cancel
            if key % 256 == 27:
                print("Capture cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()

            # SPACE = take picture
            elif key % 256 == 32:
                img = frame.copy()
                print("Photo captured!")
                break

        # --- Close camera ---
        cap.release()
        cv2.destroyAllWindows()

        # --- Convert to grayscale ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Detect and crop face ---
        cropped_face = crop_face_from_webcam(gray)
        if cropped_face is None:
            print("No face detected.")
            sys.exit()

        else:
            # --- Resize face to 48x48 for CNN input ---
            resized_cropped_face = cv2.resize(cropped_face, (48, 48))
            cv2.imshow("Detected Face", resized_cropped_face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # --- Get landmarks ---
            landmarks = facial_recognition_and_landmarks_locating(
                resized_cropped_face,
                findRectangle=False,
                PrintFoto=True
            )

        # --- Emotion labels dictionary ---
        emotion_dict = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral"
        }

        # --- If landmarks found, predict emotion ---
        if landmarks is not None:
            # Prepare image input
            img_input = resized_cropped_face.astype("float32").reshape(1, 48, 48, 1) / 255.0

            # Prepare landmark input
            landmarks_input = np.array(landmarks).squeeze(axis=0).reshape(1, -1)
            landmarks_input[:, 0::2] /= 48.0  # normalize x
            landmarks_input[:, 1::2] /= 48.0  # normalize y

            # --- Measure prediction time ---
            start_pred = time.time()
            prediction = model.predict([img_input, landmarks_input])
            end_pred = time.time()

            prediction_time = end_pred - start_pred

            # --- Process results ---
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            print(f"\nPredicted emotion: {emotion_dict[predicted_class]} ({confidence:.3f}%)")
            print(f"Time to predict this emotion: {prediction_time:.4f} seconds")

        else:
            print("No landmarks detected in the image.")