# Needed libraries for facial recognition 
import cv2
import sys
# sys = System-specific parameters and functions

# Get the image to detect directly from the terminal
image_name = sys.argv[1]

# Load the classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# Load image to detect
image = cv2.imread(image_name)

# Convert to grey scale
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Start detection
faces = face_classifier.detectMultiScale(grey, scaleFactor = 1.1, minNeighbors = 5)

# Draw rectangles x = x coordinates ; y = y coordinates ; w = width ; h = height
for (x, y, w, h) in faces:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Report if it identified a face directly in the terminal
if len(faces) == 0:
    print("No face detected in the image.")
elif len(faces) == 1:
    print("1 face detected.")
else:
    print(f"{len(faces)} faces detected.")

# Show results
cv2.imshow("Detected faces : ", image)

# Wait for a key to be pressed to close the image
cv2.waitKey(0)
cv2.destroyAllWindows()






