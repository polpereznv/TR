# Needed libraries for the process
import cv2
import sys

# Get the image to detect name from the terminal
image_name = sys.argv[1]

# Load the classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the function which locates the 68 landmarks
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("/home/polperez/Desktop/TR/lbfmodel.yaml")
# Load image to detect
image = cv2.imread(image_name)

# Convert to grey scale
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Start detection
faces = face_classifier.detectMultiScale(grey, scaleFactor = 1.1, minNeighbors = 5)

# Draw rectangles
for (x, y, w, h) in faces:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Report if it identified a face directly in the terminal
if len(faces) == 0:
    print("No face detected in the image.")
elif len(faces) == 1:
    print("1 face detected.")
else:
    print(f"{len(faces)} faces detected.")

# Detect the landmarks on the identified face
_, landmarks = facemark.fit(grey, faces)

# Draw landmarks
for landmark in landmarks:
    for (x, y) in landmark[0]:
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1) 

# Make the image window resizable to see easier
cv2.namedWindow("Facial landmarks: ", cv2.WINDOW_NORMAL)
# Show results
cv2.imshow("Facial landmarks: ", image)

# Wait for a key to be pressed to close the image
cv2.waitKey(0)
cv2.destroyAllWindows()

