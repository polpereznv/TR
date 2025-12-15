# Code for converting the normal image to b&w 
# to add the picture to the document
import cv2

image_path = ("/home/polperez/Desktop/TR/imatge_retallada.jpeg")
imatge = cv2.imread(image_path)
grey_scale = cv2.cvtColor(imatge, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original Face", grey_scale)
cv2.waitKey(0)
cv2.destroyAllWindows()