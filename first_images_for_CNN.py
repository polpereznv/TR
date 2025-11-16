import numpy as np
import cv2
import csv

n = 1
# Prepare lists
pixels_byw = []
landmarks_coordinates = []
name_labels = []

with open("train.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)

    for line in file:
        labels, pixel_list = line.strip().split(",", 1)  # split into 2 parts
        pixels = pixel_list.split()
        pixels_array = np.array([int(x.replace('"','')) for x in pixels], dtype="float32")  
        label = int(labels)
        img = pixels_array.reshape(48,48)   # reshape flattened pixels
        pixels_byw.append(img)
        name_labels.append(label)
        n = n + 1

        if n >= 10:
        	break
    print(pixels_byw[1])
    print(name_labels[1])