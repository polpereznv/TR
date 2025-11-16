
import cv2
import csv
import numpy as np


def llegir_training(file_="train.csv"):
    #Llistes buides per emmagatzemar les dades
    Emotions = []
    Pixels = []

    with open(file_, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            Emotions.append(line['emotion'])
            #La llista deixa de ser string i se separa cada pixel i es passa a integer
            aux = [int(x) for x in line['pixels'].split(' ')]
            Pixels.append(aux)
    return Emotions, Pixels

    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    # per cada linea del fitxer
    #guardar el emotion,pixels,Usage
    #pixels --> array[48*48]
    # return result