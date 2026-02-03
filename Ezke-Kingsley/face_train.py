import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(".jpg")]
    faceSamples=[]
    ids=[]

    for imagePath in imagePaths:
        gray_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(gray_img,'uint8')
        # Extract ID safely
        filename = os.path.split(imagePath)[-1]  # e.g., user.1.1.jpg
        try:
            id = int(filename.split(".")[1])
        except:
            continue  # skip files that do not match pattern

        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids


faces, ids = getImagesAndLabels("dataset")
recognizer.train(faces, np.array(ids))
recognizer.save("trainer.yml")

print("Training complete")
