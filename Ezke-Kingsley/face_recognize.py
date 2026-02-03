import cv2
import time

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)

access_decided = False

while not access_decided:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 60:
            label = "ACCESS GRANTED"
            color = (0, 255, 0)
        else:
            label = "ACCESS DENIED"
            color = (0, 0, 255)

        # Draw rectangle and text
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Show frame
        cv2.imshow("Face Access Control", img)
        print(label)  # optional: console output

        # Keep window open for 3 seconds so user can see
        cv2.waitKey(3000)

        access_decided = True
        break

# Release camera and close window
cam.release()
cv2.destroyAllWindows()
