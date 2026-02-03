from flask import Flask, render_template, request, redirect
import cv2
import os
import numpy as np
from PIL import Image

app = Flask(__name__)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

DATASET_PATH = "dataset"
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# --- Capture face for new user ---
def capture_face(user_id, num_samples=30):
    cam = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(DATASET_PATH, f"user.{user_id}.{count}.jpg"), face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) == 27 or count >= num_samples:
            break

    cam.release()
    cv2.destroyAllWindows()

# --- Train the recognizer ---
def train_model():
    imagePaths = [os.path.join(DATASET_PATH,f) for f in os.listdir(DATASET_PATH) if f.endswith(".jpg")]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        img = Image.open(imagePath).convert('L')
        img_numpy = np.array(img, 'uint8')
        filename = os.path.split(imagePath)[-1]
        try:
            id = int(filename.split(".")[1])
        except:
            continue
        faces = faceCascade.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    if len(faceSamples) > 0:
        recognizer.train(faceSamples, np.array(ids))
        recognizer.save("trainer.yml")

# --- Verify face ---
def verify_face():
    cam = cv2.VideoCapture(0)
    access_result = "NO FACE DETECTED"

    # Give user ~5 seconds to position themselves
    start_time = cv2.getTickCount()
    timeout = 5  # seconds
    fps = cv2.getTickFrequency()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 60:
                access_result = "ACCESS GRANTED"
            else:
                access_result = "ACCESS DENIED"

            # Draw rectangle (optional)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cam.release()
            cv2.destroyAllWindows()
            return access_result

        # Stop after timeout seconds
        elapsed = (cv2.getTickCount() - start_time)/fps
        if elapsed > timeout:
            break

        cv2.imshow("Position Yourself", frame)
        if cv2.waitKey(1) == 27:  # ESC to exit early
            break

    cam.release()
    cv2.destroyAllWindows()
    return access_result


# --- Routes ---
@app.route('/', methods=['GET','POST'])
def index():
    message = ""
    if request.method == "POST":
        action = request.form['action']
        if action == "register":
            user_id = request.form['user_id']
            if not user_id.isdigit():
                message = "User ID must be numeric"
            else:
                capture_face(user_id)
                train_model()
                message = f"User {user_id} registered successfully!"
        elif action == "verify":
            if os.path.exists("trainer.yml"):
                recognizer.read("trainer.yml")
                message = verify_face()
            else:
                message = "No trained data found. Please register a user first."

    return render_template("index.html", message=message)

if __name__ == "__main__":
    app.run(debug=True)
