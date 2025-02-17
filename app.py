import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)
nimgs = 300  # Number of images to capture per user
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time\n')


def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception as e:
        print(f"Error extracting faces: {e}")
        return []


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces, labels = [], []
    userlist = os.listdir('static/faces')

    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)

    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


def extract_attendance():
    df = pd.read_csv(attendance_file)
    return df['Name'], df['Roll'], df['Time'], len(df)


def add_attendance(name):
    parts = name.split('_')
    if len(parts) < 2:
        print(f"Invalid name format: {name}")
        return

    username = "_".join(parts[:-1])  # Join everything except the last part
    userid = parts[-1]

    if not userid.isdigit():
        print(f"⚠️ Invalid userid: {userid}")
        return

    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(attendance_file)

    if int(userid) not in df['Roll'].astype(int).tolist():
        with open(attendance_file, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


def getallusers():
    userlist = os.listdir('static/faces')
    names, rolls = [], []
    for user in userlist:
        parts = user.split('_')
        names.append("_".join(parts[:-1]))
        rolls.append(parts[-1])
    return userlist, names, rolls, len(userlist)


def deletefolder(duser):
    for file in os.listdir(duser):
        os.remove(os.path.join(duser, file))
    os.rmdir(duser)


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=datetoday2, mess="⚠️ Train the model first.")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            print(f"🆔 Identified Person: {identified_person}")
            add_attendance(identified_person)
            cv2.putText(frame, identified_person, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return home()


if __name__ == '__main__':
    app.run(debug=True)
