
import cv2
import os
import smtplib
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import sys
from twilio.rest import Client
import time

# Twilio Credentials (Replace these with your actual credentials)
TWILIO_ACCOUNT_SID = "twilio acc id"
TWILIO_AUTH_TOKEN = "twilio auth token"
TWILIO_PHONE_NUMBER = "twilio phone number"

# Email Credentials (Replace with actual credentials)
EMAIL_ADDRESS = "host email address"
EMAIL_PASSWORD = "app password"

# Initialize Twilio Client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize Flask App
app = Flask(__name__)

nimgs = 300  # Number of images to capture per user

# Get today's date
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Load Face Detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ensure necessary directories exist
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

# Create today's attendance file if not exists
attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time\n')

# Function to send email
def send_email(recipient_email, username):
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            subject = "Attendance Marked Successfully"
            body = f"Hello {username}, your attendance has been marked successfully."
            message = f"Subject: {subject}\n\n{body}"
            server.sendmail(EMAIL_ADDRESS, recipient_email, message)
            print(f"✅ Email sent to {recipient_email}!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# Function to send SMS
def send_sms(phone_number, username):
    try:
        print(f"📲 Sending SMS to {phone_number}...")
        message = client.messages.create(
            body=f"Hello {username}, your attendance has been marked successfully.",
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        print(f"✅ SMS sent successfully! SID: {message.sid}")
    except Exception as e:
        print(f"❌ Failed to send SMS: {e}")

# Add Attendance of a specific user
def add_attendance(name):
    try:
        username, userid = name.rsplit('_', 1)
    except ValueError:
        print(f"⚠️ Error: Could not split {name} correctly!")
        return

    if not userid.isdigit():
        print(f"⚠️ Invalid userid: {userid}")
        return

    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(attendance_file)

    if int(userid) not in list(df['Roll']):
        print(f"📌 Adding {username} ({userid}) to attendance...")

        with open(attendance_file, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

        phone_number = "actual phone number"  # Change this to the actual number
        recipient_email = "actual email"  # Replace with actual email
        print(f"📲 Attempting to send SMS to {phone_number} for {username}...")
        send_sms(phone_number, username)
        send_email(recipient_email, username)
    else:
        print(f"✅ {username} ({userid}) is already marked present. Skipping notifications.")

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=len(os.listdir('static/faces')), datetoday2=datetoday2)


@app.route('/start')
def start():
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        if time.time() - start_time > 60:
            print("⏳ Timeout reached. Closing camera...")
            break

        ret, frame = cap.read()
        if not ret:
            print("⚠️ Error: Failed to read frame!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            model = joblib.load('static/face_recognition_model.pkl')
            identified_person = model.predict(face.reshape(1, -1))[0]
            print(f"🆔 Identified Person: {identified_person}")
            add_attendance(identified_person)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Attendance process completed."

@app.route('/add', methods=['POST'])
def add():
    return "Adding new user..."  # Placeholder response

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
