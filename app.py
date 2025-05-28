import cv2
import numpy as np
import os
import sqlite3
import smtplib
import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load the face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Admin credentials stored in app.py
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "123"

# Email credentials (configure your email)
EMAIL_SENDER = "surekha17012002@gmail.com"
EMAIL_PASSWORD = "wujx wice wlnx gmsx"
EMAIL_RECEIVER = "surekha17012002@gmail.com"



# Log folder setup
LOG_FOLDER = "logs/"
UNKNOWN_FOLDER = os.path.join(LOG_FOLDER, "unknown/")
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(UNKNOWN_FOLDER, exist_ok=True)



# Home Page
@app.route('/')
def home():
    return render_template('home.html')

# Admin Login
@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin'] = username
            return redirect(url_for('admin_dashboard'))
        else:
            return "Invalid credentials! <a href='/admin'>Try Again</a>"
    return render_template('admin_login.html')


# View Database - Show access log
@app.route('/view_database')
def view_database():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    log_entries = []
    # Check all log folders (unknown + known/{name}/)
    log_folders = [UNKNOWN_FOLDER] + [
        os.path.join(LOG_FOLDER, "known", name)
        for name in os.listdir(os.path.join(LOG_FOLDER, "known")) if os.path.isdir(os.path.join(LOG_FOLDER, "known", name))
    ]
    for folder in log_folders:
        if not os.path.exists(folder):
            print(f"❌ ERROR: {folder} folder does not exist!")
            continue
        log_files = sorted(os.listdir(folder), reverse=True)  # Sort latest first
        print(f"📝 Found log files in {folder}: {log_files}")  # Debugging output
        for index, file in enumerate(log_files, start=1):
            if file.endswith(".jpg"):  # Process only images
                timestamp = file.replace(".jpg", "")

                # Extract person's name from folder path
                name = os.path.basename(folder)

                # ✅ Convert path to be Flask-compatible
                image_path = os.path.join(os.path.basename(folder), file)
                print(f"✅ Log Entry: ID={index}, Name={name}, Timestamp={timestamp}, Image={image_path}")
                print(f"🔍 Debug Path - Full Path: {os.path.abspath(image_path)}")

                log_entries.append((index, name, image_path, timestamp))  # Now contains 4 elements
    return render_template('view_database.html', logs=log_entries)





# Logout Route
@app.route('/logout')
def logout():
    session.pop('admin', None)  # Remove admin session
    return redirect(url_for('admin_login'))





# Admin Dashboard
@app.route('/dashboard')
def admin_dashboard():
    if 'admin' in session:
        return render_template('admin_dashboard.html')
    return redirect(url_for('admin_login'))





# Add Member - Capture Images & Train Model
@app.route('/add_member', methods=['GET', 'POST'])
def add_member():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    if request.method == 'POST':
        name = request.form['name']
        user_id = str(len(os.listdir('models/')) + 1)  # Unique ID

        capture_faces(name, user_id)  # Capture images
        train_model()  # Train the model

        return redirect(url_for('admin_dashboard'))

    return render_template('add_member.html')




# Function to capture images
import os


def capture_faces(name, user_id):
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if not cam.isOpened():
        print("❌ ERROR: Cannot access webcam.")
        return
    else:
        print("✅ Webcam accessed successfully!")

    folder_path = f'captured_images/{name}_{user_id}'
    os.makedirs(folder_path, exist_ok=True)

    print(f"📸 Capturing images for {name}, ID: {user_id}")

    count = 0
    while count < 60:  # Capture 60 images
        ret, frame = cam.read()
        if not ret:
            print("❌ ERROR: Cannot access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # **Speed Optimization:** Adjust detection parameters
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))

        if len(faces) == 0:
            print("❌ No faces detected! Move closer or adjust lighting.")
        else:
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]

                # **Speed Optimization:** Save image in memory first, then write to disk in bulk
                file_path = os.path.join(folder_path, f"{name}_{count}.jpg")
                cv2.imwrite(file_path, face)
                count += 1

        if count >= 60:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"🎉 Face capturing completed for {name}.")


# Train the LBPH model

import numpy as np

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images, labels = [], []

    dataset_path = 'captured_images/'

    if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
        print("❌ ERROR: No training images found! Check the capture process.")
        return

    for person_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, person_folder)
        if not os.path.isdir(folder_path):
            continue

        user_id = int(person_folder.split('_')[-1])

        for file in os.listdir(folder_path):
            if file.endswith('.jpg'):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"❌ ERROR: Failed to read {img_path}")
                    continue

                # **Ensure Consistent Image Size**
                img = cv2.resize(img, (200, 200))

                images.append(img)
                labels.append(user_id)

    if len(images) > 0:
        recognizer.train(images, np.array(labels))
        recognizer.save('models/trained_model.yml')
        print("✅ Model trained successfully!")

        if os.path.exists('models/trained_model.yml'):
            print("✅ trained_model.yml verified!")
        else:
            print("❌ ERROR: trained_model.yml was not saved properly!")

    else:
        print("❌ ERROR: No training images found!")




# User Authentication
@app.route('/user', methods=['GET'])
def user_auth():
    return render_template('user.html')





import sqlite3
import os

def is_user_registered(user_id):
    dataset_path = 'captured_images/'

    # Convert label (user_id) to a string
    user_id_str = str(user_id)

    # Loop through all folders in `captured_images/`
    for folder_name in os.listdir(dataset_path):
        if user_id_str in folder_name:  # Match label to folder name
            print(f"✅ User {user_id} found in captured_images!")
            return True

    print(f"⚠️ User {user_id} NOT found in captured_images!")
    return False


def get_person_name_from_label(user_id):
    dataset_path = "captured_images/"

    for folder_name in os.listdir(dataset_path):
        if folder_name.endswith(f"_{user_id}"):
            person_name = folder_name.rsplit('_', 1)[0]  # Extract the name before "_ID"
            return person_name

    return "Unknown"


@app.route('/detect_user', methods=['POST'])
def detect_user():
    model_path = 'models/trained_model.yml'

    if not os.path.exists(model_path):
        print("❌ ERROR: Model file not found! Train the model first.")
        return jsonify({"status": "error", "message": "Train the model first!"})

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cam.isOpened():
        print("❌ ERROR: Camera not accessible!")
        return jsonify({"status": "error", "message": "Camera not accessible!"})

    ret, frame = cam.read()
    cam.release()

    if not ret:
        print("❌ ERROR: Failed to capture frame!")
        return jsonify({"status": "error", "message": "Failed to capture frame!"})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))

    if len(faces) == 0:
        print("❌ No face detected!")
        return jsonify({"status": "error", "message": "No face detected!"})

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))  # Ensure consistent size

        try:
            label, confidence = recognizer.predict(face)
            print(f"🔍 Predicted Label: {label}, Confidence: {confidence}")

            if confidence < 80 and is_user_registered(label):
                person_name = get_person_name_from_label(label)
                print(f"✅ Recognized: {person_name} with confidence {confidence}")
                return jsonify({"status": "Authorized", "message": f"Door Unlocked for {person_name}!"})

            else:
                print(f"❌ Unrecognized Face (Confidence: {confidence})")
                timestamp = int(time.time())
                unknown_path = os.path.join(UNKNOWN_FOLDER, f"unknown_{timestamp}.jpg")
                cv2.imwrite(unknown_path, frame)

                if os.path.exists(unknown_path):
                    print(f"✅ Unknown image saved: {unknown_path}")
                    send_email(unknown_path)
                    return jsonify(
                        {"status": "Unauthorized", "message": "Unknown person detected. Email sent to admin."})
                else:
                    return jsonify({"status": "error", "message": "Failed to save unknown person's image!"})

        except Exception as e:
            print(f"❌ ERROR: Face prediction failed: {e}")
            return jsonify({"status": "error", "message": "Face prediction failed due to an exception."})



# Send Email for Unauthorized Access
def send_email(image_path):
    print("🔍 DEBUG: send_email() function called with image path:", image_path)

    try:
        print("📤 Connecting to SMTP server...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.set_debuglevel(1)  # ✅ Enable detailed SMTP logs
        server.starttls()

        print("🔑 Attempting to log in...")
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)  # ✅ Show login errors

        print("🚀 Preparing email...")
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = "🚨 Unauthorized Access Alert!"

        # Flask server URL (change if needed)
        SERVER_URL = "http://127.0.0.1:5000"

        # Email Body with Buttons
        body = f"""
        <h2>🚨 Unauthorized Person Detected!</h2>
        <p>A person was detected trying to access the system. See the attached image below.</p>

        <img src="cid:image1" width="300"><br><br>

        <a href="{SERVER_URL}/allow_access" style="display:inline-block; padding:10px 20px; font-size:16px; font-weight:bold; color:white; background-color:green; text-decoration:none; border-radius:5px;">✔ ALLOW</a>

        <a href="{SERVER_URL}/reject_access" style="display:inline-block; padding:10px 20px; font-size:16px; font-weight:bold; color:white; background-color:red; text-decoration:none; border-radius:5px; margin-left:10px;">✖ REJECT</a>
        """

        msg.attach(MIMEText(body, 'html'))  # ✅ HTML Email Body

        # **Attach the Image**
        if os.path.exists(image_path):
            print(f"✅ Attaching image: {image_path}")
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                image = MIMEImage(img_data, name=os.path.basename(image_path))
                image.add_header('Content-ID', '<image1>')  # ✅ Ensure inline display
                msg.attach(image)
        else:
            print(f"❌ ERROR: Image file not found at {image_path}")
            return  # Stop execution if image is missing

        print("📩 Sending email...")
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

        server.quit()
        print("✅ Email sent successfully!")

    except Exception as e:
        print(f"❌ SMTP ERROR: {e}")  # ✅ Now logs all email errors


# Function to capture video frames
import cv2
import time
from flask import Response

def generate_user_frames():
    cam_index = 0  # Force a single camera index (instead of [0, 1, 2])
    camera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print(f"❌ ERROR: Unable to access camera at index {cam_index}")
        return

    time.sleep(2)  # Allow the camera to warm up

    while True:
        if not camera.isOpened():  # Check if camera is still connected
            print("❌ ERROR: Camera disconnected! Retrying...")
            camera.release()
            return  # Exit function instead of looping infinitely

        success, frame = camera.read()
        if not success:
            print("❌ ERROR: Failed to read frame! Releasing camera and retrying...")
            camera.release()
            return  # Exit function to avoid an infinite loop

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("❌ ERROR: Failed to encode frame! Retrying...")
            break

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()
    print("🔄 Camera released after failure.")




# Route for video streaming
@app.route('/user_video_feed')
def user_video_feed():
    return Response(generate_user_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')





@app.route('/capture_images', methods=['POST'])
def capture_images():
    if 'admin' not in session:
        return jsonify({"status": "error", "message": "Unauthorized access"}), 403

    name = request.form.get('name')
    if not name:
        return jsonify({"status": "error", "message": "Name is required"}), 400

    user_id = str(len(os.listdir('models/')) + 1)  # Assign unique ID
    capture_faces(name, user_id)  # Capture images
    train_model()  # Train the model

    return jsonify({"status": "success", "message": "Images captured and model trained successfully!"})


from flask import send_from_directory, abort

@app.route('/logs/<path:filename>')
def serve_image(filename):
    logs_directory = os.path.abspath(LOG_FOLDER)  # Get absolute path to logs/
    image_path = os.path.join(logs_directory, filename)

    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image not found - {image_path}")
        return abort(404)

    print(f"✅ Serving image: {image_path}")
    return send_from_directory(logs_directory, filename)




from flask import jsonify

from flask import jsonify

access_response = None  # Global variable to store access response

@app.route('/allow_access')
def allow_access():
    global access_response
    print("✅ Door Unlocked by Admin!")
    access_response = "✅ Access Granted! The door has been unlocked By Admin."
    return jsonify({"status": "success", "message": access_response})

@app.route('/reject_access')
def reject_access():
    global access_response
    print("❌ Access Denied by Admin!")
    access_response = "❌ Access Denied! The door remains locked ."
    return jsonify({"status": "error", "message": access_response})

@app.route('/check_access_response')
def check_access_response():
    global access_response
    if access_response:
        response = jsonify({"message": access_response})
        access_response = None  # Reset response after sending
        return response
    return jsonify({})  # Empty response if no update






@app.route('/test_email')
def test_email():
    test_image_path = "logs/unknown/test.jpg"  # Ensure this image exists
    print(f"📩 Manually sending test email with image: {test_image_path}")
    send_email(test_image_path)
    return "Test email sent!"









if __name__ == '__main__':
    app.run(debug=True)
