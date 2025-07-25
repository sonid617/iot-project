from flask import Flask, render_template, request, jsonify, Response
import cv2
import face_recognition
import pyttsx3
import numpy as np
import time
import os

app = Flask(__name__)

# === Voice Setup ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    print(f"🔊 Speaking: {text}")
    engine.say(text)
    engine.runAndWait()

# === Load All Known Faces ===
known_face_encodings = []
known_face_names = []

known_faces_dir = "known_faces"
print("📂 Loading known faces...")

for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        name = os.path.splitext(filename)[0]
        filepath = os.path.join(known_faces_dir, filename)
        
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
            print(f"✅ Loaded face for: {name}")
        else:
            print(f"❌ No face found in {filename}")

if not known_face_encodings:
    speak("No known faces found.")
    exit()

# === Camera Setup ===
camera = cv2.VideoCapture(0)

# === Flask Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/doorbell', methods=['POST'])
def doorbell():
    ret, frame = camera.read()
    if not ret or frame is None:
        speak("Camera error.")
        return jsonify({'message': '❌ Camera error.'})

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if not face_locations:
        speak("No face detected. Please stand properly.")
        return jsonify({'message': '😕 No face detected.'})

    try:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    except Exception:
        speak("Face encoding failed.")
        return jsonify({'message': '⚠️ Face encoding error.'})

    door_open = False
    name = "Unknown"

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, encoding)
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
            door_open = True
            break

    if door_open:
        speak(f"Face recognized. Door is open. Welcome {name}.")
        return jsonify({'message': f'✅ Face recognized. 🚪 Door is OPEN. Welcome {name}.'})
    else:
        speak("Access denied. Please state your purpose.")
        return jsonify({'message': '❌ Face not recognized. 🚪 Door is CLOSED.'})

@app.route('/speak', methods=['POST'])
def speak_to_person():
    speak("Hello. Please state your purpose.")
    return jsonify({'message': '🗣️ Speaking to the person...'})

@app.route('/open-door', methods=['POST'])
def open_door():
    speak("Door is now open.")
    return jsonify({'message': '🚪 Door opened manually.'})

@app.route('/take-picture', methods=['POST'])
def take_picture():
    ret, frame = camera.read()
    if ret:
        timestamp = int(time.time())
        filename = f"captured_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return jsonify({'message': f'📸 Picture taken and saved as {filename}'})
    return jsonify({'message': '❌ Failed to capture image.'})

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(host=<'your_ip'>, port=5000, debug=True)
