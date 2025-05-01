import cv2
import face_recognition
import os
import csv
import uuid
from datetime import datetime

# Initialize known faces database
known_face_encodings = []
known_face_names = []
known_face_ids = {}

# Load faces from 'faces' directory
faces_dir = "faces"
for filename in os.listdir(faces_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        name = os.path.splitext(filename)[0]
        unique_id = str(uuid.uuid4())[:8]  # 8-char ID
        
        image = face_recognition.load_image_file(os.path.join(faces_dir, filename))
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
            known_face_ids[name] = unique_id
            print(f"Registered: {name} (ID: {unique_id})")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Create CSV with headers
os.makedirs("attendance", exist_ok=True)
csv_path = "attendance\\attendance_records.csv"  # NEW: Single consolidated file

# NEW: Write headers if file doesn't exist
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Name", "ID", "Time"])  # NEW: Added Date column

recognized_faces = set()

try:
    while True:
        ret, frame = video_capture.read()
        if not ret: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            user_id = "N/A"

            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
                user_id = known_face_ids[name]
                
                if name not in recognized_faces:
                    recognized_faces.add(name)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    current_date = datetime.now().strftime("%Y-%m-%d")  # NEW: Get current date
                    with open(csv_path, 'a', newline='') as file:
                        csv.writer(file).writerow([current_date, name, user_id, current_time])  # NEW: Added date
                    print(f"Logged: {current_date} | {name} | ID: {user_id} | Time: {current_time}")

            # Display box and info
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({user_id})", (left+6, bottom-6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    video_capture.release()
    cv2.destroyAllWindows()