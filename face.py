import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os

# Paths
desktop_path = r"C:\Users\sayya\OneDrive\Desktop\python"
csv_folder_path = desktop_path  # Save CSV in the same folder

# Video capture initialization
video_capture = cv2.VideoCapture(0)

# Load known face encodings and metadata
def load_face_data(image_path, name, id_number, section):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    return encoding, name, id_number, section

face_data = [
    load_face_data(os.path.join(desktop_path, "tata.jpg"), "Ratan Tata", 101, 1),
    load_face_data(os.path.join(desktop_path, "elon1.jpg"), "Elon Musk", 102, 2),
    load_face_data(os.path.join(desktop_path, "tesla.jpg"), "Tesla", 103, 1),
    load_face_data(os.path.join(desktop_path, "sameer.jpg"), "Sameer", 269, 4),
]

known_face_encodings = [data[0] for data in face_data]
known_faces_names = [data[1] for data in face_data]
known_faces_ids = [data[2] for data in face_data]
known_faces_sections = [data[3] for data in face_data]

# Copy names for attendance tracking
students = known_faces_names.copy()

# Create CSV file for attendance
current_date = datetime.now().strftime("%Y-%m-%d")
csv_file_path = os.path.join(csv_folder_path, f"output_{current_date}.csv")

with open(csv_file_path, 'w+', newline='') as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(['Name', 'ID', 'Section', 'Time'])

    # Start video processing
    while True:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Resize and process frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find faces and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=1)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
                id_number = known_faces_ids[best_match_index]
                section = known_faces_sections[best_match_index]
                face_names.append(name)

                if name in students:
                    students.remove(name)
                    lnwriter.writerow([name, id_number, section, current_time])

        # Draw rectangles and labels on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{name} - ID: {id_number} - Section: {section} - Time: {current_time}",
                (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
            )

        # Display the frame
        cv2.imshow("Attendance System", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
video_capture.release()
cv2.destroyAllWindows()

print(f"Attendance data saved in {csv_file_path}")
