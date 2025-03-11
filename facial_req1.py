#! /usr/bin/python

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import csv
from datetime import datetime
import os

# Directory containing multiple pickle files with face encodings
encodings_dir = r"C:\Users\weika\OneDrive\Desktop\Degree Y2S3\IOT Assignment\face"

# Initialize the 'currentname' to trigger only when a new person is identified
currentname = "unknown"
all_data = []  # To store encodings from all pickle files

# Check and load all pickle files in the folder
if os.path.exists(encodings_dir):
    print("[INFO] Loading all face encodings from the 'face' directory...")
    for file_name in os.listdir(encodings_dir):
        if file_name.endswith(".pickle"):
            pickle_path = os.path.join(encodings_dir, file_name)
            print(f"[INFO] Loading encodings from {file_name}...")
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
                all_data.append(data)
else:
    print("[ERROR] The directory with pickle files was not found.")
    exit(1)

# Flatten all encodings and names from all pickle files
all_encodings = []
all_names = []

for data in all_data:
    all_encodings.extend(data["encodings"])
    all_names.extend(data["names"])

# Initialize the video stream and allow the camera sensor to warm up
vs = VideoStream(src=0, framerate=10).start()
time.sleep(2.0)

# Start the FPS counter
fps = FPS().start()

# Create or open the CSV file in append mode
csv_file = open("recognized_faces.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)

# Write the header if the file is empty
if csv_file.tell() == 0:
    csv_writer.writerow(["Name", "Timestamp"])

# Function to record the recognized person in real time and write to CSV
def record_to_csv(name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] Detected {name} at {timestamp}")
    csv_writer.writerow([name, timestamp])
    csv_file.flush()  # Ensure the record is written to the file immediately

# Lowered threshold for face distance to consider a match (more strict)
FACE_MATCH_THRESHOLD = 0.4  # You can adjust this value based on your testing

# Loop over frames from the video file stream
while True:
    frame = vs.read()

    if frame is None:
        print("[ERROR] Failed to grab frame from video stream")
        break

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=500)

    # Detect face locations
    boxes = face_recognition.face_locations(frame)

    # Compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    # Loop over the facial embeddings
    for encoding in encodings:
        # Compute distances between the detected face and all stored encodings
        face_distances = face_recognition.face_distance(all_encodings, encoding)
        best_match_index = None

        # Find the closest match
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()  # Find index of minimum distance
            best_distance = face_distances[best_match_index]

            # Secondary check using compare_faces to confirm the match
            matches = face_recognition.compare_faces(all_encodings, encoding, tolerance=0.5)
            name = "Unknown"  # Default to unknown

            # Only consider the face a match if the distance is below the threshold
            if matches[best_match_index] and best_distance <= FACE_MATCH_THRESHOLD:
                name = all_names[best_match_index]

        # Update the name if it's a new detection
        if currentname != name:
            currentname = name
            record_to_csv(name)

        names.append(name)

    # Loop over the recognized faces and draw boxes around them
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)

    # Display the image on the screen
    cv2.imshow("Facial Recognition is Running", frame)

    # Check for ESC (key = 27) or 'q' key to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        print("Exiting...")
        break

    # Update the FPS counter
    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Cleanup
csv_file.close()
cv2.destroyAllWindows()
vs.stop()
