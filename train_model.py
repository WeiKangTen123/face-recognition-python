# #! /usr/bin/python

# # import the necessary packages
# from imutils import paths
# import face_recognition
# #import argparse
# import pickle
# import cv2
# import os

# # our images are located in the dataset folder
# print("[INFO] start processing faces...")
# imagePaths = list(paths.list_images("dataset"))

# # initialize the list of known encodings and known names
# knownEncodings = []
# knownNames = []

# # loop over the image paths
# for (i, imagePath) in enumerate(imagePaths):
# 	# extract the person name from the image path
# 	print("[INFO] processing image {}/{}".format(i + 1,
# 		len(imagePaths)))
# 	name = imagePath.split(os.path.sep)[-2]

# 	# load the input image and convert it from RGB (OpenCV ordering)
# 	# to dlib ordering (RGB)
# 	image = cv2.imread(imagePath)
# 	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 	# detect the (x, y)-coordinates of the bounding boxes
# 	# corresponding to each face in the input image
# 	boxes = face_recognition.face_locations(rgb,
# 		model="hog")

# 	# compute the facial embedding for the face
# 	encodings = face_recognition.face_encodings(rgb, boxes)

# 	# loop over the encodings
# 	for encoding in encodings:
# 		# add each encoding + name to our set of known names and
# 		# encodings
# 		knownEncodings.append(encoding)
# 		knownNames.append(name)

# # dump the facial encodings + names to disk
# print("[INFO] serializing encodings...")
# data = {"encodings": knownEncodings, "names": knownNames}
# f = open("encodings.pickle", "wb")
# f.write(pickle.dumps(data))
# f.close()


#! /usr/bin/python

# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os

# Path to the dataset folder
dataset_path = r"C:\Users\weika\OneDrive\Desktop\Degree Y2S3\IOT Assignment\dataset"

# Start processing images from the dataset folder
print("[INFO] start processing faces...")
imagePaths = list(paths.list_images(dataset_path))

# Initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# Loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # Extract the person name from the image path (the folder name)
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # Load the input image and convert it from BGR (OpenCV ordering) to RGB (dlib ordering)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the face locations in the image using HOG model
    boxes = face_recognition.face_locations(rgb, model="hog")

    # Compute the facial embeddings for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Loop over the encodings
    for encoding in encodings:
        # Add each encoding + name to our known names and encodings list
        knownEncodings.append(encoding)
        knownNames.append(name)

# Serialize the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Done processing and encoding!")
