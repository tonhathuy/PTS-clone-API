
from pyimagesearch.face_blurring import anonymize_face_pixelate
import numpy as np
import argparse
import cv2
import os


def blur(img_name, load_folder, save_folder):
    load_path = os.path.join(load_folder, img_name)
    print(load_path)
    save_path = os.path.join(save_folder, img_name)

    print("[INFO] loading face detector model...")
    prototxtPath = "face_detector/deploy.prototxt"
    weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    image = cv2.imread(load_path)
    orig = image.copy()
    (h, w) = image.shape[: 2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater
        # than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]

            face = anonymize_face_pixelate(face,
                                           blocks=10)

            # store the blurred face in the output image
            image[startY:endY, startX:endX] = face
    return save_path
