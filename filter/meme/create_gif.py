# USAGE
# python create_gif.py --config config.json --image images/vampire.jpg --output out.gif

# import the necessary packages
from imutils import face_utils
from imutils import paths
import numpy as np
import argparse
import imutils
import shutil
import json
import dlib
import cv2
import sys
import os


def overlay_image(bg, fg, fgMask, coords):

    (sH, sW) = fg.shape[:2]
    (x, y) = coords

    overlay = np.zeros(bg.shape, dtype="uint8")
    overlay[y:y + sH, x:x + sW] = fg

    alpha = np.zeros(bg.shape[:2], dtype="uint8")
    alpha[y:y + sH, x:x + sW] = fgMask
    alpha = np.dstack([alpha] * 3)

    output = alpha_blend(overlay, bg, alpha)

    return output


def alpha_blend(fg, bg, alpha):

    fg = fg.astype("float")
    bg = bg.astype("float")
    alpha = alpha.astype("float") / 255

    fg = cv2.multiply(alpha, fg)
    bg = cv2.multiply(1 - alpha, bg)

    output = cv2.add(fg, bg)

    # return the output image
    return output.astype("uint8")


def create_gif(inputPath, outputPath, delay, finalDelay, loop):
    # grab all image paths in the input directory
    imagePaths = sorted(list(paths.list_images(inputPath)))

    # remove the last image path in the list
    lastPath = imagePaths[-1]
    imagePaths = imagePaths[:-1]

    cmd = "convert -delay {} {} -delay {} {} -loop {} {}".format(
        delay, " ".join(imagePaths), finalDelay, lastPath, loop,
        outputPath)
    os.system(cmd)


def gif(img_name, load_folder, save_folder):
    load_path = os.path.join(load_folder, img_name)
    print(load_path)
    img_gif = img_name.split('.')
    save_path = os.path.join(save_folder, f"{img_gif[0]}.gif")

    config = json.loads(open("config.json").read())  # changed
    sg = cv2.imread(config["sunglasses"])
    sgMask = cv2.imread(config["sunglasses_mask"])

    shutil.rmtree(config["temp_dir"], ignore_errors=True)
    os.makedirs(config["temp_dir"])

    print("[INFO] loading models...")
    detector = cv2.dnn.readNetFromCaffe(config["face_detector_prototxt"],
                                        config["face_detector_weights"])
    predictor = dlib.shape_predictor(config["landmark_predictor"])

    image = cv2.imread(load_path)  # changed
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    print("[INFO] computing object detections...")
    detector.setInput(blob)
    detections = detector.forward()

    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]

    if confidence < config["min_confidence"]:
        print("[INFO] no reliable faces found")
        sys.exit(0)

    # compute the (x, y)-coordinates of the bounding box for the face
    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
    (startX, startY, endX, endY) = box.astype("int")

    rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
    shape = predictor(image, rect)
    shape = face_utils.shape_to_np(shape)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # rotate the sunglasses image by our computed angle, ensuring the
    # sunglasses will align with how the head is tilted
    sg = imutils.rotate_bound(sg, angle)

    sgW = int((endX - startX) * 0.9)
    sg = imutils.resize(sg, width=sgW)

    sgMask = cv2.cvtColor(sgMask, cv2.COLOR_BGR2GRAY)
    sgMask = cv2.threshold(sgMask, 0, 255, cv2.THRESH_BINARY)[1]
    sgMask = imutils.rotate_bound(sgMask, angle)
    sgMask = imutils.resize(sgMask, width=sgW, inter=cv2.INTER_NEAREST)

    steps = np.linspace(0, rightEyeCenter[1], config["steps"],
                        dtype="int")

    # start looping over the steps
    for (i, y) in enumerate(steps):

        shiftX = int(sg.shape[1] * 0.25)
        shiftY = int(sg.shape[0] * 0.35)
        y = max(0, y - shiftY)

        # add the sunglasses to the image
        output = overlay_image(image, sg, sgMask,
                               (rightEyeCenter[0] - shiftX, y))

        if i == len(steps) - 1:

            dwi = cv2.imread(config["deal_with_it"])
            dwiMask = cv2.imread(config["deal_with_it_mask"])
            dwiMask = cv2.cvtColor(dwiMask, cv2.COLOR_BGR2GRAY)
            dwiMask = cv2.threshold(dwiMask, 0, 255,
                                    cv2.THRESH_BINARY)[1]

            # resize both the text image and mask to be 80% the width of
            # the output image
            oW = int(W * 0.8)
            dwi = imutils.resize(dwi, width=oW)
            dwiMask = imutils.resize(dwiMask, width=oW,
                                     inter=cv2.INTER_NEAREST)

            # compute the coordinates of where the text will go on the
            # output image and then add the text to the image
            oX = int(W * 0.1)
            oY = int(H * 0.8)
            output = overlay_image(output, dwi, dwiMask, (oX, oY))

        # write the output image to our temporary directory
        p = os.path.sep.join([config["temp_dir"], "{}.jpg".format(
            str(i).zfill(8))])
        cv2.imwrite(p, output)

    print("[INFO] creating GIF...")
    create_gif(config["temp_dir"], save_path, config["delay"],
               config["final_delay"], config["loop"])
    return save_path


config = json.loads(open("config.json").read())
print("[INFO] cleaning up...")
shutil.rmtree(config["temp_dir"], ignore_errors=True)
