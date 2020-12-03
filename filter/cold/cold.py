import cv2
import copy
import numpy as np
import os
from scipy.interpolate import UnivariateSpline


def spreadLookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


def coldImage(image):
    increaseLookupTable = spreadLookupTable(
        [0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spreadLookupTable(
        [0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))


def cold(img_name, load_folder, save_folder):
    load_path = os.path.join(load_folder, img_name)
    print(load_path)
    save_path = os.path.join(save_folder,'cold' ,img_name)
    img = cv2.imread(load_path)
    res = coldImage(img)
    cv2.imwrite(save_path, res)
    return save_path
