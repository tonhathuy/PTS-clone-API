import cv2
import os
def dodgeV2(x, y):
    return cv2.divide(x, 255 - y, scale=256)
def sketch(img_name, load_folder, save_folder):
    load_path = os.path.join(load_folder, img_name)
    print(load_path)
    save_path = os.path.join(save_folder,'sketch' ,img_name)
    img = cv2.imread(load_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21),sigmaX=0, sigmaY=0)
    final_img = dodgeV2(img_gray, img_smoothing)
    cv2.imwrite(save_path,final_img)
    return save_path
