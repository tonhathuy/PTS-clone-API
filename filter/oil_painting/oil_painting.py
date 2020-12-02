import cv2


def oil_painting(img_name, load_folder, save_folder):
    load_path = os.path.join(load_folder, img_name)
    print(load_path)
    save_path = os.path.join(save_folder, img_name)
    img = cv2.imread(load_path)
    res = cv2.xphoto.oilPainting(img, 7, 1)
    cv2.imwrite(save_path, res)
    return save_path
