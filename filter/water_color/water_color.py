import cv2


def water_color(img_name, load_folder, save_folder):
    load_path = os.path.join(load_folder, img_name)
    print(load_path)
    save_path = os.path.join(save_folder, img_name)
    img = cv2.imread(load_path)
    res = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    cv2.imwrite(save_path, res)
    return save_path

# sigma_s controls the size of the neighborhood. Range 1 - 200
# sigma_r controls the how dissimilar colors within the neighborhood will be averaged.
#  A larger sigma_r results in large regions of constant color. Range 0 - 1
