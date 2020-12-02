import cv2


def bw_color(img_name, load_folder, save_folder):
    load_path = os.path.join(load_folder, img_name)
    print(load_path)
    save_path = os.path.join(save_folder, img_name)
    img = cv2.imread(load_path)
    dst_gray, dst_color = cv2.pencilSketch(
        img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    cv2.imwrite(save_path, dst_gray)
    cv2.imwrite(save_path, dst_color)
    return save_path


# sigma_s and sigma_r are the same as in stylization.
# shade_factor is a simple scaling of the output image intensity.
#  The higher the value, the brighter is the result. Range 0 - 0.1
