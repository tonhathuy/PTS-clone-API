import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
from flask import Flask, request
import json
import base64
import json
import cv2
import numpy as np
from PIL import Image
import io
import logging
import time

from filter.gray.gray import gray
from filter.sketch.sketch_potrait import sketch
from filter.bw_pencil.bw_pencil import bw_pencil
from filter.water_color.water_color import water_color
from filter.oil_painting.oil_painting import oil_painting
from filter.cold.cold import cold
from filter.warm.warm import warm
# from filter.cartoonizer.cartoonizer import cartoonizer
import tensorflow as tf 
from filter.cartoonizer import network
from filter.cartoonizer import guided_filter

from time import gmtime, strftime

from flask import Flask, render_template, Response, request, jsonify, send_from_directory, redirect, url_for
from flask import send_file, send_from_directory, safe_join, abort

from utils.parser import get_config
# from utils.utils import load_class_names, get_image


# create backup dir
if not os.path.exists('backup'):
    os.mkdir('backup')

# create json dir
if not os.path.exists('json_dir'):
    os.mkdir('json_dir')

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')

# create log_file, rcode
LOG_PATH = cfg.SERVICE.LOG_PATH
UPLOAD = cfg.SERVICE.UPLOAD_DIR
RESULT = cfg.SERVICE.RESULT_DIR
HOST = cfg.SERVICE.SERVICE_IP
PORT = cfg.SERVICE.SERVICE_PORT
HOST_URL = cfg.SERVICE.SERVICE_URL
MODEL_PATH_CARTOONIZER = cfg.SERVICE.MODEL_PATH_CARTOONIZER

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
#app.config["CLIENT_IMAGES"] = "static/result/gray"

# -------------------CARTOONIZER CODE------------------------------------------ 
input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
network_out = network.unet_generator(input_photo)
final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

all_vars = tf.trainable_variables()
gene_vars = [var for var in all_vars if "generator" in var.name]
saver = tf.train.Saver(var_list=gene_vars)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH_CARTOONIZER))

def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image

def cartoonizer(img_name, load_folder, save_folder):
    name_list = os.listdir(load_folder)
    load_path = os.path.join(load_folder, img_name)
    print(load_path)
    save_path = os.path.join(save_folder, 'cartoonizer',img_name)
    image = cv2.imread(load_path)
    image = resize_crop(image)
    batch_image = image.astype(np.float32) / 127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)
    output = sess.run(final_out, feed_dict={input_photo: batch_image})
    output = (np.squeeze(output) + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, output)
    return save_path


@app.route('/send', methods=['POST'])
def send_img():
    # user_image = request.files['image']
    if (request.method == 'POST'):
        print(request.json['image-name'])
        img_name = request.json['image-name']
        data_base64 = request.json['base64']

        jpg_original = base64.b64decode(data_base64)
        #print(jpg_original)
        print(len(jpg_original))
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        print(len(jpg_as_np))
        img = cv2.imdecode(jpg_as_np, flags=1)
        print(img.shape)
        path_img_save = UPLOAD + '/'+str(img_name)
        cv2.imwrite(path_img_save, img)

        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # cv2.imwrite(str(img_name) + '-gray.jpg', img_gray)

        with open(path_img_save, "rb") as f:
            im_b64 = base64.b64encode(f.read())
        request.json['base64'] = im_b64.decode("utf-8")
        # with open(str(img_name) + '-gray'+'.json', 'w') as outfile:
        #     json.dump(request.json, outfile, ensure_ascii=False, indent=4)

    return request.json
@app.route('/result', methods=['POST'])
def result():
    if (request.method == 'POST'):
        print(request.json['image-name'])
        print(request.json['filter-id'])
        img_name = request.json['image-name']
        filter_id = request.json['filter-id']
        list_filter = [gray, sketch,bw_pencil,water_color,oil_painting,cold,warm,cartoonizer]
        output_URLs = []
        for i in filter_id:
            output = list_filter[i](img_name, UPLOAD, RESULT)
            output_URL =HOST_URL + output
            output_URLs.append(output_URL)
        return str(output_URLs)

# @app.route("/static/result/<filter_name>/<image_name>")
# def get_image_v2(image_name, filter_name):
#     path = os.path.join(app.config["CLIENT_IMAGES"], filter_name)
#     try:
#         return send_from_directory(path, filename=image_name, as_attachment=True)
#     except FileNotFoundError:
#         abort(404)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)
