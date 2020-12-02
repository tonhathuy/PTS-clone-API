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

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
app.config["CLIENT_IMAGES"] = "static/result/gray"


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
        list_filter = [gray, sketch]
        output = list_filter[request.json['filter-id']](img_name, UPLOAD, RESULT)
        output_URL =HOST +':' + str(PORT) +'/' + output
        return str(output_URL)

# @app.route("/static/result/<filter_name>/<image_name>")
# def get_image_v2(image_name, filter_name):
#     path = os.path.join(app.config["CLIENT_IMAGES"], filter_name)
#     try:
#         return send_from_directory(path, filename=image_name, as_attachment=True)
#     except FileNotFoundError:
#         abort(404)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)
