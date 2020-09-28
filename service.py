import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--classes", dest="classes", type=int, default=10)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--port", dest="port", default="50000")
parser.add_argument("--password", dest="password")

args = parser.parse_args()
print(args)
N_CLASS = args.classes
if not args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
import numpy as np
from dataset import CIFAR


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

app = Flask(__name__)
model_path = "models/cifar{}MobileNet".format(N_CLASS)
cifar = CIFAR(n_class=N_CLASS)
model = keras.models.load_model(model_path)


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x + 1e-5)
    return y / y.sum(axis=axis, keepdims=True)


def predict_img(img_path):
    img = keras.preprocessing.image.load_img(img_path, target_size=(96, 96))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    preds = model.predict(x)
    sf = softmax(preds.ravel())
    top3_idx = np.argsort(sf)[-3:][::-1]
    top3_name = [cifar.classes_zh[i] for i in top3_idx]
    top3_percent = sf[top3_idx]
    return [{"name": n, "percent": "{:6.2f}%".format(p*100)} for n, p in zip(top3_name, top3_percent)]


@app.route("/ai/predict/mobilenet/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # if request.form.get("password") != args.password:
        #     return Response("password err", status=403, mimetype='application/text')
        f = request.files["file"]
        file_path = "tmp_upload/" + secure_filename(f.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        f.save(file_path)
        top3 = predict_img(file_path)
        os.remove(file_path)
        print(top3)
        return jsonify(top3)
    else:
        return render_template("mobilenet.html", classes=", ".join(cifar.classes_zh), post_url=request.path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=args.port, debug=True)