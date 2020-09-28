import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--classes", dest="classes", type=int, default=10)
parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=32)
parser.add_argument("-e", "--epoch", dest="epoch", type=int, default=15)
parser.add_argument("--cpu", dest="cpu", action="store_true")
parser.add_argument("--proxy", dest="proxy", action="store_true")
parser.add_argument("--soft_gpu", dest="soft_gpu", action="store_true")
args = parser.parse_args()
print(args)

N_CLASS = args.classes
BATCH_SIZE = args.batch_size
EPOCH = args.epoch

if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if args.proxy:
    os.environ["http_proxy"] = "http://web-proxy.oa.com:8080"
    os.environ["https_proxy"] = "http://web-proxy.oa.com:8080"

import tensorflow as tf         # tf.version = 2.3.1
from dataset import CIFAR
from tensorflow.keras.models import Model
from tensorflow import keras
import matplotlib.pyplot as plt


def set_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus and args.soft_gpu:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


def train(x_train, y_train, x_test, y_test):
    base_model = keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        alpha=1.0,
        include_top=False,
        weights="imagenet",
        pooling="max"
    )
    x = base_model.output
    # x = keras.layers.Dense(256)(x)
    o = keras.layers.Dense(N_CLASS)(x)
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True
    model = Model(inputs=[base_model.input], outputs=[o])
    # print(model.summary())
    model.compile(
        optimizer=keras.optimizers.RMSprop(0.0001),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    train_dg = keras.preprocessing.image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.4, 1),
        # rescale=1./255.,
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
    )
    test_dg = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
    )
    train_dg.fit(x_train)
    test_dg.fit(x_test)
    train_g = train_dg.flow(x_train, y_train, batch_size=BATCH_SIZE)
    test_g = test_dg.flow(x_test, y_test, batch_size=BATCH_SIZE)
    model.fit(
        train_g, steps_per_epoch=len(x_train)//BATCH_SIZE, epochs=EPOCH,
        max_queue_size=50, workers=8, use_multiprocessing=False,
        validation_data=test_g, validation_steps=5
    )

    # x_train = keras.applications.mobilenet_v2.preprocess_input(tf.cast(x_train, tf.float32))
    # model.fit(x_train, y_train, validation_split=0.1, batch_size=BATCH_SIZE, epochs=EPOCH, shuffle=True, max_queue_size=150, workers=8,)

    model.save(model_path)


def restore(x_test, y_test):
    n = 100
    x_test, y_test = x_test[:n], y_test[:n]
    x = tf.cast(x_test, tf.float32)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    print(x[:2, :4, :4, 0])
    model = keras.models.load_model(model_path)
    pred = model.predict(x)
    print(pred.argmax(axis=1))
    print(y_test.ravel())
    print((pred.argmax(axis=1) == y_test.ravel()).sum()/100)
    show_cifar(x_test, y_test, pred.argmax(axis=1))


def show_cifar(x, y, pred):
    x, y, pred = x[:25], y[:25], pred[:25]
    plt.figure(1, (10, 10))
    for i in range(5):
        for j in range(5):
            n = i * 5 + j
            plt.subplot(5, 5, n+1)
            plt.imshow(x[n])
            plt.xticks(())
            plt.yticks(())
            plt.xlabel(cifar.classes[int(y[n])] + " - " + cifar.classes[int(pred[n])])
    plt.tight_layout()
    plt.savefig("t{}.png".format(N_CLASS))
    # plt.show()


if __name__ == "__main__":
    model_path = "models/cifar{}MobileNet".format(N_CLASS)
    set_gpu()
    cifar = CIFAR(n_class=N_CLASS)
    (x_train, y_train), (x_test, y_test) = cifar.load()
    x_train = keras.backend.resize_images(x_train, height_factor=3, width_factor=3, data_format="channels_last")
    x_test = keras.backend.resize_images(x_test, height_factor=3, width_factor=3, data_format="channels_last")
    train(x_train, y_train, x_test, y_test)
    restore(x_test, y_test)

