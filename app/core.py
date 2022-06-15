import os

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from keras.preprocessing.image import img_to_array
from matplotlib.pyplot import gcf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

matplotlib.use('Agg')

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
BIRD_DATA = os.path.join(THIS_DIR, 'data', 'bird_data.xlsx')


def fig2img(fig):
    """ Transforms matplotlib figure to image """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    w, h, d = buf.shape
    return Image.frombytes("RGB", (w, h), buf.tostring())


def create_spectrogram(file):
    """ loads audio file and creates spectrogram """
    signal, sr = librosa.load(file, duration=10)
    fig = gcf()
    DPI = fig.get_dpi()
    fig = plt.figure()
    fig.set_size_inches(224 / float(DPI), 224 / float(DPI))

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    S = librosa.feature.melspectrogram(y=signal, sr=sr,
                                       n_fft=1024,
                                       hop_length=1024,
                                       n_mels=128,
                                       htk=True,
                                       fmin=1400,
                                       fmax=sr / 2)
    librosa.display.specshow(librosa.power_to_db(
        S ** 2, ref=np.max), fmin=1400, y_axis='linear')

    image = fig2img(fig)
    image = img_to_array(image)
    image = np.array([image])
    return image, fig


def predict(model, image):
    """ makes prediction out of the spectrogram """
    net = MobileNetV2(include_top=False,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=(224, 224, 3))
    x = net.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(5, activation='softmax')(x)
    loaded_model = Model(inputs=net.input, outputs=output_layer)
    loaded_model.load_weights(model)
    loaded_model.compile(optimizer=Adam(),
                         loss='categorical_crossentropy', metrics=['accuracy'])
    pred = loaded_model.predict(image)
    return pred


def get_bird_data(bird):
    df = pd.read_excel(BIRD_DATA)
    df = df[df['species'] == bird].reset_index(drop=True)
    name = df['name'][0]
    en_name = df['en_name'][0]
    desc = df['desc'][0]
    return name, en_name, desc


def create_bird_path(bird):
    img_path = '/static/images/'
    bird = bird.lower()
    img_file = bird + '.jpg'
    bird_path = img_path + img_file
    return bird_path


def create_result(pred, classes):
    """ creates results (bird class and probability) """
    top = np.argsort(pred[0])[:-2:-1]
    result = {'bird': classes[top[0]], 'probability': int(round(pred[0][top[0]], 2) * 100)}
    return result
