from flask import Flask, render_template, request, flash, redirect
import pandas as pd
import librosa  
import ffmpeg
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import gcf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from keras.preprocessing.image import img_to_array
from PIL import Image

def fig2img(fig):
    ''' Transforms matplotlib figure to image '''
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w,h,4)
    buf = np.roll(buf,3,axis = 2)
    w, h, d = buf.shape
    return Image.frombytes("RGB",(w,h),buf.tostring())

def create_spectrogram(file):
    ''' loads audio file and creates spectrogram '''
    signal, sr = librosa.load(file,duration=10)   
    fig = gcf()
    DPI = fig.get_dpi()
    fig = plt.figure()
    fig.set_size_inches(224/float(DPI),224/float(DPI))
    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    S = librosa.feature.melspectrogram(y=signal,sr=sr,
                                            n_fft=1024,
                                            hop_length=1024, 
                                            n_mels=128, 
                                            htk=True, 
                                            fmin=1400, 
                                            fmax=sr/2) 
    librosa.display.specshow(librosa.power_to_db(S**2,ref=np.max), fmin=1400,y_axis='linear')
    
    image = fig2img(fig)
    image = img_to_array(image)
    image = np.array([image]) 
    return image, fig

def predict(model, image):
    ''' makes prediction out of the spectrogram '''
    net = MobileNetV2(include_top=False,
                            weights='imagenet',
                            input_tensor=None,
                            input_shape=(224,224,3))
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

def create_result(pred, classes):
    ''' creates results (bird class and probability) ''' 
    top = np.argsort(pred[0])[:-2:-1]
    result = {'bird': '', 'probability': ''}
    result['bird'] = classes[top[0]]
    result['probability'] = int(round(pred[0][top[0]],2)*100)
    return result
