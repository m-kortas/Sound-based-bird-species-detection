#import flask
from flask import Flask, render_template, request, jsonify
import pandas as pd
import sys, os
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
from keras.preprocessing.image import load_img,img_to_array
from PIL import Image
from werkzeug import secure_filename

app = Flask(__name__)

model = 'AM_mobilenet_5classes.h5'
classes = np.array(['Acroc','Ember','Parus','Phyll','Sylvi'])

def fig2img(fig):
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w,h,4)
    buf = np.roll(buf,3,axis = 2)
    w, h, d = buf.shape
    return Image.frombytes("RGB",(w,h),buf.tostring())

def create_spectogram(file):
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
    top = np.argsort(pred[0])[:-2:-1]
    result = {'bird': '', 'probability': ''}
    result['bird'] = classes[top[0]]
    result['probability'] = int(round(pred[0][top[0]],2)*100)
    return result


def create_result(pred, classes):
    top = np.argsort(pred[0])[:-2:-1]
    result = {'bird': '', 'probability': ''}
    result['bird'] = classes[top[0]]
    result['probability'] = int(round(pred[0][top[0]],2)*100)
    return result

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route("/prediction", methods=["GET","POST"])
def prediction():
    if request.method == 'POST':
        file = request.files['file']
        image, fig = create_spectogram(file)
        pred = predict(model, image)
        result = create_result(pred, classes)
        if result['probability'] > 50:
            text = 'There is a ' + str(result['probability']) + '% chance that it is a ' + result['bird']
        else:
            text = 'Tt is not any of our 5 birds!'
        
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
        
        if result['bird'] == 'Parus':
            bird_path = '/static/images/parus.jpg'
        
    return render_template("result.html", text = text, image=pngImageB64String, bird = bird_path) 



if __name__ == '__main__':
    app.run(debug = True)


