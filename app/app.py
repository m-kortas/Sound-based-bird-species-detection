import base64
import io
import os

import matplotlib
import numpy as np
from flask import Flask, render_template, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from core import (create_bird_path, create_result, create_spectrogram,
                  get_bird_data, predict)

matplotlib.use('Agg')

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
BIRD_DATA = os.path.join(THIS_DIR, 'data', 'bird_data.xlsx')
TEMPLATES = os.path.join(THIS_DIR, 'templates')
NOT_A_BIRD = 'not_a_bird.html'
UPLOAD = 'upload.html'
RESULT = 'result.html'
ERROR = 'error.html'
model = os.path.join(THIS_DIR, 'model', 'model.h5')
classes = np.array(['Acroc', 'Ember', 'Parus', 'Phyll', 'Sylvi'])
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
# app.secret_key = ""
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/')
def upload_file():
    """ renders upload page to upload audiofile """
    return render_template(UPLOAD)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/prediction", methods=["POST"])
def prediction():
    """ makes prediction (uploads file, creates spectrogram, applies neural
    networks and displays result on result page) """
    file = None
    file = request.files['file']

    if not file or file.filename == '':
        error = 'No selected file'
        return render_template(ERROR, error=error)

    if file and allowed_file(file.filename):

        image, fig = create_spectrogram(file)
        pred = predict(model, image)
        result = create_result(pred, classes)

        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        spectrogram = "data:image/png;base64,"
        spectrogram += base64.b64encode(pngImage.getvalue()).decode('utf8')

        if result['probability'] > 74:
            bird_path = create_bird_path(result['bird'])
            probability = str(result['probability'])
            bird_type = result['bird']
            name, en_name, desc = get_bird_data(bird_type)

            return render_template(RESULT, image=spectrogram, bird=bird_path,
                                   probability=probability, bird_type=bird_type, name=name,
                                   en_name=en_name, desc=desc)

        else:
            return render_template(NOT_A_BIRD, image=spectrogram)

    else:
        error = 'Wrong file format'
        return render_template(ERROR, error=error)


@app.errorhandler(413)
def error413():
    error = 'Too big file'
    return render_template(ERROR, error=error), 413


if __name__ == '__main__':
    app.run(debug=True)
