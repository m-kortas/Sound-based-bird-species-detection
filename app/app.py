from core import *

model = 'AM_mobilenet_5classes.h5'
classes = np.array(['Acroc','Ember','Parus','Phyll','Sylvi'])
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
#app.secret_key = ""
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def upload_file():             
    ''' renders upload page to upload audiofile '''
    return render_template('upload.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/prediction", methods=["POST"]) 
def prediction():
    ''' makes prediction (uploads file, creates spectrogram, applies neural networks and displays result on result page) '''
    file = request.files['file']
    
    if file.filename == '':
            error = 'No selected file'
            return render_template("error.html", error = error)
        
    if file and allowed_file(file.filename):
        image, fig = create_spectrogram(file)
        pred = predict(model, image)
        result = create_result(pred, classes)
        
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        spectrogram = "data:image/png;base64,"
        spectrogram += base64.b64encode(pngImage.getvalue()).decode('utf8')

        if result['probability'] > 50:
          #  text = 'There is a ' + str(result['probability']) + '% chance that it is a ' + result['bird'] to be done in the template
            bird_path = create_bird_path(result['bird'])
            probability = str(result['probability'])
            bird_type = result['bird']
            name, en_name, desc = get_bird_data(bird_type)
            
            return render_template("result.html", image = spectrogram, bird = bird_path, probability = probability,
                                  bird_type = bird_type, name = name, en_name = en_name, desc = desc)

        else:
          #  text = 'It is not any of our 5 birds!' to be done in the template
        #    bird_path = '/static/images/not_a_bird.jpg' to be done in the template

            return render_template("not_a_bird.html", image = spectrogram)
    else:
        error = 'Wrong file format'
        return render_template("error.html", error = error)
    
@app.errorhandler(413)
def error413(e):
    error = 'Too big file'
    return render_template("error.html", error = error), 413    

if __name__ == '__main__':
    app.run(debug = True)
