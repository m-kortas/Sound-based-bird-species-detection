from core import *

app = Flask(__name__)

model = 'AM_mobilenet_5classes.h5'
classes = np.array(['Acroc','Ember','Parus','Phyll','Sylvi'])

@app.route('/')
def upload_file():             
    ''' renders upload page to upload audiofile '''
    return render_template('upload.html')

@app.route("/prediction", methods=["POST"]) 
def prediction():
    ''' makes prediction (uploads file, creates spectogram, applies neural networks and displays result on result page) '''
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
    spectogram = "data:image/png;base64,"
    spectogram += base64.b64encode(pngImage.getvalue()).decode('utf8')
        
    if result['bird'] == 'Parus':
        bird_path = '/static/images/parus.jpg'
        
    return render_template("result.html", text = text, image = spectogram, bird = bird_path) 

if __name__ == '__main__':
    app.run(debug = True)


