import inspect
import os

import numpy as np

from flask import Flask
from flask import flash
from flask import json
from flask import redirect
from flask import render_template
from flask import request
from flask import send_from_directory
from flask import url_for
from werkzeug.utils import secure_filename

from feature_extractor.run import FeatureExtractor
from src import model_bnf

app = Flask(__name__)

UPLOAD_FOLDER = '/home/gangeshwark/PycharmProjects/QbE_System/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'wav', 'mp3'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def hello_world():
    data = "Hello World"
    return render_template('audio5js.html', name=data)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/v1/start_search', methods=['GET', 'POST'])
def upload_audio():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            '''
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #file.save(path)
            path = '/projects/PycharmProjects/QbE_System/uploads/hello1.wav'
            query_features_path = FeatureExtractor.bnf(path)
            corpus_features_path = os.path.abspath('../corpus_features/bnf_database/raw_bnfea_fbank_pitch.1.scp')
            print corpus_features_path, query_features_path
            os.chdir(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
            AQS = model_bnf.AQSearch(query_features_path, corpus_features_path)

            matrix, top_k = AQS.search()
            '''
            top_k = np.array([[10, 2, 3],
                                [20, 3, 4],
                                [30, 3, 4],
                                [40, 5, 6],
                                [60, 5, 6],
                                [80, 5, 6],
                                [100, 5, 6],
                                [120, 5, 6],
                                [140, 5, 6],
                                [160, 5, 6],
                                [180, 5, 6],
                                [200, 5, 6],
                                [220, 5, 6],
                                [240, 5, 6],
                                [260, 5, 6],
                                [280, 5, 6],
                                [300, 5, 6],
                                [320, 5, 6],
                                [340, 5, 6],
                                [360, 5, 6]])
            print type(top_k)
            return json.dumps(top_k.tolist())


@app.route('/static/<path:path>')
def send(path):
    return send_from_directory('client', path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
