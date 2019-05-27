import flask
import pandas as pd
from sklearn import preprocessing
from keras.models import load_model
from flask import Flask, request, render_template
import tensorflow as tf
import keras.backend as K
import numpy as np
import shutil
# from werkzeug import secure_filename
import os
import dash
import json
app = Flask(__name__)
# app_dash = dash.Dash(__name__, server=app, url_base_pathname='/pathname')

global graph,model
result = 0
graph = tf.get_default_graph()
# APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'upload')
# app.config['UPLOAD_FOLDER'] = '/home/ec2-user/upload'

@app.route('/predict/<rul>', methods=['GET','POST'])
def predict(rul):
    rul = request.args.get('rul')
    grp = request.args.get('grp')

    with graph.as_default():
        global result
        if request.method == 'POST':
            if not 'file' in request.files:
                return jsonify({'error': 'no file'}), 400
            data = request.files.get('file')
#             latestfile = request.files['customlogo']
#             full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'logo.csv')
#             data.save(full_filename)
#             filename = secure_filename(data.filename)
#             shutil.copy('filename', 'file')
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            df = pd.read_csv(data)
            processed_df = preprocess(df)
            y_pred = model.predict(processed_df)
            result = pd.DataFrame(y_pred)
        return render_template('view.html', rul = result)
        # return render_template('view.html', grp = 'graph')

        # return flask.jsonify(result)



def preprocess(df):
    sequence_length = 50
#     df['cycle_norm'] = df['cycle']
    cols_normalize = df.columns.difference(['id'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_df = pd.DataFrame(min_max_scaler.fit_transform(df[cols_normalize]),
                             columns=cols_normalize,
                             index=df.index)
    join_df = df[df.columns.difference(cols_normalize)].join(norm_df)
    df = join_df.reindex(columns = df.columns)
    sensor_cols = ['s' + str(i) for i in range(1,22)]
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle']
    sequence_cols.extend(sensor_cols)
    seq_array_test_last = [df[df['id']==df['id'][0]][sequence_cols].values[-sequence_length:]]
    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
    return seq_array_test_last

def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

if __name__ == "__main__":
    print("Loading model")
    model = load_model("regression_model.h5",custom_objects={'r2_keras': r2_keras})
    app.run(host = '0.0.0.0', port = 5000, debug=True)