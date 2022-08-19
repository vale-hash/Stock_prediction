import csv
import json
import pandas as pd
from flask import Flask, request, render_template
import numpy as np
import pickle
import plotly
import plotly.express as px
import os

app = Flask(__name__)
app.secret_key = 'goat_goat'

pic_folder = os.path.join('static', 'img')

app.config['UPLOAD_FOLDER'] = pic_folder


def get_model():
    global model
    model = pickle.load(open('/root/anaconda3/bin/flaskie/stockmodel.pkl', 'rb'))
    print('model loaded')
    print(' model loading...')


get_model()

global bg
bg = os.path.join(app.config['UPLOAD_FOLDER'],'stock_pic.jpeg')
@app.route('/')
def start_page():
    return render_template('sample.html', bg_image=bg)


@app.route('/data', methods=['GET', 'POST'])
def predict_callback():
    if request.method == 'POST':
        f = request.form['prediction_data']
        global data
        data = []
        with open(f) as file:
            csvfile = csv.reader(file)
            for row in csvfile:
                data.append(row)
            data = pd.read_csv(f)

        y = data.Open
        x = data.drop('Open', axis=1)
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        training_set = y_train
        print('data preprocessed')
        # feature scaling for LSTM use
        from sklearn.preprocessing import MinMaxScaler
        scale = MinMaxScaler(feature_range=(0, 1))
        test_set = y_test
        test_set_reshaped = test_set.values.reshape(-1, 1)
        # scaling
        test_set_scaled = scale.fit_transform(test_set_reshaped)
        z = len(training_set) // 20

        Total_data = pd.concat((training_set, test_set), axis=0)
        inputs = Total_data[len(Total_data) - len(test_set) - z:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scale.transform(inputs)
        X_test = []
        # creating a data structure with 20 intervals and 1 output
        for i in range(z, len(test_set)):
            X_test.append(test_set_scaled[i - z:i, 0])
        X_test = np.array(X_test)
        predicted_price = model.predict(X_test)
        predicted_price = predicted_price.reshape(-1, 1)
        predicted_price = scale.inverse_transform(predicted_price)
        real_price = test_set.values.reshape(-1, 1)
        fig = px.bar(predicted_price, barmode='group')
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('result.html', bg_image=bg, graphJSON=graphJSON)
