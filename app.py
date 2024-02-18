from flask import Flask,jsonify,request
import numpy as np
import pickle
import json


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def home():
    return "Welcome to the predictor"

@app.route('/predict',methods=['POST'])
def predict():
    data = request.json['data']
    new_data = np.array(list(data)).reshape(1,-1)
    output = model.predict(new_data)
    return jsonify({'output':output[0]})

if __name__=="__main__":
    app.run(debug=True)