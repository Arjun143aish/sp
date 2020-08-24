import numpy as np
import pandas as pd
from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

obj = pickle.load(open('BOW.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        Vector = obj.transform(data).toarray()
        my_prediction = model.predict(Vector)
        
        return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
    