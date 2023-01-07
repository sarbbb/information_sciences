from flask import Flask, render_template,request
import pickle
import numpy as np
import joblib

sc_X = joblib.load('std_scaler.bin')
model = pickle.load(open('model_LR.pkl','rb'))
app = Flask(__name__)

@app.route("/")
def man():
    return render_template('home.html')

@app.route("/predict",methods=['POST'])
def home():
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    data5 = float(request.form['e'])
    data6 = float(request.form['f'])
    data7 = float(request.form['g'])
    data8 = float(request.form['h'])
    arr = np.array([[data1,data2,data3,data4,data5,data6,data7,data8]])
    arr_sc = sc_X.transform(arr)
    pred = model.predict(arr_sc)[0]
    if pred==0:
        pred_prob = model.predict_proba(arr_sc)[0][0]
    else:
        pred_prob = model.predict_proba(arr_sc)[0][1]

    pred_prob = str((round(pred_prob,4))*100) + '%'
    return render_template('after.html', data = [pred,pred_prob])


if __name__ =='__main__':
    app.run("192.168.0.15",port=3000,debug=True)