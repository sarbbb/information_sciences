from flask import Flask, render_template,request
import pickle
import numpy as np

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
    pred = model.predict(arr)
    return render_template('after.html',data = pred)


if __name__ =='__main__':
    app.run(port=3000,debug=True)