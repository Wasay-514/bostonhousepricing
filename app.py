import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
from matplotlib import scaler
import numpy as np
import pandas as pd

app = Flask(__name__)
##load the model

regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open("Scaling.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/predict_api' ,methods=['post']) 
# def predict_api():
#     # new_house=request.json['new_house']
#     new_house=request.json
#     print(np.array(list(new_house.values()))) #.reshape(1,-1)
#     new_data=scale.Transform(np.array(list(new_house.values()))) 
#     output=regmodel.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])
@app.route('/predict_api', methods=['POST'])
def predict_api():
    
    new_house = request.json
    
    columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
               'DIS','RAD','TAX','PTRATIO','B','LSTAT']
    
    arr = np.array([new_house[col] for col in columns]).reshape(1, -1)
    
    new_data = scaler.transform(arr)
    # new_data = scaler.transform(arr)
    
    output = regmodel.predict(new_data)
    
    return jsonify(float(output[0]))


if __name__=="__main__":
    app.run(debug=True)

