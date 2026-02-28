import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
from matplotlib import scale
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
#     new_data=scaler.Transform(np.array(list(new_house.values()))) 
#     output=regmodel.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])
@app.route('/predict_api', methods=['POST'])
def predict_api():
    
    new_house = request.json['new_house']
    
    columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
               'DIS','RAD','TAX','PTRATIO','B','LSTAT']
    
    arr = np.array([new_house[col] for col in columns]).reshape(1, -1)
    
    new_data = scaler.transform(arr)
    # new_data = scaler.transform(arr)
    
    output = regmodel.predict(new_data)
    
    return jsonify(float(output[0]))


@app.route('/predict',methods=['POST'])
def predict(): #This is function 
    new_house=[float(x) for x in request.form.values()]
    final_input= scaler.transform(np.array(new_house).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template('home.html',prediction_text="The house price prediction is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)

