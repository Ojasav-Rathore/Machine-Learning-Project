from distutils.log import debug
import joblib
from flask import Flask, request
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

model = joblib.load(r'C:\Users\Dell\Documents\GitHub\data-science-project\SVR_model')
scaler = joblib.load(r"C:\Users\Dell\Documents\GitHub\data-science-project\Scaling_parameters")

@app.route('/predict_cost',methods=['POST'])

def predict_cost():
    event = json.loads(request.data)
    values = event['values']
    values = list(map(np.float, values))
    pre = np.array(values)
    pre = pre.reshape(1,-1)
    pre = scaler.transform(pre)
    res = model.predict(pre)
    print(res)
    return str(res[0])

if __name__ == '__main__':
    app.run(debug = True)
