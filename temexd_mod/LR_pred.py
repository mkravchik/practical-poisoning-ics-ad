from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import pickle
import numpy as np

"""
>> clear classes;
>> mod = py.importlib.import_module('LR_pred');
>> py.importlib.reload(mod);
>> res = py.LR_pred.predict();
[[22.9      49.401245 25.320041 37.292848  1.86    ]]
[0]
"""
# f_model = open("../LogisticRegression.mdl", 'rb')
# f_model = open("../RandomForestClassifier.mdl", 'rb')
model = None #pickle.load(f_model)

x_att = np.array([22.9, 49.401245, 25.320041, 37.292848, 1.8599999999999994]).reshape(1,-1)

def load_model(suffix=""):
    global model
    model_name = "../RandomForestClassifier" + suffix + ".mdl"
    with open(model_name, 'rb') as f:
        print("Loading", model_name)
        model = pickle.load(f) 

def predict(x=x_att):
    x = np.array(x).reshape(1, -1)
    print(x)
    proba = model.predict_proba(x)
    # res = model.predict(x)    
    res = proba[0][1]
    print(res)
    return res