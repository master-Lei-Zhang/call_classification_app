from flask import Flask, jsonify, request, render_template
import json
import numpy as np
from pandas import DataFrame
import pickle

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from tools import FeatureSelector, dayofweek_transformer



with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = []
    d = None
    if request.method == 'POST':
        d = request.form.to_dict()
        df = DataFrame([d.values()], columns=d.keys())
        pred = model.predict(df)[0]+1
    return render_template('index.html', pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)