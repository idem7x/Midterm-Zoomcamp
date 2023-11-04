import pickle

from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template

with open('./bin/model-randomforest.pkl', 'rb') as f_model:
    model = pickle.load(f_model)

with open('./bin/dv-randomforest.pkl', 'rb') as f_dv:
    dv = pickle.load(f_dv)

with open('./bin/model-logistic.bin', 'rb') as f_model:
    modell = pickle.load(f_model)

with open('./bin/dv-logistic.bin', 'rb') as f_dv:
    dvl = pickle.load(f_dv)

app = Flask('credit', template_folder='server/templates')

@app.route('/')
def show_form():
    return render_template('predict.html')

@app.route('/predict', methods=["POST"])
def predict():
    client = request.get_json()
    X = dv.transform(client)
    print(X.toarray())
    y_pred = model.predict(X)[0]

    result = {
       'cancel': float(y_pred)
    }

    return jsonify(result)

@app.route('/proba', methods=["POST"])
def proba():
    client = request.get_json()
    X = dvl.transform([client])
    print(X)
    result = modell.predict_proba(X)[0, 1]

    result = {
       'cancel': float(result)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
