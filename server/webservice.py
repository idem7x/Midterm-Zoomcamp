import pickle

from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template

with open('../bin/model-randomforest.pkl', 'rb') as f_model:
    model = pickle.load(f_model)

with open('../bin/dv-randomforest.pkl', 'rb') as f_dv:
    dv = pickle.load(f_dv)

app = Flask('credit', template_folder='templates')
app.static_url_path = '/static'
app.static_folder = 'static'

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

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
