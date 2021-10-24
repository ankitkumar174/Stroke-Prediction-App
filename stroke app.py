import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

app = Flask(__name__)
model = pickle.load(open('Stroke_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('new.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    input = scaler.fit_transform(final_features)
    prediction = model.predict(input)

    if prediction==0:
            return render_template('new.html', prediction_text='You are Safe - No chances of Stroke')
    else:
            return render_template('new.html', prediction_text='You are in Danger - High chances of Stroke prediction')


if __name__ == "__main__":
    app.run(debug=True)