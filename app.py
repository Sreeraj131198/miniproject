import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, render_template

# Load models
with open('marital.pkl', 'rb') as f:
    le_marital = pickle.load(f)
with open('education.pkl', 'rb') as f:
    le_education = pickle.load(f)
with open('contact.pkl', 'rb') as f:
    le_contact = pickle.load(f)
with open('poutcome.pkl', 'rb') as f:
    le_poutcome = pickle.load(f)
with open('scaling.pkl', 'rb') as f:
    normalisation = pickle.load(f)
with open('feature_selector.pkl', 'rb') as f:
    selector = pickle.load(f)
with open('RF_model.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_and_predict(raw_data: dict):
    df = pd.DataFrame([raw_data])
    df.drop(columns=["day", "month", "job"], axis=1, inplace=True)
    df['marital'] = le_marital.transform(df['marital'])
    df["education"] = le_education.transform(df["education"])
    df["contact"] = le_contact.transform(df["contact"])
    df["poutcome"] = le_poutcome.transform(df["poutcome"])
    df['housing'] = 1 if df['housing'].iloc[0] == 'yes' else 0
    df["loan"] = 1 if df['loan'].iloc[0] == 'yes' else 0
    df['default'] = 1 if df['default'].iloc[0] == 'yes' else 0
    expected_cols = normalisation.feature_names_in_
    df = df[expected_cols]
    x_scaled = normalisation.transform(df)
    x_selected = selector.transform(x_scaled)
    prediction = model.predict(x_selected)
    prediction_proba = model.predict_proba(x_selected)
    result = "yes" if prediction[0] == 1 else "no"
    return result, prediction_proba[0].tolist()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        prediction_result, probabilities = preprocess_and_predict(data)
        return jsonify({
            'prediction': prediction_result,
            'probability_no': probabilities[0],
            'probability_yes': probabilities[1]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000)