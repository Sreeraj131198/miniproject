import pickle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Global variables for models
le_marital = None
le_education = None
le_contact = None
le_poutcome = None
normalisation = None
selector = None
model = None

def load_models():
    """Load all pickle files with error handling"""
    global le_marital, le_education, le_contact, le_poutcome, normalisation, selector, model
    
    try:
        print("Loading models...")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        
        with open('marital.pkl', 'rb') as f:
            le_marital = pickle.load(f)
            print("✓ Loaded marital.pkl")
        
        with open('education.pkl', 'rb') as f:
            le_education = pickle.load(f)
            print("✓ Loaded education.pkl")
        
        with open('contact.pkl', 'rb') as f:
            le_contact = pickle.load(f)
            print("✓ Loaded contact.pkl")
        
        with open('poutcome.pkl', 'rb') as f:
            le_poutcome = pickle.load(f)
            print("✓ Loaded poutcome.pkl")
        
        with open('scaling.pkl', 'rb') as f:
            normalisation = pickle.load(f)
            print("✓ Loaded scaling.pkl")
        
        with open('feature_selector.pkl', 'rb') as f:
            selector = pickle.load(f)
            print("✓ Loaded feature_selector.pkl")
        
        with open('RF_model.pkl', 'rb') as f:
            model = pickle.load(f)
            print("✓ Loaded RF_model.pkl")
        
        print("All models loaded successfully!")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file - {e}")
        return False
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def preprocess_and_predict(raw_data: dict):
    """Preprocess data and make prediction"""
    try:
        df = pd.DataFrame([raw_data])
        
        # Drop columns
        df.drop(columns=["day", "month", "job"], axis=1, inplace=True)
        
        # Encode categorical variables
        df['marital'] = le_marital.transform(df['marital'])
        df["education"] = le_education.transform(df["education"])
        df["contact"] = le_contact.transform(df["contact"])
        df["poutcome"] = le_poutcome.transform(df["poutcome"])
        
        # Convert yes/no to 1/0
        df['housing'] = 1 if df['housing'].iloc[0] == 'yes' else 0
        df["loan"] = 1 if df['loan'].iloc[0] == 'yes' else 0
        df['default'] = 1 if df['default'].iloc[0] == 'yes' else 0
        
        # Reorder columns
        expected_cols = normalisation.feature_names_in_
        df = df[expected_cols]
        
        # Scale and select features
        x_scaled = normalisation.transform(df)
        x_selected = selector.transform(x_scaled)
        
        # Make prediction
        prediction = model.predict(x_selected)
        prediction_proba = model.predict_proba(x_selected)
        
        result = "yes" if prediction[0] == 1 else "no"
        return result, prediction_proba[0].tolist()
    
    except Exception as e:
        print(f"Error in preprocess_and_predict: {e}")
        raise

@app.route('/')
def home():
    """Home page"""
    if model is None:
        return "Models not loaded. Please check server logs.", 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    data = request.get_json(force=True)
    
    try:
        prediction_result, probabilities = preprocess_and_predict(data)
        return jsonify({
            'prediction': prediction_result,
            'probability_no': probabilities[0],
            'probability_yes': probabilities[1]
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    if model is None:
        return jsonify({'status': 'unhealthy', 'message': 'Models not loaded'}), 500
    return jsonify({'status': 'healthy', 'message': 'All models loaded'})

if __name__ == '__main__':
    # Load models before starting the app
    if load_models():
        print("Starting Flask app...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load models. Exiting...")
        exit(1)