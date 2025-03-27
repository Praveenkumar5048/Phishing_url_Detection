import os
import pickle
import numpy as np
import torch
import validators
import traceback
from transformers import BertTokenizer, BertModel
from flask import Flask, request, render_template, jsonify


# Initialize Flask app
app = Flask(__name__)

# Load TinyBERT tokenizer & model (used for feature extraction)
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
bert_model = BertModel.from_pretrained("prajjwal1/bert-tiny").eval()

# Function to load a pickle file
def load_pickle_model(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Loaded model: {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ File not found: {model_path}")
    except pickle.UnpicklingError as e:
        print(f"❌ Unpickling error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error loading {model_path}:\n{traceback.format_exc()}")
    return None

# Load all models separately
catboost_model = load_pickle_model("catboost_model.pkl")
xgboost_model = load_pickle_model("xgboost_model.pkl")
lightgbm_model = load_pickle_model("lightgbm_model.pkl")
meta_model = load_pickle_model("meta_model.pkl")

# Ensure all models are loaded
if not all([catboost_model, xgboost_model, lightgbm_model, meta_model]):
    raise ValueError("❌ One or more models failed to load!")

# Function to extract features using TinyBERT
def extract_features(url):
    """Extract features from a given URL using TinyBERT."""
    inputs = tokenizer(url, truncation=True, padding="max_length", max_length=100, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return features


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input', methods=['GET'])
def input_page():
    return render_template('input.html')

@app.route('/output', methods=['GET'])
def output_page():
    return render_template('output.html')

@app.route('/analyze', methods=['POST'])
def analyze_url():
    # Validate if all models are available
    if not all([catboost_model, xgboost_model, lightgbm_model, meta_model]):
        return jsonify({
            'error': 'Phishing detection models are not available.'
        }), 500
    
    # Get URL from request
    url = request.form.get('url', '')
    
    # Validate URL format
    if not url or not validators.url(url):
        return jsonify({
            'error': 'Please provide a valid URL.'
        }), 400
    
    try:
        # Extract features from TinyBERT
        features = extract_features(url)

        # Get predictions from base models
        p1 = catboost_model.predict(features)
        p2 = xgboost_model.predict(features)
        p3 = lightgbm_model.predict(features)

        # Stack predictions and pass to meta-model
        meta_features = np.column_stack((p1, p2, p3))
        final_prediction = meta_model.predict(meta_features)[0]

        # Convert prediction to label
        prediction_label = "Phishing" if final_prediction == 0 else "Safe"

        return jsonify({
            'url': url,
            'prediction': prediction_label,
            'is_safe': prediction_label == 'Safe'
        })
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred during prediction: {str(e)}'
        }), 500

@app.route('/store', methods=['POST'])
def store_output():
    return jsonify({
        'message': 'Output stored successfully',
        'status': 'success'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)