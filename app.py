import os
import pickle
import numpy as np
import torch
import validators
import traceback
from transformers import BertTokenizer, BertModel
from flask import Flask, request, render_template, jsonify

class PhishingDetector:
    def __init__(self, catboost_model, xgboost_model, lightgbm_model, meta_model):
        # Load Tiny BERT Tokenizer & Model
        self.tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.bert_model = BertModel.from_pretrained("prajjwal1/bert-tiny").eval()
        
        # Load the base models
        self.catboost = catboost_model
        self.xgboost = xgboost_model
        self.lightgbm = lightgbm_model
        
        # Load the meta-model
        self.meta = meta_model
    
    def extract_features(self, url):
        inputs = self.tokenizer(url, truncation=True, padding="max_length", max_length=100, return_tensors="pt")
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Convert to NumPy
        return features
    
    def predict(self, url):
        # Step 1: Extract features from Tiny BERT
        features = self.extract_features(url)
        
        # Step 2: Get predictions from base models
        p1 = self.catboost.predict(features)
        p2 = self.xgboost.predict(features)
        p3 = self.lightgbm.predict(features)
        
        # Step 3: Stack base model predictions and feed into meta-model
        meta_features = np.column_stack((p1, p2, p3))
        final_prediction = self.meta.predict(meta_features)[0]  # Get final output
        
        return "Phishing" if final_prediction == 0 else "Safe"

# Load the phishing detector model
def load_model():
    # Print current working directory and potential model paths
    print("Current Working Directory:", os.getcwd())
    
    # List multiple potential paths
    potential_paths = [
        "phishing_detector.pkl",
        "Phishing_Detection/phishing_detector.pkl",
        os.path.join(os.path.dirname(__file__), "phishing_detector.pkl"),
        os.path.join(os.path.dirname(__file__), "Phishing_Detection/phishing_detector.pkl")
    ]
    
    for model_path in potential_paths:
        try:
            print(f"Attempting to load model from: {model_path}")
            print(f"Absolute path: {os.path.abspath(model_path)}")
            
            # Check if file exists
            if not os.path.exists(model_path):
                print(f"File does not exist: {model_path}")
                continue
            
            # Try to load the model
            with open(model_path, "rb") as f:
                detector = pickle.load(f)
            
            print(f"Model successfully loaded from {model_path}")
            return detector
        
        except FileNotFoundError:
            print(f"File not found: {model_path}")
        except pickle.UnpicklingError as e:
            print(f"Unpickling error for {model_path}: {e}")
        except Exception as e:
            print(f"Unexpected error loading model from {model_path}:")
            print(traceback.format_exc())
    
    print("Failed to load model from all potential paths")
    return None

# Initialize Flask app
app = Flask(__name__)

# Load the model globally
try:
    detector = load_model()
    if detector is None:
        raise ValueError("Model could not be loaded")
except Exception as e:
    print(f"Error loading model: {e}")
    detector = None

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
    # Check if model is loaded
    if detector is None:
        return jsonify({
            'error': 'Phishing detection model is not available.'
        }), 500
    
    # Get URL from request
    url = request.form.get('url', '')
    
    # Validate URL
    if not url or not validators.url(url):
        return jsonify({
            'error': 'Please provide a valid URL.'
        }), 400
    
    try:
        # Predict URL safety
        prediction = detector.predict(url)
        
        # Return prediction
        return jsonify({
            'url': url,
            'prediction': prediction,
            'is_safe': prediction == 'Safe'
        })
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred during prediction: {str(e)}'
        }), 500

@app.route('/store', methods=['POST'])
def store_output():
    # In a real-world scenario, you'd implement actual storage logic
    # For this example, we'll just return a success message
    return jsonify({
        'message': 'Output stored successfully',
        'status': 'success'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)