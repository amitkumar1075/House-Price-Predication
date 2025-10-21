from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import logging

app = Flask(__name__)

# Load trained model
model = None
try:
	# use path relative to this file so Render finds model.pkl in the repo root
	model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
	with open(model_path, 'rb') as f:
		model = pickle.load(f)
except Exception as e:
	logging.exception(f"Failed to load model from {model_path}: {e}")

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form se values le lo
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Prediction 
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        
        return render_template('index.html', prediction_text=f'Predicted Price: ${output} (x100000)')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
	# Use PORT provided by Render, default to 5000 for local dev
	port = int(os.environ.get("PORT", 5000))
	app.run(host="0.0.0.0", port=port, debug=False)
