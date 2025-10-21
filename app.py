from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

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
    app.run(debug=True)
