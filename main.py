from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('red_wine_regression.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home2')
def home2():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_names = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
            'density', 'ph', 'sulphates', 'alcohol'
        ]

        # Collect input values and convert to float
        input_data = {name: request.form[name] for name in feature_names}
        features = [float(input_data[name]) for name in feature_names]

        # Predict
        prediction = model.predict([np.array(features)])
        result = round(prediction[0], 2)

        return render_template('index2.html', prediction=f'Predicted Wine Quality: {result}', values=input_data)

    except Exception as e:
        return render_template('index2.html', prediction=f'Error: {str(e)}', values=request.form)

if __name__ == '__main__':
    app.run(debug=True)
