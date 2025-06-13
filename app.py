from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    input_array = np.array([data])
    output = model.predict(input_array)
    return render_template('index.html', prediction_text=f"Estimated House Price: ${round(output[0]*1000, 2)}")

if __name__ == '__main__':
    app.run(debug=True)
