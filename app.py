import json
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

#from TrainedModel import mymodel

import chatbot as chatbot

app = Flask(__name__)
CORS(app)

# Configure a folder to store uploaded images
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/ChatBot")
def chatboat():
    return render_template('chatbot.html')


@app.route("/contact")
def contact():
    return render_template('contact.html')


@app.route("/classify")
def classify():
    return render_template('classify.html')


@app.route('/solve', methods=['POST'])
def solve():
    prompt = request.form['query']
    response = chatbot.generate_response(prompt)
    return response

import mymodel

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    # If the user does not select a file, browser may also submit an empty part without filename
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    print("Predicting on a normal chest X-ray:")

    if file:
        # Save the file to the UPLOAD_FOLDER
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        filepath = "upload/"+filename
        print("file path:", filepath)

        # Call the predict_image function from model.py
        prediction, accuracy = mymodel.predict_image(filepath)

        return jsonify({"prediction": prediction, "accuracy": accuracy})
    
if __name__ == "__main__":
    app.run(debug=True)
