from flask import Flask, render_template, redirect
import pandas as pd
from flask_pymongo import PyMongo
import dog_breed_predictor

# Setup app
app = Flask(__name__)

mongo = PyMongo(app, uri="mongodb://localhost:27017/dog_predictor_app")

@app.route('/')
def index ():

    return render_template("index.html")

@app.route('/predict')
def predict():

    return redirect('/', code = 302)

@app.route("/uploads/<path:filename>")
def get_upload(filename):
    return mongo.send_file(filename)

@app.route("/uploads/<path:filename>", methods=["POST"])
def save_upload(filename):
    mongo.save_file(filename, request.files["file"])
    return redirect("/predict", filename=filename)



if __name__ == '__main__':
    app.run(debug=True)
