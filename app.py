from flask import Flask, render_template, redirect, request, jsonify
import pandas as pd
from flask_pymongo import PyMongo
import dog_breed_predictor

# Setup app
app = Flask(__name__)

mongo = PyMongo(app, uri="mongodb://localhost:27017/dog_predictor_app")


@app.route('/')
def index():

    return render_template("index.html")


@app.route('/predict')
def predict():
    # dog_predictor = dog_predictor.predict()
    return redirect('/', code=302)


# @app.route("/uploads/<path:filename>")
# def get_upload(filename):
#     return mongo.send_file(filename)


#  Only need to handle POST requests here
# -----------------------------------------------v
@app.route("/upload", methods=["POST"])
def save_upload():
    # https://stackoverflow.com/a/46136495/1175496
    # > Once you fetch the actual file with file = request.files['file'], you can get the filename with file.filename.
    # use same as the name attribute of the HTML <input>, which is "image"
    # -----------------------------v
    image_file = request.files["image"]
    # mongo.save_file(image_file.filename, image_file)

    # return redirect("/predict", code=302)
    # er.py", line 273, in default
    #     o.__class__.__name__)
    # TypeError: Object of type float32 is not JSON serializable
    # https://keras.io/applications/
    # >  decode the results into a list of tuples (class, description, probability)
    return pd.DataFrame(dog_breed_predictor.predict_from_file(image_file), columns=('class', 'description', 'probability')).to_json()


if __name__ == '__main__':
    # https://github.com/jrosebr1/simple-keras-rest-api/issues/5#issuecomment-413461944
    # Nate changed to avoid builtins.ValueError
    # ValueError: Tensor Tensor("predictions/Softmax:0", shape=(?, 1000), dtype=float32) is not an element of this graph.
    app.run(debug=False, threaded=False)
