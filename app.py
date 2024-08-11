# Importing Dependencies
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
)

# Assuming 'prediction_model' is in the parent directory
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT.parent))
print(PACKAGE_ROOT)

from packaging_ml_model.prediction_model.config import config
from packaging_ml_model.prediction_model.processing.data_handling import load_pipeline

app = Flask(__name__)

classification_pipeline = load_pipeline(config.MODEL_NAME)


# Views
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if request.method == "POST":
        request_data = dict(request.form)
        del request_data["First_Name"]
        del request_data["Last_Name"]
        request_data = {k: int(v) for k, v in request_data.items()}
        data = pd.DataFrame([request_data])
        prediction = classification_pipeline.predict(data)
        prediction_value = prediction[0]
        # print(f"prediction is {prediction}")

        if int(prediction_value) == 0:
            result = "The price of the house is up 250000 USD"
        if int(prediction_value) == 1:
            result = "The price of the house is between 250000 and 350000 USD"
        if int(prediction_value) == 2:
            result = "The price of the house is between 350000 and 450000 USD"
        if int(prediction_value) == 3:
            result = "The price of the house is between 450000 and 650000 USD"
        if int(prediction_value) == 4:
            result = "The price of the house is bigger than 350000 USD"

        return render_template("index.html", prediction=result)


@app.errorhandler(500)
def internal_error(error):
    return "500: Something went wrong"


@app.errorhandler(404)
def not_found(error):
    return "404: Page not found", 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)


# data_test = [
#     "austin",
#     "A large set of bedrooms with modern amenities",
#     "Single Family",
#     30.2672,
#     -97.7431,
#     2,
#     True,
#     2010,
#     3,
#     8000,
#     8,
#     15,
#     3,
#     4,
# ]
