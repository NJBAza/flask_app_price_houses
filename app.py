# Importing Dependencies
import os
import sys
from pathlib import Path

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
from packaging_ml_model.prediction_model.predict import predictions

app = Flask(__name__)

RANDOM_NUMBER = 16092023
ORIGINAL_FEATURES = config.ORIGINAL_FEATURES
CATEGORICAL_FEATURES = config.CATEGORICAL_FEATURES
NON_REQUIRED_FEATURES = ["uid", "priceRange", "MedianStudentsPerTeacher"]

NUMERICAL_FEATURES = [
    element
    for element in ORIGINAL_FEATURES
    if element not in CATEGORICAL_FEATURES and element not in NON_REQUIRED_FEATURES
]


# Views
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Extract all form data into request_data dictionary
        request_data = dict(request.form)

        # Example: Add specific fields manually (like 'textarea') if needed
        text = request.form.get("description")
        if text is not None:
            request_data["description"] = text.strip()  # Add the text field to request_data

        # Process numeric, boolean, and text fields as before
        numeric_fields = [
            "garageSpaces",
            "yearBuilt",
            "numOfPatioAndPorchFeatures",
            "lotSizeSqFt",
            "avgSchoolRating",
            "numOfBathrooms",
            "numOfBedrooms",
        ]
        boolean_fields = ["hasSpa"]
        text_fields = [
            "city",
            "homeType",
            "description",
            "textarea",
        ]  # Add 'textarea' if it's a text field

        for field in numeric_fields:
            if field in request_data:
                try:
                    request_data[field] = int(request_data[field])
                except ValueError:
                    return f"Invalid input for {field}: must be an integer", 400

        for field in boolean_fields:
            if field in request_data:
                request_data[field] = bool(int(request_data[field]))

        for field in text_fields:
            if field in request_data:
                request_data[field] = request_data[field].strip()
                if request_data[field] == "":
                    return f"Invalid input for {field}: cannot be empty", 400

        # Adding default values for missing fields
        if "MedianStudentsPerTeacher" not in request_data:
            request_data["MedianStudentsPerTeacher"] = RANDOM_NUMBER
        if "priceRange" not in request_data:
            request_data["priceRange"] = "ANY VALUE"

        # Convert the processed data to a DataFrame
        data = pd.DataFrame([request_data])

        # Assuming the predictions function is defined and handles transformations
        prediction_value = predictions(data)["Predictions"][0]

        # Interpret the prediction result
        if prediction_value == "0-250000":
            result = "The price of the house is up to 250,000 USD"
        elif prediction_value == "650000+":
            result = "The price of the house is greater than 650,000 USD"
        else:
            result = f"The price of the house is between {prediction_value} USD"

        return render_template("index.html", prediction=result)


@app.errorhandler(500)
def internal_error(error):
    return "500: Something went wrong"


@app.errorhandler(404)
def not_found(error):
    return "404: Page not found", 404


if __name__ == "__main__":
    app.run(debug=True)
