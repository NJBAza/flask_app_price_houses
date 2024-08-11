# Importing Dependencies
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Assuming 'prediction_model' is in the parent directory
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT.parent))
print(PACKAGE_ROOT)

from packaging_ml_model.prediction_model.config import config
from packaging_ml_model.prediction_model.predict import predictions
from packaging_ml_model.prediction_model.processing.data_handling import (
    load_dataset,
    load_pipeline,
)

test_data = load_dataset(config.TEST_FILE)
test_data[:1]
print(test_data[:1])

classification_pipeline = load_pipeline(config.MODEL_NAME)
inputs = [
    "austin",
    "A large set of bedrooms with modern amenities",
    "Single Family",
    30.2672,
    -97.7431,
    2,
    True,
    2010,
    3,
    8000,
    8,
    15,
    3,
    4,
]

dictionary = {
    "uid": 20230916,
    "city": "austin",
    "description": "A small place with modern amenities",
    "homeType": "Single Family",
    "latitude": 30.2672,
    "longitude": -97.7431,
    "garageSpaces": 2,
    "hasSpa": False,
    "yearBuilt": 2012,
    "numOfPatioAndPorchFeatures": 3,
    "lotSizeSqFt": 4000,
    "avgSchoolRating": 2,
    "MedianStudentsPerTeacher": 30,
    "numOfBathrooms": 1,
    "numOfBedrooms": 2,
    "priceRange": "any",
}

data = pd.DataFrame([dictionary])

print(data)
print(predictions(test_data[:1]))
print(predictions(data)["Predictions"][0])
# print(predictions([dictionary]))

# def single_prediction():
#     test_data = load_dataset(config.TEST_FILE)
#     single_row = test_data[:1]
#     return predictions(single_row)


# len(input)

# len(config.ORIGINAL_FEATURES)
# data = pd.DataFrame(data=inputs, columns=config.ORIGINAL_FEATURES)
# print(data)
# # prediction = classification_pipeline.predict(data)
# # print(prediction)


# # def predict(inputs):
# #     data = pd.DataFrame(inputs)
# #     prediction = classification_pipeline.predict(data)
# #     prediction_value = prediction[0]
# #     # print(f"prediction is {prediction}")

# #     if int(prediction_value) == 0:
# #         result = "The price of the house is up 250000 USD"
# #     if int(prediction_value) == 1:
# #         result = "The price of the house is between 250000 and 350000 USD"
# #     if int(prediction_value) == 2:
# #         result = "The price of the house is between 350000 and 450000 USD"
# #     if int(prediction_value) == 3:
# #         result = "The price of the house is between 450000 and 650000 USD"
# #     if int(prediction_value) == 4:
# #         result = "The price of the house is bigger than 350000 USD"

# #     return print(result)


# # inputs = [
# #     "austin",
# #     "A large set of bedrooms with modern amenities",
# #     "Single Family",
# #     30.2672,
# #     -97.7431,
# #     2,
# #     True,
# #     2010,
# #     3,
# #     8000,
# #     8,
# #     15,
# #     3,
# #     4,
# # ]

# # predict(inputs=input)
# # # data_test = [
# # #     "austin",
# # #     "A large set of bedrooms with modern amenities",
# # #     "Single Family",
# # #     30.2672,
# # #     -97.7431,
# # #     2,
# # #     True,
# # #     2010,
# # #     3,
# # #     8000,
# # #     8,
# # #     15,
# # #     3,
# # #     4,
# # # ]
