# In this file we will use the OneHotEncoder to encode the new data.

import json
import pandas as pd
import joblib

# Read the inference JSON file
data = pd.read_json("data/inference_data.json")

print(data)

# Load the OneHotEncoder
with open("encoders/ohe_encoder.pkl", "rb") as f:
    ohe = joblib.load(f)

# Apply the OneHotEncoder
data_encoded = ohe.transform(data)

print(data_encoded)