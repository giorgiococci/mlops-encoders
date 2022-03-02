import pandas as pd
import category_encoders as ce
import os
import joblib

from sklearn.preprocessing import OneHotEncoder

# read data from csv file
data = pd.read_csv('data/train_data.csv', sep=";")

# use one-hot encoding
ohe = ce.OneHotEncoder(cols=["Gender", "Degree"])

# encode the data
data_encoded = ohe.fit_transform(data)

print(data_encoded)

# Save the OneHotEncoder to use it in a second step
encoder_folder = "encoders"
os.makedirs(encoder_folder, exist_ok=True)
ohe_encoder_path = os.path.join(encoder_folder, f"ohe_encoder.pkl")

with open(ohe_encoder_path, 'wb') as f:
    joblib.dump(ohe, f)
    f.close()
