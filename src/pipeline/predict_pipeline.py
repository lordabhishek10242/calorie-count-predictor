import pandas as pd
import os
import pickle

class CustomData:
    def __init__(self, gender, age, height, weight, duration, heart_rate, body_temp):
        self.gender = gender
        self.age = age
        self.height = height
        self.weight = weight
        self.duration = duration
        self.heart_rate = heart_rate
        self.body_temp = body_temp

    def get_data_as_data_frame(self):
        return pd.DataFrame({
            "Gender": [self.gender],
            "Age": [self.age],
            "Height": [self.height],
            "Weight": [self.weight],
            "Duration": [self.duration],
            "Heart_Rate": [self.heart_rate],
            "Body_Temp": [self.body_temp]
        })

class PredictPipeline:
    def __init__(self):
        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

    def predict(self, features):
        data = self.preprocessor.transform(features)
        return self.model.predict(data)
