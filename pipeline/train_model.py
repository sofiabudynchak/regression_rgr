
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib


data = pd.read_csv("/Users/sonyabudynchak/Desktop/навчання/ІТС/ргр/regression_rgr/data/train.csv")

# Визначення категоріальних ознак
categorical_features = ['region', 'sex', 'smoker']

# Визначення pipeline для кодування категоріальних ознак та навчання моделі
pipeline = Pipeline(steps=[
('encoder', ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_features)])),
('regressor', RandomForestRegressor(random_state=42))
])

# Розділення на ознаки та цільову змінну
X = data.drop(columns=["charges"])
Y = data["charges"]
    
# Навчання моделі
pipeline.fit(X, Y)

joblib.dump(pipeline, "/Users/sonyabudynchak/Desktop/навчання/ІТС/ргр/regression_rgr/models/random_forest_model.pkl")