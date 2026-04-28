import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

def setup():
    if os.path.exists('models/random_forest.pkl'):
        print("Model already exists!")
        return

    print("Generating model...")
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    np.random.seed(42)
    n = 50000
    crops    = ['Rice','Wheat','Maize','Soybean','Cotton','Barley']
    regions  = ['North','South','East','West']
    soils    = ['Sandy','Clay','Loam','Silt','Chalky','Peaty']
    weathers = ['Sunny','Rainy','Cloudy','Windy','Stormy']

    rows = []
    for _ in range(n):
        crop    = np.random.choice(crops)
        region  = np.random.choice(regions)
        soil    = np.random.choice(soils)
        weather = np.random.choice(weathers)
        rain    = np.random.uniform(100, 1000)
        temp    = np.random.uniform(10, 45)
        fert    = np.random.choice([True, False])
        irrig   = np.random.choice([True, False])
        days    = np.random.randint(30, 180)

        yield_val = (
            rain * 0.006
            + (1 if fert else 0) * 1.8
            + (1 if irrig else 0) * 1.2
            - abs(temp - 25) * 0.05
            + np.random.normal(0, 0.3)
        )
        yield_val = max(0.1, round(yield_val, 3))
        rows.append([region, soil, crop, round(rain,1),
                    round(temp,1), fert, irrig,
                    weather, days, yield_val])

    df = pd.DataFrame(rows, columns=[
        'Region','Soil_Type','Crop','Rainfall_mm',
        'Temperature_Celsius','Fertilizer_Used',
        'Irrigation_Used','Weather_Condition',
        'Days_to_Harvest','Yield_tons_per_hectare'
    ])

    df = df[df['Yield_tons_per_hectare'] > 0]
    df['Fertilizer_Used'] = df['Fertilizer_Used'].astype(int)
    df['Irrigation_Used'] = df['Irrigation_Used'].astype(int)

    encoders = {}
    for col in ['Region','Soil_Type','Crop','Weather_Condition']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    df['rainfall_temp_ratio'] = df['Rainfall_mm'] / (df['Temperature_Celsius'] + 1)
    df['farming_score']       = df['Fertilizer_Used'] + df['Irrigation_Used']
    df['fert_irrig_combined'] = df['Fertilizer_Used'] * df['Irrigation_Used']
    df['rainfall_category']   = pd.cut(df['Rainfall_mm'],
                                bins=[0,400,700,1100],
                                labels=[0,1,2]).astype(int)
    df['temp_category']       = pd.cut(df['Temperature_Celsius'],
                                bins=[0,20,30,50],
                                labels=[0,1,2]).astype(int)

    X = df.drop('Yield_tons_per_hectare', axis=1)
    y = df['Yield_tons_per_hectare']

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X, y)

    feature_names = list(X.columns)
    joblib.dump(model,         'models/random_forest.pkl')
    joblib.dump(encoders,      'models/encoders.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print("Model generated and saved!")

if __name__ == "__main__":
    setup()