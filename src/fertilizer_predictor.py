import pandas as pd
import numpy as np
import joblib
import os

ECO_ALTERNATIVES = {
    'Urea': {
        'eco_name'   : 'Neem-Coated Urea / Vermicompost',
        'eco_desc'   : 'Slow-release nitrogen, reduces leaching by 30%',
        'eco_score'  : 4,
        'cost_saving': '15-20%',
        'benefit'    : 'Improves soil organic matter',
    },
    'DAP': {
        'eco_name'   : 'Rock Phosphate + Bio-fertilizer',
        'eco_desc'   : 'Natural phosphate source with mycorrhizal fungi',
        'eco_score'  : 5,
        'cost_saving': '25-30%',
        'benefit'    : 'Long-lasting phosphorus release',
    },
    'MOP': {
        'eco_name'   : 'Wood Ash / Banana Peel Compost',
        'eco_desc'   : 'Natural potassium source, improves soil pH',
        'eco_score'  : 5,
        'cost_saving': '40-50%',
        'benefit'    : 'Free from farm waste!',
    },
    'NPK': {
        'eco_name'   : 'Compost + Green Manure',
        'eco_desc'   : 'Balanced nutrients from organic sources',
        'eco_score'  : 5,
        'cost_saving': '30-40%',
        'benefit'    : 'Improves soil structure long-term',
    },
    'SSP': {
        'eco_name'   : 'Bone Meal + Rock Phosphate',
        'eco_desc'   : 'Natural phosphate and sulfur source',
        'eco_score'  : 4,
        'cost_saving': '20-25%',
        'benefit'    : 'Provides calcium and sulfur too',
    },
    'Compost': {
        'eco_name'   : 'Vermicompost / Farm Yard Manure',
        'eco_desc'   : 'Already eco-friendly! Enhance with biofertilizers',
        'eco_score'  : 5,
        'cost_saving': '50-60%',
        'benefit'    : 'Best for long-term soil health',
    },
    'Zinc Sulphate': {
        'eco_name'   : 'Zinc Solubilizing Bacteria (ZSB)',
        'eco_desc'   : 'Bio-fertilizer that makes soil zinc available',
        'eco_score'  : 5,
        'cost_saving': '35-45%',
        'benefit'    : 'Improves zinc uptake naturally',
    },
}

def predict_fertilizer(soil_type, soil_ph, soil_moisture,
                        organic_carbon, ec, nitrogen,
                        phosphorus, potassium, temperature,
                        humidity, rainfall, crop_type,
                        growth_stage, season, irrigation,
                        previous_crop, region,
                        fert_last_season='Urea', yield_last_season=4.0):
    try:
        model        = joblib.load('models/fertilizer_model.pkl')
        encoders     = joblib.load('models/fertilizer_encoders.pkl')
        feature_cols = joblib.load('models/fertilizer_features.pkl')
        target_le    = encoders['__target__']
    except Exception as e:
        return {'error': str(e)}

    def safe_encode(value, col):
        le = encoders[col]
        if value in le.classes_:
            return le.transform([value])[0]
        return 0

    input_data = {
        'Soil_Type'               : safe_encode(soil_type, 'Soil_Type'),
        'Soil_pH'                 : soil_ph,
        'Soil_Moisture'           : soil_moisture,
        'Organic_Carbon'          : organic_carbon,
        'Electrical_Conductivity' : ec,
        'Nitrogen_Level'          : nitrogen,
        'Phosphorus_Level'        : phosphorus,
        'Potassium_Level'         : potassium,
        'Temperature'             : temperature,
        'Humidity'                : humidity,
        'Rainfall'                : rainfall,
        'Crop_Type'               : safe_encode(crop_type, 'Crop_Type'),
        'Crop_Growth_Stage'       : safe_encode(growth_stage, 'Crop_Growth_Stage'),
        'Season'                  : safe_encode(season, 'Season'),
        'Irrigation_Type'         : safe_encode(irrigation, 'Irrigation_Type'),
        'Previous_Crop'           : safe_encode(previous_crop, 'Previous_Crop'),
        'Region'                  : safe_encode(region, 'Region'),
        'Fertilizer_Used_Last_Season': safe_encode(previous_crop, 'Fertilizer_Used_Last_Season') if 'Fertilizer_Used_Last_Season' in encoders else 0,
        'Yield_Last_Season'       : 4.0,
    }

    input_df   = pd.DataFrame([input_data])[feature_cols]
    pred_idx   = model.predict(input_df)[0]
    pred_prob  = model.predict_proba(input_df)[0]
    confidence = round(max(pred_prob) * 100, 1)
    fertilizer = target_le.inverse_transform([pred_idx])[0]
    eco        = ECO_ALTERNATIVES.get(fertilizer, {})

    top3_idx = pred_prob.argsort()[-3:][::-1]
    top3 = [{'fertilizer': target_le.inverse_transform([i])[0],
              'probability': round(pred_prob[i] * 100, 1)}
             for i in top3_idx]

    return {
        'fertilizer' : fertilizer,
        'confidence' : confidence,
        'eco'        : eco,
        'top3'       : top3,
    }
