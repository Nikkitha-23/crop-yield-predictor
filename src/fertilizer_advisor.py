CROP_NPK = {
    'Rice':    {'N':(80,120),  'P':(40,60),  'K':(40,60),  'growth_days':120, 'water_need':'High',   'notes':'Split N - 50% basal, 50% at tillering'},
    'Wheat':   {'N':(60,100),  'P':(30,50),  'K':(30,50),  'growth_days':110, 'water_need':'Medium', 'notes':'Apply P and K at sowing, N in splits'},
    'Maize':   {'N':(100,150), 'P':(50,70),  'K':(40,60),  'growth_days':95,  'water_need':'Medium', 'notes':'High N feeder - apply in 3 splits'},
    'Potato':  {'N':(120,180), 'P':(60,80),  'K':(80,120), 'growth_days':90,  'water_need':'High',   'notes':'High K requirement for tuber development'},
    'Soybean': {'N':(20,40),   'P':(40,60),  'K':(40,60),  'growth_days':100, 'water_need':'Medium', 'notes':'Fixes own N - minimal N fertilizer needed'},
    'Cotton':  {'N':(80,120),  'P':(40,60),  'K':(40,80),  'growth_days':160, 'water_need':'Medium', 'notes':'Apply micronutrients - Boron critical'},
    'Barley':  {'N':(60,90),   'P':(30,50),  'K':(30,50),  'growth_days':90,  'water_need':'Low',    'notes':'Drought tolerant - avoid excess N'},
}

SOIL_ADJUSTMENTS = {
    'Sandy':  {'N':1.2, 'P':1.0, 'K':1.3, 'reason':'Sandy soil leaches nutrients quickly - increase N and K'},
    'Clay':   {'N':0.9, 'P':1.2, 'K':0.9, 'reason':'Clay retains nutrients well - reduce N slightly'},
    'Loam':   {'N':1.0, 'P':1.0, 'K':1.0, 'reason':'Loam is ideal - standard application rates'},
    'Silt':   {'N':1.0, 'P':1.1, 'K':1.0, 'reason':'Good water retention - standard rates'},
    'Chalky': {'N':1.1, 'P':1.3, 'K':1.1, 'reason':'Alkaline pH reduces P availability - increase P'},
    'Peaty':  {'N':0.8, 'P':1.0, 'K':1.2, 'reason':'Rich in organic N - reduce N, increase K'},
}

def get_fertilizer_recommendation(crop, soil_type, rainfall, yield_target=None):
    crop = str(crop).strip()
    soil_type = str(soil_type).strip()

    if crop not in CROP_NPK:
        crop = 'Rice'
    if soil_type not in SOIL_ADJUSTMENTS:
        soil_type = 'Loam'

    base = CROP_NPK[crop]
    adj = SOIL_ADJUSTMENTS[soil_type]

    N_low  = round(base['N'][0] * adj['N'])
    N_high = round(base['N'][1] * adj['N'])
    P_low  = round(base['P'][0] * adj['P'])
    P_high = round(base['P'][1] * adj['P'])
    K_low  = round(base['K'][0] * adj['K'])
    K_high = round(base['K'][1] * adj['K'])

    if rainfall > 700:
        N_high = round(N_high * 1.1)
        K_high = round(K_high * 1.1)
        rain_note = "High rainfall - nutrients leach faster. Use split applications."
    elif rainfall < 300:
        N_low  = round(N_low  * 0.85)
        N_high = round(N_high * 0.85)
        rain_note = "Low rainfall - reduce N to avoid salt stress. Prioritize irrigation."
    else:
        rain_note = "Rainfall is in optimal range for nutrient uptake."

    return {
        'crop'       : crop,
        'soil_type'  : soil_type,
        'npk'        : {
            'N': (N_low,  N_high),
            'P': (P_low,  P_high),
            'K': (K_low,  K_high),
        },
        'products'   : {
            'Urea (46-0-0)'         : (round(N_low/0.46),  round(N_high/0.46)),
            'DAP (18-46-0)'         : (round(P_low/0.46),  round(P_high/0.46)),
            'Muriate of Potash MOP' : (round(K_low/0.60),  round(K_high/0.60)),
        },
        'timing'     : {
            'Basal (at sowing)' : 'Full P + K + 30% N',
            '4 weeks later'     : '40% N as top dress',
            'At flowering'      : 'Remaining 30% N',
        },
        'soil_note'  : adj['reason'],
        'rain_note'  : rain_note,
        'crop_notes' : base['notes'],
        'water_need' : base['water_need'],
        'growth_days': base['growth_days'],
    }