"""
Crop Recommender Module
Scores each crop based on farm conditions
and returns ranked recommendations.
"""

# ── Crop ideal conditions ─────────────────────────────
# Each crop has ideal ranges for each factor
# Format: (min, optimal_low, optimal_high, max)
CROP_CONDITIONS = {
    'Rice': {
        'rainfall'   : (150, 200, 300, 500),
        'temperature': (20,  24,  32,  38),
        'humidity'   : (60,  75,  90,  100),
        'soil_types' : ['Clay', 'Loam', 'Silt'],
        'regions'    : ['North', 'South', 'East', 'West'],
        'irrigation' : 'High',
        'description': 'Staple grain, loves water and warmth',
        'emoji'      : '🌾',
    },
    'Wheat': {
        'rainfall'   : (50,  75,  120, 250),
        'temperature': (10,  15,  25,  30),
        'humidity'   : (40,  50,  70,  80),
        'soil_types' : ['Loam', 'Clay', 'Silt'],
        'regions'    : ['North', 'South', 'East', 'West'],
        'irrigation' : 'Medium',
        'description': 'Cool season crop, moderate water needs',
        'emoji'      : '🌿',
    },
    'Maize': {
        'rainfall'   : (80,  100, 180, 300),
        'temperature': (18,  21,  30,  35),
        'humidity'   : (50,  55,  75,  85),
        'soil_types' : ['Loam', 'Sandy', 'Silt'],
        'regions'    : ['North', 'South', 'East', 'West'],
        'irrigation' : 'Medium',
        'description': 'Versatile crop, good for warm climates',
        'emoji'      : '🌽',
    },
    'Cotton': {
        'rainfall'   : (60,  90,  120, 200),
        'temperature': (21,  25,  35,  42),
        'humidity'   : (40,  50,  70,  80),
        'soil_types' : ['Sandy', 'Loam', 'Chalky'],
        'regions'    : ['North', 'South', 'East', 'West'],
        'irrigation' : 'Medium',
        'description': 'Cash crop, loves heat and dry conditions',
        'emoji'      : '🌸',
    },
    'Soybean': {
        'rainfall'   : (80,  100, 180, 300),
        'temperature': (20,  22,  30,  35),
        'humidity'   : (55,  60,  80,  90),
        'soil_types' : ['Loam', 'Clay', 'Silt'],
        'regions'    : ['North', 'South', 'East', 'West'],
        'irrigation' : 'Medium',
        'description': 'Protein-rich legume, fixes soil nitrogen',
        'emoji'      : '🫘',
    },
    'Barley': {
        'rainfall'   : (40,  60,  100, 200),
        'temperature': (10,  15,  22,  30),
        'humidity'   : (35,  45,  65,  75),
        'soil_types' : ['Sandy', 'Loam', 'Chalky', 'Peaty'],
        'regions'    : ['North', 'South', 'East', 'West'],
        'irrigation' : 'Low',
        'description': 'Drought tolerant, suits dry cool areas',
        'emoji'      : '🌱',
    },
}

# ── Scoring weights ────────────────────────────────────
WEIGHTS = {
    'rainfall'   : 0.35,   # most important
    'temperature': 0.30,   # very important
    'soil_type'  : 0.20,   # important
    'humidity'   : 0.15,   # moderate
}


def score_factor(value, min_val, opt_low, opt_high, max_val):
    """
    Scores a single factor from 0 to 100.
    100 = perfectly in optimal range
    0   = completely outside viable range
    """
    if value < min_val or value > max_val:
        return 0
    elif opt_low <= value <= opt_high:
        return 100
    elif value < opt_low:
        return int(((value - min_val) /
                    (opt_low - min_val)) * 100)
    else:
        return int(((max_val - value) /
                    (max_val - opt_high)) * 100)


def get_suitability_label(score):
    """Returns label and emoji based on score."""
    if score >= 80:
        return "Excellent", "🟢", "#1B5E20"
    elif score >= 60:
        return "Good",      "🟡", "#F57F17"
    elif score >= 40:
        return "Average",   "🟠", "#E65100"
    else:
        return "Poor",      "🔴", "#B71C1C"


def recommend_crops(rainfall, temperature,
                    soil_type, humidity=65,
                    irrigation=True):
    """
    Scores all crops and returns ranked recommendations.

    Args:
        rainfall    : Annual rainfall in mm
        temperature : Average temperature in Celsius
        soil_type   : Soil type string
        humidity    : Relative humidity %
        irrigation  : Boolean — is irrigation available?

    Returns:
        List of crop recommendations sorted by score
    """
    recommendations = []

    for crop_name, conditions in CROP_CONDITIONS.items():
        # Score each factor
        rain_score = score_factor(
            rainfall, *conditions['rainfall']
        )
        temp_score = score_factor(
            temperature, *conditions['temperature']
        )
        humid_score = score_factor(
            humidity, *conditions['humidity']
        )

        # Soil type score
        soil_score = (100 if soil_type in
                      conditions['soil_types'] else 30)

        # Weighted total score
        total_score = int(
            rain_score   * WEIGHTS['rainfall'] +
            temp_score   * WEIGHTS['temperature'] +
            soil_score   * WEIGHTS['soil_type'] +
            humid_score  * WEIGHTS['humidity']
        )

        # Irrigation penalty
        if not irrigation and conditions['irrigation'] == 'High':
            total_score = int(total_score * 0.6)

        label, emoji_dot, color = get_suitability_label(
            total_score
        )

        recommendations.append({
            'crop'        : crop_name,
            'emoji'       : conditions['emoji'],
            'score'       : total_score,
            'label'       : label,
            'emoji_dot'   : emoji_dot,
            'color'       : color,
            'description' : conditions['description'],
            'rain_score'  : rain_score,
            'temp_score'  : temp_score,
            'soil_score'  : soil_score,
            'humid_score' : humid_score,
            'irrigation'  : conditions['irrigation'],
            'soil_match'  : soil_type in conditions['soil_types'],
        })

    # Sort by score descending
    recommendations.sort(
        key=lambda x: x['score'], reverse=True
    )
    return recommendations


if __name__ == "__main__":
    results = recommend_crops(
        rainfall=500,
        temperature=28,
        soil_type='Loam',
        humidity=70,
        irrigation=True
    )
    print("=" * 50)
    print("CROP RECOMMENDATIONS")
    print("=" * 50)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['emoji']} {r['crop']:<10} "
              f"Score: {r['score']:>3}/100  "
              f"{r['label']}")
        