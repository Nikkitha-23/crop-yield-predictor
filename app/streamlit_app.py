import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
# Auto setup model on cloud
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup
setup.setup()
sys.path.append('src')
from fertilizer_advisor import get_fertilizer_recommendation
from weather_api import get_current_weather, get_weather_farming_advice
from crop_recommender import recommend_crops
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY", "")


# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="🌾 Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border-left: 5px solid #2E7D32;
    }
    .metric-card h3 { color: #1B5E20; margin: 0; font-size: 1rem; }
    .metric-card h2 { color: #1B5E20; margin: 0.3rem 0 0 0; }
    .yield-result {
        background: linear-gradient(135deg, #1B5E20, #2E7D32);
        color: white;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .yield-number {
        font-size: 3.5rem;
        font-weight: bold;
        color: #A5D6A7;
    }
    .insight-box {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .insight-excellent { background:#1B5E20; color:#A5D6A7; }
    .insight-good      { background:#2E7D32; color:#C8E6C9; }
    .insight-average   { background:#E65100; color:#FFE0B2; }
    .insight-low       { background:#B71C1C; color:#FFCDD2; }
    .confidence-box {
        background: #263238;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        text-align: center;
        color: #90A4AE;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    .stButton > button {
        background-color: #2E7D32;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1.1rem;
        border: none;
        width: 100%;
    }
    .stButton > button:hover { background-color: #1B5E20; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load('models/random_forest.pkl')
    encoders      = joblib.load('models/encoders.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, encoders, feature_names

model, encoders, feature_names = load_model()

# Average yield per crop for comparison chart
CROP_AVG_YIELD = {
    'Barley': 4.63, 'Cotton': 4.67, 'Maize': 4.65,
    'Rice': 4.66,   'Soybean': 4.66, 'Wheat': 4.65
}

def encode_input(value, column):
    le = encoders[column]
    return le.transform([value])[0] if value in le.classes_ else 0

def get_insight(yield_val):
    if yield_val >= 7.0:
        return "🌟 Exceptional yield expected!", "insight-excellent"
    elif yield_val >= 5.0:
        return "✅ Good yield expected!", "insight-good"
    elif yield_val >= 3.0:
        return "⚠️ Average yield — consider more inputs", "insight-average"
    else:
        return "❌ Low yield — review conditions carefully", "insight-low"


# ── Header ────────────────────────────────────────────
st.markdown('<div class="main-header">🌾 Crop Yield Predictor</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered yield prediction '
            'for smarter farming decisions</div>',
            unsafe_allow_html=True)

# ── Stats banner ──────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, title, value in zip(
    [c1, c2, c3, c4],
    ["🎯 R² Score", "📏 Avg Error", "🌲 Model", "📊 Trained On"],
    ["91.15%", "±0.40 t/ha", "Random Forest", "100K Rows"]
):
    col.markdown(f'<div class="metric-card"><h3>{title}</h3>'
                 f'<h2>{value}</h2></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar — About ───────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 How to use")
    st.info(
        "1. Select your farm details\n"
        "2. Adjust environmental sliders\n"
        "3. Toggle fertilizer & irrigation\n"
        "4. Click **Predict Crop Yield**\n"
        "5. View results & improvement tips"
    )
    st.markdown("## 🌡️ Optimal Ranges")
    st.markdown("""
    | Factor | Optimal |
    |--------|---------|
    | Rainfall | 400-800mm |
    | Temperature | 20-30°C |
    | Days | 60-150 |
    """)
    st.markdown("## 📌 Model Info")
    st.success("Random Forest\n\n100 trees | Depth 15\n\nR²: 91.15%")

# ── Input Form ────────────────────────────────────────
st.subheader("🌱 Enter Farm Details")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**📍 Farm Information**")
    region    = st.selectbox("Region",
                    list(encoders['Region'].classes_))
    soil_type = st.selectbox("Soil Type",
                    list(encoders['Soil_Type'].classes_))
    crop      = st.selectbox("Crop",
                    list(encoders['Crop'].classes_))
    weather   = st.selectbox("Weather Condition",
                    list(encoders['Weather_Condition'].classes_))

with col_right:
    st.markdown("**🌡️ Environmental Conditions**")
    rainfall    = st.slider("Rainfall (mm)",
                    100.0, 1000.0, 500.0, 10.0)
    temperature = st.slider("Temperature (°C)",
                    10.0, 45.0, 25.0, 0.5)
    days        = st.slider("Days to Harvest",
                    30, 180, 90, 5)

    st.markdown("**🚜 Farming Practices**")
    fc, ic = st.columns(2)
    with fc:
        fertilizer = st.toggle("Fertilizer Used", value=True)
    with ic:
        irrigation = st.toggle("Irrigation Used", value=True)

# ── Input validation warnings ─────────────────────────
warnings_shown = False
if rainfall < 200:
    st.warning("⚠️ Very low rainfall (< 200mm) — predictions may be less reliable. Consider adding irrigation.")
    warnings_shown = True
if temperature > 40:
    st.warning("⚠️ Very high temperature (> 40°C) — most crops struggle above 40°C.")
    warnings_shown = True
if days < 45:
    st.warning("⚠️ Very short growing period (< 45 days) — verify this is realistic for your crop.")
    warnings_shown = True

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Yield Predictor",
    "🌿 Fertilizer Advisor",
    "🌦️ Live Weather",
    "🌾 Crop Recommender"
])
with tab1:
    predict_clicked = st.button("🔍 Predict Crop Yield")

if predict_clicked:
    # ── Build features ────────────────────────────────
    rtr  = rainfall / (temperature + 1)
    fs   = int(fertilizer) + int(irrigation)
    fic  = int(fertilizer) * int(irrigation)
    rc   = 0 if rainfall <= 400 else (1 if rainfall <= 700 else 2)
    tc   = 0 if temperature <= 20 else (1 if temperature <= 30 else 2)

    input_dict = {
        'Region'              : encode_input(region, 'Region'),
        'Soil_Type'           : encode_input(soil_type, 'Soil_Type'),
        'Crop'                : encode_input(crop, 'Crop'),
        'Rainfall_mm'         : rainfall,
        'Temperature_Celsius' : temperature,
        'Fertilizer_Used'     : int(fertilizer),
        'Irrigation_Used'     : int(irrigation),
        'Weather_Condition'   : encode_input(weather, 'Weather_Condition'),
        'Days_to_Harvest'     : days,
        'rainfall_temp_ratio' : rtr,
        'farming_score'       : fs,
        'fert_irrig_combined' : fic,
        'rainfall_category'   : rc,
        'temp_category'       : tc,
    }

    input_df   = pd.DataFrame([input_dict])[feature_names]
    yield_pred = max(0.1, round(model.predict(input_df)[0], 3))
    MAE        = 0.40
    low_est    = max(0.0, round(yield_pred - MAE, 2))
    high_est   = round(yield_pred + MAE, 2)
    insight, css_class = get_insight(yield_pred)

    # ── Results ───────────────────────────────────────
    st.subheader("📊 Prediction Results")
    res1, res2 = st.columns([1, 1])

    with res1:
        st.markdown(f"""
        <div class="yield-result">
            <p style="font-size:1.1rem;margin:0;color:#C8E6C9">
                Predicted Yield</p>
            <div class="yield-number">{yield_pred:.2f}</div>
            <p style="font-size:1.2rem;margin:0">tons per hectare</p>
        </div>
        <div class="confidence-box">
            📊 Confidence range: <strong>{low_est} – {high_est} t/ha</strong>
            &nbsp;|&nbsp; Model MAE: ±0.40 t/ha
        </div>
        <div class="insight-box {css_class}" style="margin-top:0.8rem">
            {insight}
        </div>
        """, unsafe_allow_html=True)

    with res2:
        st.markdown("**📋 Input Summary**")
        summary = {
            "Crop": crop, "Region": region,
            "Soil Type": soil_type, "Weather": weather,
            "Rainfall": f"{rainfall} mm",
            "Temperature": f"{temperature}°C",
            "Fertilizer": "✅ Yes" if fertilizer else "❌ No",
            "Irrigation": "✅ Yes" if irrigation else "❌ No",
            "Days to Harvest": f"{days} days",
        }
        for k, v in summary.items():
            st.markdown(f"- **{k}:** {v}")

    # ── Comparison chart ──────────────────────────────
    st.markdown("---")
    st.subheader("📈 How Does Your Farm Compare?")

    avg_yield   = CROP_AVG_YIELD.get(crop, 4.65)
    chart_data  = pd.DataFrame({
    'Category': ['Dataset Max', f'{crop} Average', 'Your Farm'],
    'Yield'   : [9.53,          avg_yield,          yield_pred]
    })
    colors_map  = {
        'Your Farm'       : '#2E7D32',
        f'{crop} Average' : '#1565C0',
        'Dataset Max'     : '#E65100'
    }

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    bars = ax.barh(
        chart_data['Category'],
        chart_data['Yield'],
        color=[colors_map[c] for c in chart_data['Category']],
        edgecolor='none', height=0.5
    )
    for bar, val in zip(bars, chart_data['Yield']):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.2f} t/ha', va='center',
                color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel('Yield (tons/hectare)', color='#90A4AE')
    ax.tick_params(colors='white')
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.set_xlim(0, 11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Improvement tips ──────────────────────────────
    st.markdown("---")
    st.subheader("💡 Yield Improvement Tips")
    t1, t2, t3 = st.columns(3)

    with t1:
        if not fertilizer:
            st.warning("🌿 **Add Fertilizer**\n\n"
                      "Can increase yield by ~37%\n\n"
                      f"Estimated with fertilizer: "
                      f"**{min(9.5, yield_pred*1.37):.2f} t/ha**")
        else:
            st.success("🌿 **Fertilizer** ✅\n\nAlready applied!")

    with t2:
        if not irrigation:
            st.warning("💧 **Add Irrigation**\n\n"
                      "Can boost yield by ~25%\n\n"
                      f"Estimated with irrigation: "
                      f"**{min(9.5, yield_pred*1.25):.2f} t/ha**")
        else:
            st.success("💧 **Irrigation** ✅\n\nAlready active!")

    with t3:
        if rainfall < 400:
            st.warning("🌧️ **Low Rainfall**\n\n"
                      "Consider water-efficient crops "
                      "or increase irrigation frequency")
        elif rainfall > 800:
            st.info("🌧️ **High Rainfall**\n\n"
                   "Ensure good drainage to prevent waterlogging")
        else:
            st.success("🌧️ **Rainfall** ✅\n\nOptimal range!")
with tab2:
    st.subheader("🌿 Smart Fertilizer Advisor")
    st.markdown("Precise NPK recommendations based on your crop, soil and rainfall.")

    import sys
    sys.path.append('src')
    from fertilizer_advisor import get_fertilizer_recommendation

    rec = get_fertilizer_recommendation(crop, soil_type, rainfall)

    if rec:
        st.markdown("### 📊 NPK Requirements (kg/hectare)")
        n_col, p_col, k_col = st.columns(3)

        with n_col:
            st.markdown(f"""
            <div style="background:#1a3a1a;border-radius:12px;
                        padding:1.2rem;text-align:center;
                        border:2px solid #2E7D32">
                <h2 style="color:#69F0AE;margin:0">N</h2>
                <h1 style="color:white;margin:0.3rem 0">
                    {rec['npk']['N'][0]}–{rec['npk']['N'][1]}</h1>
                <p style="color:#A5D6A7;margin:0">kg Nitrogen/ha</p>
            </div>""", unsafe_allow_html=True)

        with p_col:
            st.markdown(f"""
            <div style="background:#1a2a3a;border-radius:12px;
                        padding:1.2rem;text-align:center;
                        border:2px solid #1565C0">
                <h2 style="color:#82B1FF;margin:0">P</h2>
                <h1 style="color:white;margin:0.3rem 0">
                    {rec['npk']['P'][0]}–{rec['npk']['P'][1]}</h1>
                <p style="color:#90CAF9;margin:0">kg Phosphorus/ha</p>
            </div>""", unsafe_allow_html=True)

        with k_col:
            st.markdown(f"""
            <div style="background:#3a2a1a;border-radius:12px;
                        padding:1.2rem;text-align:center;
                        border:2px solid #E65100">
                <h2 style="color:#FFAB40;margin:0">K</h2>
                <h1 style="color:white;margin:0.3rem 0">
                    {rec['npk']['K'][0]}–{rec['npk']['K'][1]}</h1>
                <p style="color:#FFCC80;margin:0">kg Potassium/ha</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🛒 Fertilizer Products (kg/hectare)")

        for product, (low, high) in rec['products'].items():
            st.markdown(f"""
            <div style="background:#1E1E1E;border-radius:8px;
                        padding:0.8rem 1.2rem;margin:0.4rem 0;
                        border-left:4px solid #2E7D32;">
                <span style="color:white;font-weight:600">
                    {product}</span>
                <span style="color:#69F0AE;font-size:1.1rem;
                             font-weight:bold;float:right">
                    {low} – {high} kg/ha</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📅 Application Schedule")
        for timing, what in rec['timing'].items():
            st.info(f"**{timing}:** {what}")

        st.markdown("### ⚠️ Important Notes")
        st.warning(f"🌱 **Soil:** {rec['soil_note']}")
        st.info(f"🌧️ **Rainfall:** {rec['rain_note']}")
        st.success(f"📋 **Crop tip:** {rec['crop_notes']}")

        s1, s2 = st.columns(2)
        with s1:
            st.metric("Water Requirement", rec['water_need'])
        with s2:
            st.metric("Days to Harvest", f"{rec['growth_days']} days")
    else:
        st.error("Crop not found in database.")

    with tab3:
        st.subheader("🌦️ Live Weather & Farm Advisory")
    st.markdown("Enter your city to get real-time weather "
                "and auto-fill farm conditions.")

    # ── API Key input ──────────────────────────────────
    st.markdown("#### 🔑 OpenWeatherMap API Key")
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste your free API key here...",
        help="Get free key at openweathermap.org/api"
    )

    if not api_key:
        st.info("👆 Enter your API key above to get started. "
                "Free key available at openweathermap.org")
        st.markdown("""
        **How to get free API key:**
        1. Go to [openweathermap.org/api](https://openweathermap.org/api)
        2. Click Sign Up (free)
        3. Go to API Keys tab
        4. Copy and paste key above
        """)
    else:
        # ── City search ───────────────────────────────
        st.markdown("#### 📍 Your Farm Location")
        col_city, col_btn = st.columns([3, 1])

        with col_city:
            city = st.text_input(
                "City Name",
                placeholder="e.g. Chennai, Mumbai, Delhi..."
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            fetch_clicked = st.button("🔍 Get Weather")

        if fetch_clicked and city:
            with st.spinner(f"Fetching weather for {city}..."):
                weather = get_current_weather(city, api_key)

            if weather['success']:
                # ── Weather display ────────────────────
                st.success(f"✅ Weather data loaded for "
                           f"**{weather['city']}, "
                           f"{weather['country']}**")

                w1, w2, w3, w4 = st.columns(4)
                w1.metric("🌡️ Temperature",
                          f"{weather['temperature']}°C")
                w2.metric("💧 Humidity",
                          f"{weather['humidity']}%")
                w3.metric("🌤️ Condition",
                          weather['condition'])
                w4.metric("🌧️ Est. Rainfall",
                          f"{weather['est_rainfall']} mm")

                st.markdown(f"*Current condition: "
                            f"{weather['description']}*")

                # ── Auto-fill suggestion ───────────────
                st.markdown("---")
                st.markdown("### 🔄 Auto-fill Suggestions")
                st.info(
                    f"Based on live weather in **{weather['city']}**, "
                    f"here are suggested values for your prediction:\n\n"
                    f"- **Temperature:** {weather['temperature']}°C\n"
                    f"- **Rainfall:** {weather['est_rainfall']} mm\n"
                    f"- **Weather Condition:** {weather['condition']}\n\n"
                    f"👈 Go to **Yield Predictor** tab and set "
                    f"these values for accurate prediction!"
                )

                # ── Farming advice ─────────────────────
                st.markdown("---")
                st.markdown("### 🌾 Today's Farming Advisory")
                advice_list = get_weather_farming_advice(weather)

                for advice in advice_list:
                    if advice['type'] == 'warning':
                        st.warning(advice['message'])
                    elif advice['type'] == 'success':
                        st.success(advice['message'])
                    elif advice['type'] == 'info':
                        st.info(advice['message'])

                # ── Quick predict with weather ─────────
                st.markdown("---")
                st.markdown("### ⚡ Quick Predict with Live Weather")
                st.markdown("Using your selected crop and soil "
                            "with live weather data:")

                quick_col1, quick_col2 = st.columns(2)
                with quick_col1:
                    st.markdown(f"**Crop:** {crop}")
                    st.markdown(f"**Region:** {region}")
                    st.markdown(f"**Soil:** {soil_type}")

                with quick_col2:
                    st.markdown(
                        f"**Temperature:** {weather['temperature']}°C")
                    st.markdown(
                        f"**Rainfall:** {weather['est_rainfall']} mm")
                    st.markdown(
                        f"**Condition:** {weather['condition']}")

                if st.button("⚡ Predict with Live Weather Data"):
                    rtr = (weather['est_rainfall'] /
                           (weather['temperature'] + 1))
                    fs  = int(fertilizer) + int(irrigation)
                    fic = int(fertilizer) * int(irrigation)
                    rc  = (0 if weather['est_rainfall'] <= 400
                           else 1 if weather['est_rainfall'] <= 700
                           else 2)
                    tc  = (0 if weather['temperature'] <= 20
                           else 1 if weather['temperature'] <= 30
                           else 2)

                    weather_input = {
                        'Region'             : encode_input(region, 'Region'),
                        'Soil_Type'          : encode_input(soil_type, 'Soil_Type'),
                        'Crop'               : encode_input(crop, 'Crop'),
                        'Rainfall_mm'        : weather['est_rainfall'],
                        'Temperature_Celsius': weather['temperature'],
                        'Fertilizer_Used'    : int(fertilizer),
                        'Irrigation_Used'    : int(irrigation),
                        'Weather_Condition'  : encode_input(
                            weather['condition'], 'Weather_Condition'),
                        'Days_to_Harvest'    : days,
                        'rainfall_temp_ratio': rtr,
                        'farming_score'      : fs,
                        'fert_irrig_combined': fic,
                        'rainfall_category'  : rc,
                        'temp_category'      : tc,
                    }

                    w_input_df = pd.DataFrame(
                        [weather_input])[feature_names]
                    w_yield    = max(
                        0.1, round(model.predict(w_input_df)[0], 3))

                    st.markdown(f"""
                    <div class="yield-result">
                        <p style="color:#C8E6C9;margin:0">
                            Live Weather Prediction</p>
                        <div class="yield-number">{w_yield:.2f}</div>
                        <p style="margin:0">tons per hectare</p>
                        <p style="color:#A5D6A7;font-size:0.9rem;
                                  margin-top:0.5rem">
                            Based on real weather in
                            {weather['city']}</p>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.error(f"❌ {weather['error']}")  

with tab4:
    st.subheader("🌾 Smart Crop Recommender")
    st.markdown("Find the best crop for your farm "
                "based on your conditions.")

    st.info("ℹ️ Using your farm inputs from the left panel. "
            "Adjust Region, Soil, Rainfall, Temperature "
            "and Irrigation on the main form to update "
            "recommendations!")

    # ── Get recommendations ───────────────────────────
    recs = recommend_crops(
        rainfall    = rainfall,
        temperature = temperature,
        soil_type   = soil_type,
        irrigation  = irrigation
    )

    # ── Top recommendation banner ─────────────────────
    top = recs[0]
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,
                #1B5E20,#2E7D32);
                border-radius:16px;padding:1.5rem;
                text-align:center;margin-bottom:1rem">
        <p style="color:#A5D6A7;margin:0;font-size:1rem">
            Best crop for your farm</p>
        <h1 style="color:white;margin:0.3rem 0;
                   font-size:2.5rem">
            {top['emoji']} {top['crop']}</h1>
        <h2 style="color:#69F0AE;margin:0">
            {top['score']}/100 — {top['label']}!</h2>
        <p style="color:#C8E6C9;margin:0.5rem 0 0 0">
            {top['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── All crops ranked ──────────────────────────────
    st.markdown("### 📊 All Crops Ranked")

    for i, rec in enumerate(recs):
        rank_emoji = ["🥇","🥈","🥉",
                      "4️⃣","5️⃣","6️⃣"][i]

        # Progress bar color based on score
        bar_color = rec['color']

        st.markdown(f"""
        <div style="background:#1E1E1E;
                    border-radius:10px;
                    padding:1rem 1.2rem;
                    margin:0.5rem 0;
                    border-left:5px solid {bar_color}">
            <div style="display:flex;
                        justify-content:space-between;
                        align-items:center">
                <span style="font-size:1.1rem;
                             color:white;
                             font-weight:600">
                    {rank_emoji} {rec['emoji']}
                    {rec['crop']}</span>
                <span style="font-size:1.2rem;
                             color:{bar_color};
                             font-weight:bold">
                    {rec['score']}/100
                    {rec['emoji_dot']} {rec['label']}
                </span>
            </div>
            <div style="background:#333;
                        border-radius:4px;
                        height:8px;margin-top:8px">
                <div style="background:{bar_color};
                            width:{rec['score']}%;
                            height:8px;
                            border-radius:4px">
                </div>
            </div>
            <div style="display:flex;gap:1rem;
                        margin-top:0.5rem;
                        font-size:0.85rem;
                        color:#888">
                <span>🌧️ Rain: {rec['rain_score']}/100</span>
                <span>🌡️ Temp: {rec['temp_score']}/100</span>
                <span>🌱 Soil: {rec['soil_score']}/100</span>
                <span>💧 Irrigation:
                    {rec['irrigation']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Crops to avoid ────────────────────────────────
    poor_crops = [r for r in recs if r['score'] < 40]
    if poor_crops:
        st.markdown("### ⚠️ Crops to Avoid")
        for rec in poor_crops:
            reasons = []
            if rec['rain_score'] < 40:
                reasons.append("rainfall mismatch")
            if rec['temp_score'] < 40:
                reasons.append("temperature mismatch")
            if not rec['soil_match']:
                reasons.append("soil not suitable")
            reason_str = ", ".join(reasons) if reasons \
                         else "conditions not ideal"
            st.error(f"❌ **{rec['crop']}** — "
                    f"Score: {rec['score']}/100 — "
                    f"Reason: {reason_str}")

    # ── Factor breakdown ──────────────────────────────
    st.markdown("### 📋 Your Farm Conditions Summary")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("🌧️ Rainfall",    f"{rainfall} mm")
    f2.metric("🌡️ Temperature", f"{temperature}°C")
    f3.metric("🌱 Soil Type",   soil_type)
    f4.metric("💧 Irrigation",
              "✅ Yes" if irrigation else "❌ No")
      
# ── Footer ────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ using Random Forest & Streamlit "
    "| Trained on 100,000 farm records | R²: 91.15%"
    "</small></center>",
    unsafe_allow_html=True
)
