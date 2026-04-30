import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

st.set_page_config(
    page_title="🌾 Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Auto setup ─────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup
setup.setup()

# ── CSS ────────────────────────────────────────────────
st.markdown("""
<style>
.main-header{font-size:2.5rem;font-weight:bold;color:#2E7D32;text-align:center}
.sub-header{font-size:1.1rem;color:#888;text-align:center;margin-bottom:2rem}
.metric-card{background:linear-gradient(135deg,#E8F5E9,#C8E6C9);border-radius:12px;padding:1.2rem;text-align:center;border-left:5px solid #2E7D32}
.metric-card h3{color:#1B5E20;margin:0;font-size:1rem}
.metric-card h2{color:#1B5E20;margin:0.3rem 0 0 0}
.yield-result{background:linear-gradient(135deg,#1B5E20,#2E7D32);color:white;border-radius:16px;padding:2rem;text-align:center;margin:1rem 0}
.yield-number{font-size:3.5rem;font-weight:bold;color:#A5D6A7}
.insight-box{border-radius:10px;padding:1rem 1.2rem;margin:0.5rem 0;font-size:1.05rem;font-weight:600}
.insight-excellent{background:#1B5E20;color:#A5D6A7}
.insight-good{background:#2E7D32;color:#C8E6C9}
.insight-average{background:#E65100;color:#FFE0B2}
.insight-low{background:#B71C1C;color:#FFCDD2}
.confidence-box{background:#263238;border-radius:10px;padding:0.8rem 1.2rem;text-align:center;color:#90A4AE;font-size:0.95rem;margin-top:0.5rem}
.stButton>button{background-color:#2E7D32;color:white;border-radius:8px;padding:0.6rem 2rem;font-size:1.1rem;border:none;width:100%}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load('models/random_forest.pkl')
    encoders      = joblib.load('models/encoders.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, encoders, feature_names

model, encoders, feature_names = load_model()

CROP_AVG_YIELD = {
    'Barley':4.63,'Cotton':4.67,'Maize':4.65,
    'Rice':4.66,'Soybean':4.66,'Wheat':4.65
}

API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

def encode_input(value, column):
    le = encoders[column]
    return le.transform([value])[0] if value in le.classes_ else 0

def get_insight(yield_val):
    if yield_val >= 7.0:
        return "🌟 Exceptional yield expected!", "insight-excellent"
    elif yield_val >= 5.0:
        return "✅ Good yield expected!", "insight-good"
    elif yield_val >= 3.0:
        return "⚠️ Average yield", "insight-average"
    else:
        return "❌ Low yield - review conditions", "insight-low"

# ── Header ─────────────────────────────────────────────
st.markdown('<div class="main-header">🌾 Crop Yield Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered yield prediction for smarter farming</div>', unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
for col,title,value in zip([c1,c2,c3,c4],
    ["🎯 R² Score","📏 Avg Error","🌲 Model","📊 Trained On"],
    ["91.15%","±0.40 t/ha","Random Forest","100K Rows"]):
    col.markdown(f'<div class="metric-card"><h3>{title}</h3><h2>{value}</h2></div>',
                 unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 How to use")
    st.info("1. Select farm details\n2. Adjust sliders\n3. Toggle fertilizer & irrigation\n4. Click Predict!")
    st.markdown("## 🌡️ Optimal Ranges")
    st.markdown("| Factor | Optimal |\n|--------|------|\n| Rainfall | 400-800mm |\n| Temperature | 20-30°C |")
    st.markdown("## 📌 Model Info")
    st.success("Random Forest\n100 trees | Depth 15\nR²: 91.15%")

# ── Inputs ─────────────────────────────────────────────
st.subheader("🌱 Enter Farm Details")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**📍 Farm Information**")
    region    = st.selectbox("Region",    list(encoders['Region'].classes_))
    soil_type = st.selectbox("Soil Type", list(encoders['Soil_Type'].classes_))
    crop      = st.selectbox("Crop",      list(encoders['Crop'].classes_))
    weather   = st.selectbox("Weather Condition", list(encoders['Weather_Condition'].classes_))

with col_right:
    st.markdown("**🌡️ Environmental Conditions**")
    rainfall    = st.slider("Rainfall (mm)",    100.0, 1000.0, 500.0, 10.0)
    temperature = st.slider("Temperature (°C)", 10.0,  45.0,   25.0,  0.5)
    days        = st.slider("Days to Harvest",  30,    180,    90,    5)
    st.markdown("**🚜 Farming Practices**")
    fc, ic = st.columns(2)
    with fc:
        fertilizer = st.toggle("Fertilizer Used", value=True)
    with ic:
        irrigation = st.toggle("Irrigation Used", value=True)

if rainfall < 200:
    st.warning("⚠️ Very low rainfall — predictions may be less reliable.")
if temperature > 40:
    st.warning("⚠️ Very high temperature — most crops struggle above 40°C.")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Yield Predictor",
    "🌿 Fertilizer Advisor",
    "🌦️ Live Weather",
    "🌾 Crop Recommender"
])

# ══════════════════════════════════════════════════════
# TAB 1 — YIELD PREDICTOR
# ══════════════════════════════════════════════════════
with tab1:
    predict_clicked = st.button("🔍 Predict Crop Yield")

    if predict_clicked:
        rtr = rainfall / (temperature + 1)
        fs  = int(fertilizer) + int(irrigation)
        fic = int(fertilizer) * int(irrigation)
        rc  = 0 if rainfall <= 400 else (1 if rainfall <= 700 else 2)
        tc  = 0 if temperature <= 20 else (1 if temperature <= 30 else 2)

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
        low_est    = max(0.0, round(yield_pred - 0.40, 2))
        high_est   = round(yield_pred + 0.40, 2)
        insight, css_class = get_insight(yield_pred)

        st.subheader("📊 Prediction Results")
        res1, res2 = st.columns(2)

        with res1:
            st.markdown(f"""
            <div class="yield-result">
                <p style="color:#C8E6C9;margin:0">Predicted Yield</p>
                <div class="yield-number">{yield_pred:.2f}</div>
                <p style="margin:0">tons per hectare</p>
            </div>
            <div class="confidence-box">
                📊 Range: <strong>{low_est} – {high_est} t/ha</strong> | MAE: ±0.40
            </div>
            <div class="insight-box {css_class}" style="margin-top:0.8rem">{insight}</div>
            """, unsafe_allow_html=True)

        with res2:
            st.markdown("**📋 Input Summary**")
            for k,v in {"Crop":crop,"Region":region,"Soil":soil_type,
                        "Weather":weather,"Rainfall":f"{rainfall}mm",
                        "Temperature":f"{temperature}°C",
                        "Fertilizer":"✅ Yes" if fertilizer else "❌ No",
                        "Irrigation":"✅ Yes" if irrigation else "❌ No"}.items():
                st.markdown(f"- **{k}:** {v}")

        st.markdown("---")
        st.subheader("📈 How Does Your Farm Compare?")
        avg_yield = CROP_AVG_YIELD.get(crop, 4.65)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 2.5))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        categories = ['Your Farm', f'{crop} Avg', 'Max']
        values     = [yield_pred, avg_yield, 9.53]
        colors     = ['#2E7D32','#1565C0','#E65100']
        bars = ax.barh(categories, values, color=colors, height=0.5)
        for bar, val in zip(bars, values):
            ax.text(val+0.05, bar.get_y()+bar.get_height()/2,
                    f'{val:.2f}', va='center', color='white', fontsize=10, fontweight='bold')
        ax.set_xlabel('Yield (t/ha)', color='#90A4AE')
        ax.tick_params(colors='white')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.set_xlim(0, 11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.subheader("💡 Improvement Tips")
        t1,t2,t3 = st.columns(3)
        with t1:
            if not fertilizer:
                st.warning(f"🌿 **Add Fertilizer**\n\n+37% yield\n\nEst: **{min(9.5,yield_pred*1.37):.2f} t/ha**")
            else:
                st.success("🌿 **Fertilizer** ✅")
        with t2:
            if not irrigation:
                st.warning(f"💧 **Add Irrigation**\n\n+25% yield\n\nEst: **{min(9.5,yield_pred*1.25):.2f} t/ha**")
            else:
                st.success("💧 **Irrigation** ✅")
        with t3:
            if rainfall < 400:
                st.warning("🌧️ **Low Rainfall**\n\nConsider water-efficient crops")
            elif rainfall > 800:
                st.info("🌧️ **High Rainfall**\n\nEnsure good drainage")
            else:
                st.success("🌧️ **Rainfall** ✅")

# ══════════════════════════════════════════════════════
# TAB 2 — FERTILIZER ADVISOR
# ══════════════════════════════════════════════════════
with tab2:
    sys.path.append('src')
    from fertilizer_advisor import get_fertilizer_recommendation

    st.subheader("🌿 Smart Fertilizer Advisor")
    rec = get_fertilizer_recommendation(crop, soil_type, rainfall)

    if rec:
        st.markdown("### 📊 NPK Requirements (kg/hectare)")
        n_col,p_col,k_col = st.columns(3)
        with n_col:
            st.markdown(f'<div style="background:#1a3a1a;border-radius:12px;padding:1.2rem;text-align:center;border:2px solid #2E7D32"><h2 style="color:#69F0AE;margin:0">N</h2><h1 style="color:white;margin:0.3rem 0">{rec["npk"]["N"][0]}–{rec["npk"]["N"][1]}</h1><p style="color:#A5D6A7;margin:0">kg Nitrogen/ha</p></div>', unsafe_allow_html=True)
        with p_col:
            st.markdown(f'<div style="background:#1a2a3a;border-radius:12px;padding:1.2rem;text-align:center;border:2px solid #1565C0"><h2 style="color:#82B1FF;margin:0">P</h2><h1 style="color:white;margin:0.3rem 0">{rec["npk"]["P"][0]}–{rec["npk"]["P"][1]}</h1><p style="color:#90CAF9;margin:0">kg Phosphorus/ha</p></div>', unsafe_allow_html=True)
        with k_col:
            st.markdown(f'<div style="background:#3a2a1a;border-radius:12px;padding:1.2rem;text-align:center;border:2px solid #E65100"><h2 style="color:#FFAB40;margin:0">K</h2><h1 style="color:white;margin:0.3rem 0">{rec["npk"]["K"][0]}–{rec["npk"]["K"][1]}</h1><p style="color:#FFCC80;margin:0">kg Potassium/ha</p></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🛒 Products (kg/hectare)")
        for product,(low,high) in rec['products'].items():
            st.markdown(f'<div style="background:#1E1E1E;border-radius:8px;padding:0.8rem 1.2rem;margin:0.4rem 0;border-left:4px solid #2E7D32"><span style="color:white;font-weight:600">{product}</span><span style="color:#69F0AE;font-size:1.1rem;font-weight:bold;float:right">{low}–{high} kg/ha</span></div>', unsafe_allow_html=True)

        st.markdown("### 📅 Application Schedule")
        for timing,what in rec['timing'].items():
            st.info(f"**{timing}:** {what}")

        st.markdown("### ⚠️ Notes")
        st.warning(f"🌱 **Soil:** {rec['soil_note']}")
        st.info(f"🌧️ **Rainfall:** {rec['rain_note']}")
        st.success(f"📋 **Crop tip:** {rec['crop_notes']}")

        s1,s2 = st.columns(2)
        s1.metric("Water Requirement", rec['water_need'])
        s2.metric("Days to Harvest",   f"{rec['growth_days']} days")

# ══════════════════════════════════════════════════════
# TAB 3 — LIVE WEATHER
# ══════════════════════════════════════════════════════
with tab3:
    from weather_api import get_current_weather, get_weather_farming_advice

    st.subheader("🌦️ Live Weather & Farm Advisory")
    api_key = API_KEY

    if not api_key:
        st.warning("Weather API not configured.")
    else:
        city = st.text_input("📍 City Name", placeholder="e.g. Chennai, Mumbai...")
        if st.button("🔍 Get Weather") and city:
            with st.spinner(f"Fetching weather for {city}..."):
                weather = get_current_weather(city, api_key)

            if weather['success']:
                st.success(f"✅ {weather['city']}, {weather['country']}")
                w1,w2,w3,w4 = st.columns(4)
                w1.metric("🌡️ Temperature", f"{weather['temperature']}°C")
                w2.metric("💧 Humidity",    f"{weather['humidity']}%")
                w3.metric("🌤️ Condition",   weather['condition'])
                w4.metric("🌧️ Est. Rainfall", f"{weather['est_rainfall']} mm")

                st.markdown("---")
                st.markdown("### 🌾 Today's Farming Advisory")
                for advice in get_weather_farming_advice(weather):
                    if advice['type'] == 'warning':
                        st.warning(advice['message'])
                    elif advice['type'] == 'success':
                        st.success(advice['message'])
                    else:
                        st.info(advice['message'])

                st.markdown("---")
                st.info(f"💡 **Auto-fill suggestion:** Set Temperature={weather['temperature']}°C, Rainfall={weather['est_rainfall']}mm in Yield Predictor tab!")
            else:
                st.error(f"❌ {weather['error']}")

# ══════════════════════════════════════════════════════
# TAB 4 — CROP RECOMMENDER
# ══════════════════════════════════════════════════════
with tab4:
    from crop_recommender import recommend_crops

    st.subheader("🌾 Smart Crop Recommender")
    st.info("ℹ️ Adjust farm inputs on the left panel to update recommendations!")

    recs = recommend_crops(rainfall=rainfall, temperature=temperature,
                           soil_type=soil_type, irrigation=irrigation)
    top  = recs[0]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1B5E20,#2E7D32);border-radius:16px;
                padding:1.5rem;text-align:center;margin-bottom:1rem">
        <p style="color:#A5D6A7;margin:0">Best crop for your farm</p>
        <h1 style="color:white;margin:0.3rem 0;font-size:2.5rem">{top['emoji']} {top['crop']}</h1>
        <h2 style="color:#69F0AE;margin:0">{top['score']}/100 — {top['label']}!</h2>
        <p style="color:#C8E6C9;margin:0.5rem 0 0 0">{top['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 All Crops Ranked")
    for i,rec in enumerate(recs):
        rank = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣"][i]
        st.markdown(f"""
        <div style="background:#1E1E1E;border-radius:10px;padding:1rem 1.2rem;
                    margin:0.5rem 0;border-left:5px solid {rec['color']}">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:white;font-weight:600">{rank} {rec['emoji']} {rec['crop']}</span>
                <span style="color:{rec['color']};font-weight:bold">{rec['score']}/100 {rec['emoji_dot']} {rec['label']}</span>
            </div>
            <div style="background:#333;border-radius:4px;height:8px;margin-top:8px">
                <div style="background:{rec['color']};width:{rec['score']}%;height:8px;border-radius:4px"></div>
            </div>
            <div style="color:#888;font-size:0.85rem;margin-top:0.5rem">
                🌧️ Rain:{rec['rain_score']}/100 | 🌡️ Temp:{rec['temp_score']}/100 |
                🌱 Soil:{rec['soil_score']}/100 | 💧 {rec['irrigation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<center><small>Built with ❤️ using Random Forest & Streamlit | R²: 91.15%</small></center>",
            unsafe_allow_html=True)
