import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# 🔧 Background and font styling
def set_bg_and_font():
    st.markdown("""
        <style>
        .stApp {
            background-image: url("");
            background-size: cover;
            background-attachment: fixed;
        }
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }
        .centered {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background-color: rgba(255,255,255,0.88);
            padding: 2rem;
            border-radius: 12px;
            max-width: 500px;
            margin: auto;
        }
        </style>
    """, unsafe_allow_html=True)

set_bg_and_font()

# 🔍 Load trained model
model = pickle.load(open('artifacts/model.pkl', 'rb'))

# 🧠 BMI category
def get_bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"
    elif 18.5 <= bmi < 25:
        return "normal"
    else:
        return "overweight"

# 🎯 Athlete benchmark table
athlete_benchmarks = {
    "underweight": 180.0,
    "normal": 250.0,
    "overweight": 320.0
}

# 💪 Fitness Suggestions
def get_fitness_suggestion(bmi_cat, calories):
    if bmi_cat == "underweight":
        if calories < 200:
            return "🏋️‍♂️ Focus on light strength training and calorie surplus with high protein."
        else:
            return "✅ Good! Maintain strength training and ensure proper nutrition."
    elif bmi_cat == "normal":
        if calories < 220:
            return "🚴 Increase workout intensity: Try HIIT or cardio + strength combo."
        else:
            return "💪 You're in great shape! Maintain this workout with core and flexibility."
    else:  # overweight
        if calories < 250:
            return "🔥 Aim for longer cardio sessions (45–60 mins) and track food intake."
        else:
            return "✅ Great burn! Add strength training to boost metabolism and reduce fat."

# 🧾 UI Title
st.markdown("<h1 style='text-align: center;'>🔥 Calorie Burn Predictor</h1>", unsafe_allow_html=True)

# 📥 Input form
with st.form("predict_form"):
    st.markdown('<div class="centered">', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=200, value=70)
    duration = st.number_input("Workout Duration (mins)", min_value=1, max_value=300, value=30)
    heart_rate = st.number_input("Heart Rate", min_value=50, max_value=200, value=100)
    body_temp = st.number_input("Body Temp (°C)", min_value=30.0, max_value=45.0, value=37.0)

    # ✅ BMI Calculation
    bmi = weight / ((height / 100) ** 2)
    st.markdown(f"**Your BMI:** `{bmi:.2f}`")

    submitted = st.form_submit_button("Predict 🔥")
    st.markdown('</div>', unsafe_allow_html=True)

# 🔮 Prediction Logic
if submitted:
    gender_val = 1 if gender == "Male" else 0

    # ✅ Pass all 8 features as required by model
    input_df = pd.DataFrame([[gender_val, age, height, weight, duration, heart_rate, body_temp, bmi]],
        columns=["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "BMI"])

    prediction = model.predict(input_df)[0]
    st.success(f"🔥 Estimated Calories Burned: **{prediction:.2f} kcal**")

    # 🧠 Athlete benchmark comparison
    bmi_cat = get_bmi_category(bmi)
    benchmark = athlete_benchmarks[bmi_cat]
    diff = prediction - benchmark
    percent = (diff / benchmark) * 100

    if diff > 0:
        st.markdown(f"✅ You're burning **{percent:.1f}% more** calories than the average {bmi_cat} athlete 💪")
    else:
        st.markdown(f"📉 You're burning **{abs(percent):.1f}% less** than the average {bmi_cat} athlete. Keep pushing! 🚀")

    # 📊 Chart comparison
    st.markdown("### 🔍 Visual Comparison")
    fig = go.Figure(data=[
        go.Bar(name='You', x=['Calories Burned'], y=[prediction], marker_color='limegreen'),
        go.Bar(name='Athlete Avg', x=['Calories Burned'], y=[benchmark], marker_color='orangered')
    ])
    fig.update_layout(barmode='group', title_text='Your Burn vs Athlete Benchmark', yaxis_title='Calories (kcal)')
    st.plotly_chart(fig, use_container_width=True)

    # 💡 Personalized fitness suggestion
    suggestion = get_fitness_suggestion(bmi_cat, prediction)
    st.markdown("### 💡 Fitness Suggestion")
    st.info(suggestion)
