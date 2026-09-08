import streamlit as st
import pandas as pd
import os
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #fff5f5, #f5f9ff, #f4f0ff);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    color: #d90429;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #555555;
    font-size: 1.15rem;
    margin-bottom: 30px;
}

/* Section titles */
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #2b2d42;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Predict button */
.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 18px;
    border: none;
    background: linear-gradient(90deg, #e63946, #d90429, #7b2cbf);
    color: white;
    font-size: 1.25rem;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(217, 4, 41, 0.3);
}

.stButton > button:hover {
    transform: translateY(-3px);
}

/* Footer */
.footer {
    text-align: center;
    color: #777;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "KNN_heart.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler.pkl")
)

expected_columns = joblib.load(
    os.path.join(BASE_DIR, "columns.pkl")
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">❤️ Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning powered heart disease risk assessment</div>',
    unsafe_allow_html=True
)


st.divider()


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header("👤 Patient Information")

st.write(
    "Enter the patient's health information below to generate "
    "a prediction using the K-Nearest Neighbors (KNN) model."
)


# ============================================================
# INPUTS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader("🧑 Basic Information")

    age = st.slider(
        "🎂 Age",
        18,
        100,
        40
    )

    sex = st.selectbox(
        "⚧ Sex",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "💓 Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"],
        help="""
        ATA = Atypical Angina
        NAP = Non-Anginal Pain
        TA = Typical Angina
        ASY = Asymptomatic
        """
    )

    resting_bp = st.number_input(
        "🩸 Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=200,
        value=120
    )

    cholesterol = st.number_input(
        "🧪 Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=200
    )


with col2:

    st.subheader("📊 Medical Information")

    fasting_bs = st.selectbox(
        "🍬 Fasting Blood Sugar > 120 mg/dL",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    resting_ecg = st.selectbox(
        "📈 Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "❤️ Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "🏃 Exercise-Induced Angina",
        ["Y", "N"],
        format_func=lambda x: "Yes" if x == "Y" else "No"
    )

    oldpeak = st.slider(
        "📉 Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0,
        step=0.1
    )

    st_slope = st.selectbox(
        "📊 ST Slope",
        ["Up", "Flat", "Down"]
    )


# ============================================================
# INFORMATION
# ============================================================

st.divider()

st.info(
    "💡 The model uses the patient's clinical measurements "
    "and medical information to predict heart disease risk."
)

st.warning(
    "⚠️ This application is for educational purposes only "
    "and should not be considered a medical diagnosis."
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "🔍 Predict Heart Disease Risk"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # Create raw input dictionary

    raw_input = {

        "Age": age,

        "RestingBP": resting_bp,

        "Cholesterol": cholesterol,

        "FastingBS": fasting_bs,

        "MaxHR": max_hr,

        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,

        "ChestPainType_" + chest_pain: 1,

        "RestingECG_" + resting_ecg: 1,

        "ExerciseAngina_" + exercise_angina: 1,

        "ST_Slope_" + st_slope: 1
    }


    # Create input dataframe

    input_df = pd.DataFrame([raw_input])


    # Fill missing columns with 0

    for col in expected_columns:

        if col not in input_df.columns:

            input_df[col] = 0


    # Reorder columns

    input_df = input_df[expected_columns]


    # Scale input

    scaled_input = scaler.transform(input_df)


    # Make prediction

    prediction = model.predict(scaled_input)[0]


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header("📋 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ HIGHER RISK DETECTED\n\n"
            "The model predicts a higher likelihood of "
            "heart disease based on the information provided."
        )

        st.warning(
            "Please consult a qualified healthcare professional "
            "for proper medical evaluation."
        )

    else:

        st.success(
            "💚 LOWER RISK DETECTED\n\n"
            "The model predicts a lower likelihood of "
            "heart disease based on the information provided."
        )

        st.info(
            "Continue maintaining a healthy lifestyle and "
            "consult a healthcare professional when appropriate."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    '❤️ Heart Disease Prediction<br><br>'
    'Built with Python • Scikit-learn • Streamlit • KNN<br>'
    'Developed by AK'
    '</div>',
    unsafe_allow_html=True
)

# import streamlit as st
# import pandas as pd
# import os
# import joblib

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Load saved model, scaler, and expected columns
# model = joblib.load(os.path.join(BASE_DIR, "KNN_heart.pkl"))
# scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
# expected_columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

# st.title("Heart Stroke Prediction by AK")
# st.markdown("Provide the following details to check your heart stroke risk:")

# # Collect user input
# age = st.slider("Age", 18, 100, 40)
# sex = st.selectbox("Sex", ["M", "F"])
# chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
# resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
# cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
# fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
# resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
# max_hr = st.slider("Max Heart Rate", 60, 220, 150)
# exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
# oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
# st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# # When Predict is clicked
# if st.button("Predict"):

#     # Create a raw input dictionary
#     raw_input = {
#         'Age': age,
#         'RestingBP': resting_bp,
#         'Cholesterol': cholesterol,
#         'FastingBS': fasting_bs,
#         'MaxHR': max_hr,
#         'Oldpeak': oldpeak,
#         'Sex_' + sex: 1,
#         'ChestPainType_' + chest_pain: 1,
#         'RestingECG_' + resting_ecg: 1,
#         'ExerciseAngina_' + exercise_angina: 1,
#         'ST_Slope_' + st_slope: 1
#     }

#     # Create input dataframe
#     input_df = pd.DataFrame([raw_input])

#     # Fill in missing columns with 0s
#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     # Reorder columns
#     input_df = input_df[expected_columns]

#     # Scale the input
#     scaled_input = scaler.transform(input_df)

#     # Make prediction
#     prediction = model.predict(scaled_input)[0]

#     # Show result
#     if prediction == 1:
#         st.error("⚠️ High Risk of Heart Disease")
#     else:
#         st.success("✅ Low Risk of Heart Disease")