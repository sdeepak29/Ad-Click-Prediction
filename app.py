import streamlit as st
import numpy as np
import pickle

# ⚡ set_page_config must be here FIRST
st.set_page_config(page_title="Ad Click Prediction", page_icon="📢")

# Load model
@st.cache_resource
def load_model():
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

if not st.session_state.submitted:
    st.title("📢 Ad Click Prediction Form")

    with st.form("ad_form"):
        daily_time_spent = st.number_input('Daily Time Spent on Site', min_value=0.0, max_value=500.0, value=100.0)
        age = st.number_input('Age', min_value=0, max_value=100, value=30)
        area_income = st.number_input('Area Income', min_value=0.0, max_value=1000000.0, value=50000.0)
        daily_internet_usage = st.number_input('Daily Internet Usage', min_value=0.0, max_value=500.0, value=150.0)
        male = st.selectbox('Gender', options=[('Male', 1), ('Female', 0)], format_func=lambda x: x[0])[1]
        city_code = st.number_input('City Code', min_value=0, value=0)
        country_code = st.number_input('Country Code', min_value=0, value=0)
        month = st.selectbox('Month', options=[str(i).zfill(2) for i in range(1,13)])
        hour = st.selectbox('Hour', options=[str(i).zfill(2) for i in range(0,24)])

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state.submitted = True
            st.session_state.inputs = np.array([[float(daily_time_spent), 
                                     int(age), 
                                     float(area_income), 
                                     float(daily_internet_usage),
                                     int(male), 
                                     int(city_code), 
                                     int(country_code), 
                                     int(month), 
                                     int(hour)]])

            st.experimental_rerun()

else:
    st.title("🔮 Prediction Result")

    input_data = st.session_state.inputs
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ The user is likely to CLICK on the ad!")
    else:
        st.error("❌ The user is NOT likely to click on the ad.")

    st.button("Go back", on_click=lambda: st.session_state.update({'submitted': False}))
