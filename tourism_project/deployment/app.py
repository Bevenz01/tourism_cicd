"""
Streamlit App - Wellness Tourism Package Purchase Predictor
"""
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

HF_USERNAME  = 'natkhatgopaljee'
DATASET_REPO = f'{HF_USERNAME}/tourism-dataset'
MODEL_REPO   = f'{HF_USERNAME}/tourism-wellness-model'

@st.cache_resource(show_spinner='Loading model from Hugging Face...')
def load_artifacts():
    model_path   = hf_hub_download(repo_id=MODEL_REPO,   filename='best_model.pkl',    repo_type='model')
    feature_path = hf_hub_download(repo_id=MODEL_REPO,   filename='feature_names.pkl', repo_type='model')
    encoder_path = hf_hub_download(repo_id=DATASET_REPO, filename='encoders.pkl',      repo_type='dataset')
    return joblib.load(model_path), joblib.load(feature_path), joblib.load(encoder_path)

model, feature_names, encoders = load_artifacts()

def choices(col):
    return sorted(encoders[col].classes_.tolist())

st.set_page_config(page_title='Wellness Tourism Predictor', layout='wide')
st.title('Wellness Tourism Package — Purchase Predictor')
st.markdown('Fill in the customer details and click **Predict**.')
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader('Customer Details')
    age            = st.number_input('Age', min_value=18, max_value=100, value=35)
    monthly_income = st.number_input('Monthly Income', min_value=0, value=25000, step=500)
    city_tier      = st.selectbox('City Tier', [1, 2, 3])
    occupation     = st.selectbox('Occupation', choices('Occupation'))
    gender         = st.selectbox('Gender', choices('Gender'))
    marital_status = st.selectbox('Marital Status', choices('MaritalStatus'))
    designation    = st.selectbox('Designation', choices('Designation'))
with col2:
    st.subheader('Travel Preferences')
    number_of_trips   = st.number_input('Avg Trips / Year', min_value=0, value=2)
    number_of_persons = st.number_input('Persons Visiting', min_value=1, value=2)
    number_of_children= st.number_input('Children Visiting (age<5)', min_value=0, value=0)
    preferred_star    = st.selectbox('Preferred Hotel Stars', [1,2,3,4,5], index=2)
    passport          = st.selectbox('Has Passport?', ['No (0)','Yes (1)'])
    own_car           = st.selectbox('Owns a Car?',   ['No (0)','Yes (1)'])
with col3:
    st.subheader('Sales Interaction')
    type_of_contact  = st.selectbox('Type of Contact',      choices('TypeofContact'))
    product_pitched  = st.selectbox('Product Pitched',       choices('ProductPitched'))
    duration_pitch   = st.number_input('Pitch Duration (min)', min_value=0, value=10)
    num_followups    = st.number_input('No. of Follow-ups',    min_value=0, value=3)
    pitch_score      = st.slider('Pitch Satisfaction Score', 1, 5, 3)

st.divider()
if st.button('Predict Purchase Likelihood', type='primary', use_container_width=True):
    input_dict = {
        'Age': age, 'TypeofContact': type_of_contact, 'CityTier': city_tier,
        'DurationOfPitch': duration_pitch, 'Occupation': occupation, 'Gender': gender,
        'NumberOfPersonVisiting': number_of_persons, 'NumberOfFollowups': num_followups,
        'ProductPitched': product_pitched, 'PreferredPropertyStar': preferred_star,
        'MaritalStatus': marital_status, 'NumberOfTrips': number_of_trips,
        'Passport': int(passport[0]), 'PitchSatisfactionScore': pitch_score,
        'OwnCar': int(own_car[0]), 'NumberOfChildrenVisiting': number_of_children,
        'Designation': designation, 'MonthlyIncome': monthly_income,
    }
    for col in ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation']:
        try:
            input_dict[col] = int(encoders[col].transform([str(input_dict[col])])[0])
        except:
            input_dict[col] = 0
    input_df    = pd.DataFrame([input_dict])[feature_names]
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.divider()
    if prediction == 1:
        st.success(f'LIKELY TO PURCHASE the Wellness Tourism Package  (Confidence: {probability:.1%})')
    else:
        st.error(f'NOT LIKELY TO PURCHASE the Wellness Tourism Package  (Confidence: {1-probability:.1%})')
    with st.expander('Raw input sent to model'):
        st.dataframe(input_df)
