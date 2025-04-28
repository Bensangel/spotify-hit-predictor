import streamlit as st
import pandas as pd
import pickle

# Load pre-trained model
with open('spotify_hit_predictor.pkl', 'rb') as file:
    model = pickle.load(file)

# Page config
st.set_page_config(page_title="Spotify Hit Predictor", layout="centered")
st.title("🎵 Spotify Hit Predictor")
st.markdown("Enter a song's audio features to see if it's a HIT or NOT.")

# Define the expected feature columns
feature_columns = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
    'duration_ms', 'time_signature'
]

# App inputs
st.sidebar.header("🎚️ Audio Features")
input_data = {}
for col in feature_columns:
    if col in ['mode', 'key', 'time_signature']:
        input_data[col] = st.sidebar.number_input(f"{col}", min_value=0, max_value=10, value=1)
    elif col == 'duration_ms':
        input_data[col] = st.sidebar.slider(f"{col}", min_value=100000, max_value=400000, value=210000, step=1000)
    else:
        input_data[col] = st.sidebar.slider(f"{col}", 0.0, 1.0, 0.5)

input_df = pd.DataFrame([input_data])

# Prediction button
if st.button("🎧 Predict!"):
    prediction = model.predict(input_df)[0]
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("🔥 This song is likely to be a HIT!")
    else:
        st.warning("🧊 This song might NOT be a hit.")

    st.markdown("---")
    st.subheader("🔍 Your Input Features")
    st.write(input_df)
