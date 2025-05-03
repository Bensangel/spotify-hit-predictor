import streamlit as st
import pandas as pd
import joblib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# 🎵 Spotify API Auth
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=st.secrets["SPOTIFY_CLIENT_ID"],
    client_secret=st.secrets["SPOTIFY_CLIENT_SECRET"]
))

# 🔮 Load model
model = joblib.load('spotify_hit_predictor.pkl')

# 🎨 Page config
st.set_page_config(page_title="Spotify Hit Predictor", layout="centered")
st.title("🎵 Spotify Hit Predictor")
st.markdown("Enter a song's features manually **or** paste a Spotify track link!")

# 📊 Feature columns
feature_columns = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
    'duration_ms', 'time_signature'
]

# 🔎 Extract track ID from Spotify link
def extract_track_id(spotify_url):
    if "track/" in spotify_url:
        return spotify_url.split("track/")[1].split("?")[0]
    return None

# 📥 User input
spotify_url = st.text_input("🎧 Paste a Spotify track link (optional):")
input_df = None  # Make sure this is defined at the top

# 🧠 If Spotify link is given
if spotify_url:
    track_id = extract_track_id(spotify_url)
    try:
        features = sp.audio_features([track_id])[0]
        if features:
            input_data = {}
            for col in feature_columns:
                input_data[col] = features[col]
            input_df = pd.DataFrame([input_data])
            st.success("✅ Features loaded from Spotify!")
        else:
            st.error("⚠️ Could not get features for that track.")
    except Exception as e:
        st.error(f"❌ Could not fetch song: {e}")

# 🎚️ If no Spotify link, use manual sliders
if input_df is None:
    st.sidebar.header("🎚️ Or Manually Set Features")
    input_data = {}
    for col in feature_columns:
        if col in ['mode', 'key', 'time_signature']:
            input_data[col] = st.sidebar.number_input(f"{col}", min_value=0, max_value=10, value=1)
        elif col == 'duration_ms':
            input_data[col] = st.sidebar.slider(f"{col}", min_value=100000, max_value=400000, value=210000, step=1000)
        else:
            input_data[col] = st.sidebar.slider(f"{col}", 0.0, 1.0, 0.5)
    input_df = pd.DataFrame([input_data])

# 🎧 Predict
if input_df is not None and st.button("🎧 Predict!"):
    prediction = model.predict(input_df)[0]
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("🔥 This song is likely to be a HIT!")
    else:
        st.warning("🧊 This song might NOT be a hit.")
    
    st.markdown("---")
    st.subheader("🔍 Audio Features Used")
    st.write(input_df)
