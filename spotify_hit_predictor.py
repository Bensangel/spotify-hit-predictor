import streamlit as st
import pandas as pd
import joblib

# Load pre-trained model
model = joblib.load('spotify_hit_predictor.pkl')


# Page config
st.set_page_config(page_title="Spotify Hit Predictor", layout="centered")
st.title("🎵 Spotify Hit Predictor")
st.markdown("Enter a song's audio features to see if it's a HIT or NOT.")

# Load data
def load_data():
    files = [
        'dataset-of-60s.csv', 'dataset-of-70s.csv', 'dataset-of-80s.csv',
        'dataset-of-90s.csv', 'dataset-of-00s.csv', 'dataset-of-10s.csv'
    ]
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop(['track', 'artist', 'uri', 'chorus_hit', 'sections'], axis=1, errors='ignore')
    df = df.dropna()
    return df

# Train model
def train_model(df):
    X = df.drop(['target', 'decade'], axis=1, errors='ignore')
    y = df['target']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, X.columns

# App inputs
def user_input_features(columns):
    st.sidebar.header("🎚️ Audio Features")
    values = {}
    for col in columns:
        if col in ['mode', 'key', 'time_signature']:
            values[col] = st.sidebar.number_input(f"{col}", value=1, step=1)
        elif col == 'duration_ms':
            values[col] = st.sidebar.slider(f"{col}", min_value=100000, max_value=400000, value=210000, step=1000)
        else:
            values[col] = st.sidebar.slider(f"{col}", 0.0, 1.0, 0.5)
    return pd.DataFrame([values])

# Main
try:
    df = load_data()
    model, columns = train_model(df)
    input_df = user_input_features(columns)

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

except Exception as e:
    st.error(f"Something went wrong: {e}")
