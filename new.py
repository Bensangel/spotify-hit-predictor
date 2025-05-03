import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# after training
joblib.dump(model, 'spotify_hit_predictor.pkl')

# Load all your CSVs
files = [
    'dataset-of-60s.csv', 'dataset-of-70s.csv', 'dataset-of-80s.csv',
    'dataset-of-90s.csv', 'dataset-of-00s.csv', 'dataset-of-10s.csv'
]

dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

# Combine into one
df = pd.concat(dfs, ignore_index=True)

# Drop unnecessary columns
df = df.drop(['track', 'artist', 'uri', 'chorus_hit', 'sections'], axis=1, errors='ignore')
df = df.dropna()

# Prepare data
X = df.drop(['target'], axis=1)
y = df['target']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model
import pickle

with open('spotify_hit_predictor.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model saved as spotify_hit_predictor.pkl")
