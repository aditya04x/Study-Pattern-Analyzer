import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def train_model():
    # Load dataset
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, "data", "study_data.csv")

    data = pd.read_csv(file_path)

    # Features
    X = data[["Hours_Studied", "Subjects_Studied", "Breaks_Taken", "Focus_Level", "Sleep_Hours"]]

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)

    return kmeans, scaler