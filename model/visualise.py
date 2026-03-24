import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Load data (absolute path fix)
base_dir = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(base_dir, "data", "study_data.csv")

data = pd.read_csv(file_path)

X = data[["Hours_Studied", "Subjects_Studied", "Breaks_Taken", "Focus_Level", "Sleep_Hours"]]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
data["Cluster"] = kmeans.fit_predict(X_scaled)

# -------- GRAPH 1: CLUSTER VISUALIZATION -------- #
plt.figure()

plt.scatter(data["Hours_Studied"], data["Focus_Level"], c=data["Cluster"])
plt.xlabel("Hours Studied")
plt.ylabel("Focus Level")
plt.title("Study Pattern Clustering")

plt.show()

# -------- GRAPH 2: ELBOW METHOD -------- #

wcss = []

for i in range(1, 8):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()

plt.plot(range(1, 8), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal K")

plt.show()