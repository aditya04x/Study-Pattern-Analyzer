import pandas as pd
import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import train_model

# Train model
kmeans, scaler = train_model()

# Label function
def get_label(cluster):
    if cluster == 0:
        return "Focused Learner"
    elif cluster == 1:
        return "Distracted Learner"
    else:
        return "Balanced Learner"

# Suggestion function
def suggest(category):
    if category == "Distracted Learner":
        return "Reduce breaks and improve focus time."
    elif category == "Balanced Learner":
        return "Increase consistency and focus slightly."
    else:
        return "Great job! Maintain your routine."

# -------- USER INPUT -------- #

print("\nEnter your study details:")

hours = float(input("Hours Studied: "))
subjects = float(input("Subjects Studied: "))
breaks = float(input("Breaks Taken: "))
focus = float(input("Focus Level (1-10): "))
sleep = float(input("Sleep Hours: "))

# Convert to DataFrame (fix warning issue)
user_data = pd.DataFrame([{
    "Hours_Studied": hours,
    "Subjects_Studied": subjects,
    "Breaks_Taken": breaks,
    "Focus_Level": focus,
    "Sleep_Hours": sleep
}])

# Scale + Predict
user_scaled = scaler.transform(user_data)
cluster = kmeans.predict(user_scaled)[0]

category = get_label(cluster)
advice = suggest(category)

print("\n--- RESULT ---")
print("Category:", category)
print("Suggestion:", advice)