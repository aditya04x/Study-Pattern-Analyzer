#  AI-Based Study Pattern Analyzer

An AIML project that analyzes student study habits using **K-Means Clustering** and provides personalized insights and suggestions.

---

##  Overview

Students often struggle to understand their study patterns, leading to inefficient preparation and poor performance.
This project applies **unsupervised machine learning** to identify hidden patterns in study behavior and classify students into meaningful categories.

---

##  Features

*  Data-driven analysis of study habits
*  K-Means clustering (unsupervised learning)
*  Classification into:

  * Focused Learner
  * Balanced Learner
  * Distracted Learner
*  Personalized suggestions based on behavior
*  Visualization of study patterns
*  User input system for real-time prediction
*  Modular structure (model + app separation)

---

##  Technologies Used

* Python
* pandas
* scikit-learn
* matplotlib

---

##  Project Structure

```
study-analyzer/
│── data/
│   └── study_data.csv
│── model/
│   ├── model.py
│   └── visualize.py
│── app/
│   └── main.py
|── report/
│   └── Project_report_aiml.pdf
│── infographic.png
│── README.md
```

---

##  Dataset

The dataset includes:

* Hours Studied
* Subjects Studied
* Breaks Taken
* Focus Level (1–10)
* Sleep Hours

The dataset was synthetically generated to simulate realistic student study behavior.

---

##  How It Works

1. Load dataset
2. Normalize data using `StandardScaler`
3. Apply K-Means clustering (K = 3)
4. Identify and label clusters
5. Accept user input
6. Predict category and provide suggestions

---

##  How to Run

### 1. Clone the repository

```
git clone https://github.com/aditya04x/Study-Pattern-Analyzer.git
cd study-analyzer
```

---

### 2. Install dependencies

```
pip install pandas scikit-learn matplotlib
```

---

### 3. Run the application

```
python app/main.py
```

---

### 4. Enter your study data

Example:

```
Hours Studied: 3
Subjects Studied: 2
Breaks Taken: 5
Focus Level: 6
Sleep Hours: 7
```

---

### 5. Output

```
Category: Balanced Learner
Suggestion: Increase consistency and focus slightly.
```

---

## 📈 Visualization

To view clustering graphs:

```
python model/visualize.py
```

Includes:

* Cluster visualization
* Elbow method graph

---

##  Project Infographic

![Infographic](infographic.png)<img width="1920" height="1080" alt="infographics" src="https://github.com/user-attachments/assets/52348c1b-da85-4020-9e34-ee0ef9df37ce" />


---

##  Machine Learning Concept Used

* **K-Means Clustering (Unsupervised Learning)**
  Used to group similar study patterns without labeled data.

---

##  Note

Ensure the dataset is placed correctly inside the `data/` folder before running the project.

---

##  Future Improvements

* Use real-world student data
* Add web interface (Streamlit)
* Include more features (stress level, exam scores)
* Improve clustering accuracy

---

##  Author

Aditya Singh
