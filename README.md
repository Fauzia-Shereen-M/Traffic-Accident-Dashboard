# 🚦 Traffic Accident Analysis Project

An end-to-end **Data Analysis, Visualization, and Machine Learning project** focused on understanding traffic accident patterns and predicting conditions using real-world–style accident data. The project includes **EDA, feature engineering, ML modeling, and an interactive Streamlit dashboard**.

---

## 📌 Project Overview

Road traffic accidents are a major public safety concern worldwide. This project analyzes traffic accident data to:
- Identify accident patterns and high-risk conditions
- Visualize trends using interactive dashboards
- Apply machine learning to predict accident-related conditions (Day vs Night)

The project follows a complete **data science lifecycle**, from data collection to model evaluation.

---

## 🧭 Project Workflow

Data Collection → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Visualization Dashboard → Machine Learning Model → Evaluation

<img width="376" height="1172" alt="graphviz (1)" src="https://github.com/user-attachments/assets/53cbb6a6-e1ce-457a-bb24-e9b15626b542" />

---

## 📂 Dataset Information

- **Source:** Kaggle-style synthetic traffic accident dataset
- **Format:** CSV
- **Type:** Structured tabular data
- **Features:**
  - Accident details (City, State, Date)
  - Environmental factors (Weather, Road Condition)
  - Severity indicators (Injuries, Fatalities)
  - Time-related attributes (Month, Sunrise/Sunset)

---

## 🛠️ Technologies Used

### Programming & Analysis
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

### Visualization & Deployment
- Streamlit
- Graphviz (for workflow visualization)

---

## 📊 Exploratory Data Analysis (EDA)

The following analyses and visualizations were performed:
- Accidents by City
- Accident Severity Distribution (Donut Chart)
- Accidents by Weather Condition
- Accident Severity Trend Over Time
- Severity Probability by Road Condition

These visualizations help uncover hidden patterns and risk factors associated with traffic accidents.

---

## ⚙️ Feature Engineering

- Creation of `Severity_Label` based on injuries and fatalities
- Extraction of `Month` from accident date
- Selection of numerical features for model training
- Target variable preparation for classification

---

## 🤖 Machine Learning Model

- **Algorithm Used:** Logistic Regression
- **Problem Type:** Binary Classification (Day vs Night)
- **Preprocessing:**
  - Train–Test Split (80:20)
  - Feature Scaling using StandardScaler

### 📈 Model Evaluation
- Accuracy Score
- Confusion Matrix
- Classification Report

---

## 🌐 Interactive Dashboard

An interactive dashboard was built using **Streamlit** to visualize accident insights dynamically.

🔗 **Live Dashboard:**  
https://fauzia-shereen-m-traffic-accident-analysis-in-data-s-app-v4btz0.streamlit.app/


🔗 **Dashboard Source Code:**  
https://github.com/Fauzia-Shereen-M/Traffic-Accident-Analysis-In-Data-Science-
---

## 📁 Project Structure

```
Traffic-Accident-Analysis/
│
├── traffic_accident_full_kaggle_style_dataset.csv
├── PROJECT.ipynb
├── PROJECT_requirement.txt
├── app.py
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run the Project

1. Clone the repository
```bash
git clone https://github.com/Fauzia-Shereen-M/Traffic-Accident-Analysis-In-Data-Science-.git
```

2. Install required packages
```bash
pip install -r requirements.txt
pip install -r PROJECT_requirement.txt
```

3. Run analysis script
```bash
python PROJECT.ipynb
```

4. Launch Streamlit dashboard
```bash
streamlit run app.py
```

##🔮 Predictive Modeling (Text-Based Output)

This module performs predictive analysis of traffic accidents using historical data. A Logistic Regression model is trained to classify accident severity into Severe/Fatal or Minor/Safe based on numerical features such as Injuries and Fatalities.

The dataset is preprocessed by creating a binary target variable and applying feature scaling using StandardScaler. To improve interpretability, the machine learning output is combined with rule-based conditional logic and weather conditions to generate clear, text-based risk predictions.

User inputs (injuries, fatalities, and weather) are processed to produce risk levels such as Very Low, Low, Moderate, High, or Very High Risk, without using any visualizations. This approach supports future accident risk prediction and assists in road safety planning and decision-making.

---

## 🚀 Future Scope

- Integration of real-time traffic and weather data
- GIS-based accident hotspot mapping
- Use of advanced ML/DL models for severity prediction
- Adoption by traffic authorities for safety planning

---

## 👤 Author

**Fauzia Shereen M**  
Data Science & Machine Learning Project

---

## 📜 License

This project is intended for **academic and educational purposes**.

---

⭐ *If you find this project useful, feel free to star the repository!*
