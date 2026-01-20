# 🚦 Traffic Accident Analysis Project

An end-to-end **Data Analysis, Visualization, and Machine Learning project** focused on understanding traffic accident patterns and predicting accident-related conditions using real-world–style traffic accident data.  
The project includes **Exploratory Data Analysis (EDA), feature engineering, machine learning modeling, and an interactive Streamlit dashboard**.

---

## 📌 Project Overview

Road traffic accidents are a major public safety concern worldwide. This project analyzes traffic accident data to:

- Identify accident patterns and high-risk conditions  
- Visualize accident trends using interactive dashboards  
- Apply machine learning techniques to predict accident-related conditions (Day vs Night)  

The project follows a complete **data science lifecycle**, from data collection and preprocessing to model evaluation and deployment.

---

## 🧭 Project Workflow

**Data Collection → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Visualization Dashboard → Machine Learning Model → Evaluation**

<p align="center">
  <img width="376" height="1172" alt="Project Workflow" src="https://github.com/user-attachments/assets/53cbb6a6-e1ce-457a-bb24-e9b15626b542" />
</p>

---

## 📂 Dataset Information

- **Source:** Kaggle-style synthetic traffic accident dataset  
- **Format:** CSV  
- **Type:** Structured tabular data  

### Dataset Features
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

These analyses help uncover hidden patterns and risk factors associated with traffic accidents.

---

## ⚙️ Feature Engineering

- Creation of `Severity_Label` based on injuries and fatalities  
- Extraction of `Month` from accident date  
- Selection of numerical features for model training  
- Preparation of target variable for classification  

---

## 🤖 Machine Learning Model

- **Algorithm Used:** Logistic Regression  
- **Problem Type:** Binary Classification (Day vs Night)  

### Preprocessing Steps
- Train–Test Split (80:20)  
- Feature Scaling using StandardScaler  

### 📈 Model Evaluation
- Accuracy Score  
- Confusion Matrix  
- Classification Report  

---

## 🔮 Predictive Modeling (Text-Based Output)

This module performs **predictive analysis of traffic accidents** using historical data. A **Logistic Regression model** is trained to classify accident severity into **Severe/Fatal** or **Minor/Safe** based on numerical features such as **Injuries** and **Fatalities**.

To improve interpretability, machine learning predictions are combined with **rule-based conditional logic and weather conditions** to generate **clear, text-based risk predictions**.

### Risk Levels Generated
- Very Low Risk  
- Low Risk  
- Moderate Risk  
- High Risk  
- Very High Risk  

No visualizations are used in this module, making it suitable for **decision-support systems and future road safety planning**.

---

## 🌐 Interactive Dashboard

An interactive dashboard was developed using **Streamlit** to visualize accident insights dynamically.

🔗 **Live Dashboard**  
https://fauzia-shereen-m-traffic-accident-analysis-in-data-s-app-v4btz0.streamlit.app/

🔗 **Dashboard Source Code**  
https://github.com/Fauzia-Shereen-M/Traffic-Accident-Analysis-In-Data-Science-

---

## 📁 Project Structure

Traffic-Accident-Analysis/
│
├── traffic_accident_full_kaggle_style_dataset.csv
├── PROJECT.ipynb
├── PROJECT_requirement.txt
├── app.py
├── README.md
└── requirements.txt




--- 
## ▶️ How to Run the Project 

1. Clone the repository
bash
git clone https://github.com/Fauzia-Shereen-M/Traffic-Accident-Analysis-In-Data-Science-.git

2. Install required packages
bash
pip install -r requirements.txt
pip install -r PROJECT_requirement.txt

3. Run analysis script
bash
python PROJECT.ipynb

4. Launch Streamlit dashboard
bash
streamlit run app.py

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
