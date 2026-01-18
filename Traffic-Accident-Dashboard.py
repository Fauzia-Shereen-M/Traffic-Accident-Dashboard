import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import streamlit as st

# ==============================
# 1. DATA PREP & KPI CALCULATIONS
# ==============================
df = pd.read_csv("traffic_accident_full_kaggle_style_dataset.csv")

# Clean & Prepare
df["Accident_Severity"] = df["Accident_Severity"].str.title()
df["Weather"] = df["Weather"].str.title()
df["Accident_Date"] = pd.to_datetime(df["Accident_Date"])
df["Month"] = df["Accident_Date"].dt.strftime("%b")
df["Severity_Label"] = df["Accident_Severity"].replace("Major", "Severe")
df.loc[(df["Injuries"] == 0) & (df["Fatalities"] == 0), "Severity_Label"] = "Safe"

# ML Logic
ml_df = df.copy()
ml_df['Sunrise_Sunset'] = ml_df['Sunrise_Sunset'].map({'Day': 1, 'Night': 0})
df_numeric = ml_df.select_dtypes(include=['number']).dropna()
X_ml = df_numeric.drop(columns=['Sunrise_Sunset'])
y_ml = df_numeric['Sunrise_Sunset']
X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

# KPI Values
total_accidents = len(df)
total_injuries = df["Injuries"].sum()
total_fatalities = df["Fatalities"].sum()
severe_accidents = len(df[df["Severity_Label"] == "Severe"])

# ==============================
# 2. DASHBOARD LAYOUT (Updated Spacing)
# ==============================
fig = plt.figure(figsize=(24, 15))
fig.patch.set_facecolor('#2A9D8F') 

ax_bg = fig.add_axes([0.02, 0.02, 0.96, 0.96])
ax_bg.axis('off')
board = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.01,rounding_size=0.02", 
                       facecolor="#CAF0F8", edgecolor="none") 
ax_bg.add_patch(board)

# We lower 'top' slightly to 0.80 to accommodate the extra space from the title
gs = fig.add_gridspec(2, 3, left=0.05, right=0.95, top=0.80, bottom=0.08, hspace=0.35, wspace=0.3)
sns.set_style("whitegrid")

# ==============================
# 3. PLACING KPI NUMBERS (Top Header)
# ==============================
kpi_data = [
    (f"{total_accidents:,}", "Total Accidents", 0.20, "#1D6D64"),
    (f"{total_injuries:,}", "Total Injuries", 0.40, "#2A9D8F"),
    (f"{total_fatalities:,}", "Total Fatalities", 0.60, "#E76F51"),
    (f"{severe_accidents:,}", "Severe Accidents", 0.80, "#E9C46A")
]

# The change from 0.91 to 0.88 creates the line of space above the text
for val, label, x_pos, clr in kpi_data:
    fig.text(x_pos, 0.88, val, fontsize=28, weight="bold", ha="center", color=clr) # Lowered y-pos
    fig.text(x_pos, 0.85, label, fontsize=12, weight="bold", ha="center", color="#444444") # Lowered y-pos

# ==============================
# 4. GRAPHS (Standard 2x3)
# ==============================

# CHART 1
ax1 = fig.add_subplot(gs[0, 0])
city_counts = df["City"].value_counts().head(7)
colors1 = ["#1D6D64","#2A9D8F","#48CAE4","#90E0EF","#ADE8F4","#BDE0FE","#CAF0F8"]
city_counts.plot(kind="bar", color=colors1, ax=ax1)
ax1.set_title("1. Accidents by City", fontsize=15, fontweight="bold", color="#1D6D64")

# CHART 2: Severity Distribution
ax2 = fig.add_subplot(gs[0, 1])

labels = ["Severe", "Minor", "Fatal", "Safe"]
sizes = [df["Severity_Label"].value_counts().get(l,0) for l in labels]
colors2 = ["#1D6D64", "#90E0EF", "#E76F51", "#E9C46A"]

# Capture 'wedges' to create the bullets in the legend
wedges, texts, autotexts = ax2.pie(sizes, colors=colors2, autopct="%1.1f%%", 
                                  startangle=140, pctdistance=0.75,
                                  wedgeprops={'edgecolor': 'white', 'linewidth': 2})

# Donut hole effect
ax2.add_artist(plt.Circle((0,0), 0.45, color='#B2DFDB', alpha=0.3, zorder=10))
plt.setp(autotexts, size=10, weight="bold", color="white")

# --- ADDED: 4 Bullets with Names and Colors ---
ax2.legend(wedges, labels,
          title="Severity",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1), # Places bullets to the right of the donut
          fontsize=10,
          frameon=False) # Removes box for a clean look

ax2.set_title("2. Severity Distribution", fontsize=15, fontweight="bold", color="#1D6D64")

# CHART 3
ax3 = fig.add_subplot(gs[0, 2])
w_counts = df["Weather"].value_counts().sort_values()
color_map = {"Rainy": "#4B0082", "Clear": "#E9C46A", "Stormy": "#E76F51", "Foggy": "#90E0EF", "Unknown": "#008080"}
colors3 = [color_map.get(w, "#008080") for w in w_counts.index]
ax3.barh(w_counts.index, w_counts.values, color=colors3)
ax3.set_title("3. Weather Conditions", fontsize=15, fontweight="bold", color="#1D6D64")

# CHART 4
ax4 = fig.add_subplot(gs[1, 0])
trend = df.groupby(["Month", "Severity_Label"]).size().unstack(fill_value=0)
ax4.plot(trend.index, trend["Severe"], marker="o", linewidth=3, label="Severe")
ax4.plot(trend.index, trend["Minor"], marker="o", linewidth=3, label="Minor")
ax4.plot(trend.index, trend["Fatal"], marker="o", linewidth=3, label="Fatal")
ax4.set_title("4. Severity Trend Over Time", fontsize=15, fontweight="bold", color="#1D6D64")
ax4.legend()

# CHART 5
ax5 = fig.add_subplot(gs[1, 1])
ct = pd.crosstab(df['Road_Condition'], df['Severity_Label'], normalize='index') * 100
ct = ct.reindex(columns=['Minor', 'Severe', 'Fatal'])
ct.plot(kind='barh', stacked=True, color=['#90E0EF', '#1D6D64', '#E76F51'], ax=ax5)
ax5.set_title("5. Road Condition Probability", fontsize=15, fontweight="bold", color="#1D6D64")

# CHART 6
ax6 = fig.add_subplot(gs[1, 2])
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax6,
            xticklabels=['Night (0)', 'Day (1)'], yticklabels=['Night (0)', 'Day (1)'])
ax6.set_title('6. ML: Day vs Night Confusion Matrix', fontsize=15, fontweight="bold", color="#1D6D64")
ax6.set_xlabel('Predicted')
ax6.set_ylabel('Actual')

plt.suptitle("TRAFFIC ACCIDENT ANALYSIS DASHBOARD\n\n", 
             fontsize=24, fontweight="bold", y=0.96, color="#1D6D64")

st.pyplot(fig)
