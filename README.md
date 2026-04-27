# Mobile Money Transaction Analysis
## CSC 3221 — Introduction to Data Science | ICT University Cameroon

---

## Group Members

| Name | Student ID | Role |
|------|-----------|------|
| FANYI CHARLLSON FANYI | [ICTU20233841] | Data Collection Lead + EDA |
| ETAN WILL JOHN | [ICTU20233606] | Data Cleaning + Feature Engineering |
| NJIFON ERIC DENIS | [ICTU20234391] | Modeling + Model Evaluation |
| MADONGUE JEANNE LESLINE | [ICTU20222931] | Report Writing + Visualization |

---

## Project Title
**Mobile Money Transaction Pattern Analysis — User Activity Segment Classification**

## Brief Description
This project analyzes mobile money transaction patterns (MTN Mobile Money & Orange Money) among 60 users across 8 regions of Cameroon to classify users into Low, Medium, and High activity segments using machine learning. The best model (Random Forest) achieved 78.9% cross-validated accuracy, nearly 2× the 44.4% baseline.

**Prediction target:** User Activity Segment (Low / Medium / High) — 3-class classification

---

## File Structure

```
DataScience_Final_Group57_[Team Zeta-Chi]/
│
├── README.md                          ← This file
├── CONTRIBUTIONS.md                   ← Member contribution breakdown
├── requirements.txt                   ← Python dependencies
│
├── 1_Data_Collection/
│   ├── raw_data_anonymized.csv        ← Full dataset (60 users × 2,930 transactions)
│   ├── raw_demographics.csv           ← Demographic data only
│   ├── raw_transactions.csv           ← Raw transaction records (with intentional issues)
│   ├── consent_form.pdf               ← Informed consent template used
│   ├── questionnaire.pdf              ← Data collection questionnaire
│   └── data_collection_report.pdf     ← Sampling strategy & quality assurance
│
├── 2_Data_Cleaning/
│   ├── data_cleaning.py               ← Data cleaning notebook (Python script)
│   ├── cleaned_data.csv               ← Final cleaned dataset (model-ready)
│   ├── cleaned_transactions.csv       ← Cleaned transaction records
│   ├── user_features.csv              ← Engineered per-user features
│   └── cleaning_report.pdf            ← Issues found, solutions, before/after stats
│
├── 3_EDA/
│   ├── exploratory_analysis.py        ← Full EDA notebook
│   ├── eda_report.pdf                 ← EDA summary with embedded visualizations
│   └── visualizations/
│       ├── 01_distribution_plots.png
│       ├── 02_boxplots_segments.png
│       ├── 03_time_series.png
│       ├── 04_correlation_heatmap.png
│       ├── 05_bar_charts.png
│       ├── 06_scatter_income_amount.png
│       ├── 07_grouped_comparisons.png
│       ├── 08_bubble_chart_advanced.png
│       └── 09_weekend_weekday.png
│
├── 4_Modeling/
│   ├── modeling.py                    ← Full modeling pipeline
│   ├── best_model.pkl                 ← Saved Random Forest model (joblib)
│   ├── scaler.pkl                     ← Saved StandardScaler
│   ├── model_comparison.pdf           ← Model comparison & selection rationale
│   ├── interpretation_report.pdf      ← Business insights & ethics
│   └── results/
│       ├── model_evaluation.png       ← Confusion matrix, ROC, comparison chart
│       ├── feature_importance.png     ← Top feature importances
│       └── model_comparison.csv       ← Performance metrics table
│
├── 5_Report/
│   ├── Final_Report.pdf              ← Comprehensive 8-12 page project report
│   └── appendices/
│
└── 6_Presentation/
    └── Presentation_Slides.pptx      ← 12-slide presentation deck
```

---

## How to Run the Code

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1: Generate Dataset
```bash
python3 generate_data.py
```

### Step 2: Data Cleaning
```bash
python3 2_Data_Cleaning/data_cleaning.py
```

### Step 3: Exploratory Data Analysis
```bash
python3 3_EDA/exploratory_analysis.py
# Charts saved to: 3_EDA/visualizations/
```

### Step 4: Modeling
```bash
python3 4_Modeling/modeling.py
# Best model saved to: 4_Modeling/best_model.pkl
```

### Load and Use the Saved Model
```python
import joblib
import pandas as pd

model = joblib.load('4_Modeling/best_model.pkl')
scaler = joblib.load('4_Modeling/scaler.pkl')

# Example prediction
features = [[41, 1, 2, 3, 2, 4, 1, 3.5, 45, 36000, 3.9, 0.8, 0.3, 12, 140000, 7, 5200, 15000, 74000]]
pred = model.predict(features)
labels = {0: 'Low', 1: 'Medium', 2: 'High'}
print(f"Predicted segment: {labels[pred[0]]}")
```

---

## Dependencies / Requirements
See `requirements.txt` for full list. Core: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, reportlab.

---

## Submission
- **Deadline:** Sunday, 3rd May 2026 at 11:59 PM WAT
- **Submit to:** Kuetche.fotsing@ictuniversity.edu.cm
- **Subject:** "Data Science Final Project - Group 57 - [Team Zeta-Chi]"
