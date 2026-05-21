# 📊 Capstone ML Use Case: End-to-End Business Intelligence Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-brightgreen)

## Project Overview
This project simulates a **real-world industry ML workflow** focused on **Customer Churn Prediction**. Designed as a highly professional internship capstone, it emphasizes practical machine learning, advanced feature engineering, business storytelling, and actionable intelligence.

## Business Problem
**Telecom Customer Churn**: Retaining existing customers is vastly more profitable than acquiring new ones. The goal is to predict which customers are at risk of leaving the service so the retention team can intervene proactively, thereby reducing Customer Acquisition Costs (CAC) and increasing Customer Lifetime Value (LTV).

## Dataset Information
A realistic synthetic dataset was generated modeling standard Telecom operations. It includes:
- **Demographics**: Gender, Senior Citizen status, Partners.
- **Account Data**: Tenure, Contract Type (Month-to-month, 1-Year, 2-Year), Charges.
- **Services**: Internet Type, Tech Support.
- **Target**: Churn (Yes/No).

## Workflow & Methodology
1. **Data Preprocessing**: Implemented `scikit-learn`'s `Pipeline` and `ColumnTransformer` to handle missing values (Median/Mode imputation) and apply `StandardScaler` & `OneHotEncoder`.
2. **Feature Engineering**: Created critical business features like `TenureGroup`, `SpendingCategory`, and a composite `IsHighRisk` indicator flag.
3. **Model Selection**: Evaluated Logistic Regression, Random Forest, and **XGBoost**.
4. **Hyperparameter Tuning**: Utilized `GridSearchCV` to optimize tree depths and learning rates.
5. **Explainable AI (XAI)**: Applied **SHAP** values to decode the model decisions into simple business language.

## Model Comparison & Evaluation Results
- **XGBoost** emerged as the superior model, balancing Accuracy (~85%) and ROC-AUC efficiently.
- Complex non-linear patterns regarding pricing and tenure were captured flawlessly by the gradient boosting architecture.

## Business Insights & Recommendations
Through SHAP analysis and EDA, the following insights were uncovered:
1. **Contract Type**: Month-to-month contracts are the #1 driver of churn. *Recommendation: Incentivize long-term lock-ins with slight discounts.*
2. **Tenure Flight Risk**: The highest churn occurs in the first 6 months. *Recommendation: Overhaul the 90-day onboarding sequence.*
3. **Service Friction**: Fiber optic users without tech support churn rapidly. *Recommendation: Bundle tech support with premium speeds.*

## Installation & Usage Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd capstone-ml-usecase
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete Training Pipeline:**
   ```bash
   python src/train.py
   ```

4. **Prediction Inference Example:**
   Use the `predict.py` script to test the model dynamically.
   ```bash
   python src/predict.py
   ```

5. **Generate Presentation & PDF:**
   ```bash
   python src/generate_presentation.py
   ```

## Future Improvements
- **Streamlit Web App**: Deploy the `predict.py` pipeline into a web UI for non-technical customer service agents.
- **LTV Modeling**: Predict the actual financial loss of churning customers by integrating regression for Total Lifetime Value.
