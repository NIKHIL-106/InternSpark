# 🏠 House Price Prediction Project

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## Project Overview
This project is an end-to-end Machine Learning regression pipeline to predict house prices based on various housing-related features. It is built as a robust, production-ready portfolio piece focusing heavily on **feature engineering**, **handling missing values**, and **model comparison**.

## Dataset Information
The project uses the **Ames Housing Dataset** (often featured in Kaggle's House Price Prediction competitions). It contains numerous features detailing every aspect of residential homes. For this project, a curated subset of critical features was used, including:
- `LotArea` & `LotFrontage`
- `OverallQual` (Overall Quality)
- `YearBuilt`
- `GrLivArea` (Above Grade Living Area)
- `SalePrice` (Target Variable)

## Technologies Used
- **Python**: Core programming language.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **Matplotlib & Seaborn**: Data visualization and EDA.
- **Scikit-Learn**: Machine learning modeling, preprocessing pipelines, and evaluation.
- **Joblib**: Model serialization.
- **Jupyter Notebook**: Interactive data exploration.

## Project Structure
```
house-price-prediction/
│
├── data/
│   └── housing.csv                   # The generated dataset
│
├── notebooks/
│   └── house_price_prediction.ipynb  # Interactive EDA & Modeling notebook
│
├── models/
│   └── best_house_model.pkl          # Saved best model pipeline
│
├── outputs/
│   ├── plots/                        # Saved EDA and Evaluation plots
│   └── reports/                      # Additional metrics
│
├── src/
│   ├── train.py                      # Script to train and tune models
│   ├── predict.py                    # Inference script
│   └── utils.py                      # Reusable helper functions
│
├── requirements.txt                  # Dependencies
├── README.md                         # Documentation
└── .gitignore                        # Git ignored files
```

## Project Workflow
1. **Data Collection**: Fetching data using OpenML.
2. **Exploratory Data Analysis (EDA)**: Visualizing price distributions, feature correlations, and missing values.
3. **Data Preprocessing Pipeline**:
   - *Numerical*: Median Imputation + Standard Scaling.
   - *Categorical*: Mode Imputation + One-Hot Encoding.
4. **Model Training & Tuning**: 
   - Linear Regression
   - Random Forest (Hyperparameter Tuned)
   - Gradient Boosting (Hyperparameter Tuned)
5. **Evaluation**: Utilizing RMSE, MAE, and R² scores.

## Installation Steps
1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd house-price-prediction
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Prediction Instructions & Example Usage
The `predict.py` script demonstrates how to load the saved preprocessing pipeline and model to predict the price of a new property.

**Run the inference script:**
```bash
python src/predict.py
```

**Expected Output:**
```
Input features:
LotFrontage          65.0
LotArea              8450
OverallQual             7
YearBuilt            2003
...
Predicted House Price: $210,500.00
```

## Model Comparison & Evaluation Metrics
The models are evaluated primarily on **RMSE** (Root Mean Squared Error) and **R²** (Coefficient of Determination). 
- **Gradient Boosting** and **Random Forest** successfully captured non-linear relationships and interactions.
- The comprehensive preprocessing pipeline effectively mitigated the impact of missing values and categorical scaling issues.

*(Check `outputs/plots/` for detailed Actual vs Predicted charts and Residual distributions.)*

## Future Improvements
- **Advanced Feature Engineering**: Creating composite features like `TotalBathrooms` or `TotalPorchArea`.
- **Advanced Algorithms**: Implementing XGBoost, LightGBM, or CatBoost.
- **Web App**: Developing a Streamlit or FastAPI frontend to allow users to input house features interactively and get instant price estimations.
