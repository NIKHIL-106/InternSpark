# 🚢 Titanic Survival Prediction Project

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-brightgreen)

## Project Overview
This project is an end-to-end Machine Learning classification pipeline built to predict passenger survival on the Titanic. It goes beyond basic modeling, emphasizing **Advanced Feature Engineering**, **Explainable AI (XAI)**, and production-ready `scikit-learn` preprocessing pipelines.

## Dataset Information
The project uses the classic **Titanic Survival Dataset**. It contains demographic and travel details for passengers, including:
- `Pclass`: Passenger Class (1, 2, 3)
- `Sex` & `Age`
- `SibSp` & `Parch`: Family relations aboard
- `Fare` & `Embarked`: Ticket price and Port of Embarkation
- `Survived`: Target Variable (0 = No, 1 = Yes)

## Technologies Used
- **Python**: Core programming language.
- **Pandas & NumPy**: Data processing and manipulation.
- **Matplotlib & Seaborn**: High-quality visual storytelling and EDA.
- **Scikit-Learn**: Machine learning modeling, hyperparameter tuning, and pipeline creation.
- **SHAP**: Explainable AI for interpreting complex ensemble predictions.
- **Joblib**: Model serialization.

## Workflow
1. **Data Collection**: Fetching data programmatically.
2. **Exploratory Data Analysis (EDA)**: Visualizing survival distributions, class imbalance, and demographics.
3. **Advanced Feature Engineering**: 
   - Extracting `Title` from names (e.g., Mr., Mrs., Master).
   - Creating `FamilySize` and `IsAlone` indicators.
   - Engineering `HasCabin` from sparse cabin data.
4. **Data Preprocessing**: Handling missing values with `SimpleImputer` and applying `OneHotEncoder` & `StandardScaler` inside a unified `ColumnTransformer`.
5. **Model Training**: Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting.
6. **Explainable AI**: Utilizing SHAP to visualize and interpret feature importance.

## Installation
1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd titanic-survival-prediction
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

## How to Run
Run the training script to process the data, generate plots, and save the best model:
```bash
python src/train.py
```

Generate the beautifully formatted Jupyter Notebook:
```bash
python create_notebook.py
```

## Prediction Example
The `predict.py` script automatically loads the robust model pipeline and predicts survival based on simulated input data.

```bash
python src/predict.py
```

**Expected Output:**
```
Input Passenger Data:
Pclass: 3
Sex: male
Age: 22
...
Prediction: Did Not Survive
```

## Results & Model Comparison
- **Gradient Boosting** achieved the highest ROC-AUC and Accuracy.
- **SHAP Analysis** conclusively showed that `Sex` (being Female) and `Pclass` (being in 1st class) were the most overwhelming positive predictors for survival, while being a lower-class male drastically reduced survival probability.

## Future Improvements
- **Deep Learning**: Apply a Multi-Layer Perceptron (MLP) to see if neural architectures can extract deeper patterns.
- **Web App**: Deploy a Flask or FastAPI frontend where users can enter their own details and see if they would have survived the Titanic!
