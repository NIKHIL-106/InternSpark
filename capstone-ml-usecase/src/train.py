import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.pipeline import Pipeline
import joblib

import preprocess
import evaluate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'customer_churn.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')
REPORTS_DIR = os.path.join(BASE_DIR, 'outputs', 'reports')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_eda_plots(df):
    """Generate high-quality business EDA plots."""
    print("Generating EDA plots...")
    
    # 1. Target Class Imbalance
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df, palette='Set2')
    plt.title('Customer Churn Distribution (Class Imbalance)')
    plt.savefig(os.path.join(PLOTS_DIR, 'churn_distribution.png'))
    plt.close()
    
    # 2. Churn by Contract Type
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Contract', hue='Churn', data=df, palette='viridis')
    plt.title('Churn by Contract Type')
    plt.savefig(os.path.join(PLOTS_DIR, 'churn_by_contract.png'))
    plt.close()
    
    # 3. Monthly Charges vs Churn
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set1')
    plt.title('Monthly Charges Impact on Churn')
    plt.savefig(os.path.join(PLOTS_DIR, 'monthly_charges_churn.png'))
    plt.close()
    
    # 4. Tenure vs Churn
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x='Tenure', hue='Churn', fill=True, common_norm=False, palette='Set2')
    plt.title('Tenure Distribution by Churn Status')
    plt.savefig(os.path.join(PLOTS_DIR, 'tenure_distribution.png'))
    plt.close()

def main():
    print("Loading data...")
    df = preprocess.load_data(DATA_PATH)
    
    generate_eda_plots(df)
    
    print("Applying Feature Engineering...")
    df = preprocess.feature_engineering(df)
    
    X = df.drop(columns=['CustomerID', 'Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessor
    preprocessor, num_feats, cat_feats, pass_feats = preprocess.get_preprocessor(df)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    # For XGBoost, target must be 0/1
    y_train_num = y_train.map({'No':0, 'Yes':1})
    y_test_num = y_test.map({'No':0, 'Yes':1})
    
    results = []
    best_roc = 0
    best_model_name = ""
    best_pipeline = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        
        y_train_target = y_train_num if name == 'XGBoost' else y_train
        y_test_target = y_test_num if name == 'XGBoost' else y_test
        
        # Hyperparameter Tuning
        if name == 'Random Forest':
            print("Tuning Random Forest...")
            param_grid = {'model__n_estimators': [100, 200], 'model__max_depth': [5, 10, None]}
            grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train_target)
            pipeline = grid.best_estimator_
            print(f"Best params: {grid.best_params_}")
            
        elif name == 'XGBoost':
            print("Tuning XGBoost...")
            param_grid = {'model__n_estimators': [100, 150], 'model__learning_rate': [0.05, 0.1]}
            grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train_target)
            pipeline = grid.best_estimator_
            print(f"Best params: {grid.best_params_}")
            
        else:
            pipeline.fit(X_train, y_train_target)
            
        print(f"Evaluating {name}...")
        # Evaluate model. For XGBoost we need to map back the predictions to 'Yes'/'No' for consistent evaluation if using `y_test` natively, 
        # but our evaluate.py expects y_test to be 'Yes'/'No' and handles it internally. Wait, if model predicts 0/1, we must handle it.
        # Let's map predict back to 'Yes'/'No' for evaluate.py
        if name == 'XGBoost':
            class XGBWrapper:
                def __init__(self, pipe):
                    self.pipe = pipe
                def predict(self, X):
                    preds = self.pipe.predict(X)
                    return np.array(['Yes' if p==1 else 'No' for p in preds])
                def predict_proba(self, X):
                    return self.pipe.predict_proba(X)
            metrics = evaluate.evaluate_model(XGBWrapper(pipeline), X_test, y_test, name, PLOTS_DIR)
        else:
            metrics = evaluate.evaluate_model(pipeline, X_test, y_test, name, PLOTS_DIR)
            
        results.append(metrics)
        
        if metrics['ROC-AUC'] > best_roc:
            best_roc = metrics['ROC-AUC']
            best_model_name = name
            best_pipeline = pipeline
            
    results_df = pd.DataFrame(results)
    print("\\nModel Performance:")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(REPORTS_DIR, 'model_metrics.csv'), index=False)
    
    evaluate.plot_model_comparison(results_df, PLOTS_DIR)
    
    print(f"\\nBest Model: {best_model_name} with ROC-AUC {best_roc:.4f}")
    
    # SHAP Explainability
    if best_model_name in ['Random Forest', 'XGBoost']:
        evaluate.generate_shap_explainer(best_pipeline, X_train.sample(500, random_state=42), PLOTS_DIR, num_feats, cat_feats, pass_feats)
    
    # Save Best Model
    save_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    # If XGBoost, best to save the native pipeline
    joblib.dump(best_pipeline, save_path)
    print(f"Model successfully saved to {save_path}")

if __name__ == "__main__":
    main()
