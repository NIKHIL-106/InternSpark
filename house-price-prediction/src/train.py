import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import utils

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'housing.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_eda_plots(df):
    print("Generating EDA plots...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 1. Price Distribution (Skewness)
    plt.figure(figsize=(8, 6))
    sns.histplot(df['SalePrice'], kde=True, color='blue', bins=30)
    plt.title('Sale Price Distribution')
    plt.xlabel('Sale Price')
    plt.savefig(os.path.join(PLOTS_DIR, 'price_distribution.png'))
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
    plt.close()

def main():
    print("Loading data...")
    df = utils.load_data(DATA_PATH)
    
    generate_eda_plots(df)
    
    print("Preparing data...")
    # Drop Id
    df = df.drop(columns=['Id'], errors='ignore')
    
    # Identify features
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    
    # Optional: Log transform the target since it's skewed (we will just train on raw for simplicity or do log transform)
    # y = np.log1p(y) 
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build preprocessor
    preprocessor = utils.build_preprocessor(numeric_features, categorical_features)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    # Train and evaluate models
    results = []
    trained_pipelines = {}
    best_rmse = float('inf')
    best_model_name = ""
    best_pipeline = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        # Create full pipeline (preprocessing + modeling)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        
        # Hyperparameter Tuning for Tree models
        if name == 'Random Forest':
            print("Tuning Random Forest...")
            param_grid = {'model__n_estimators': [50, 100], 'model__max_depth': [10, None]}
            grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            pipeline = grid.best_estimator_
            print(f"Best params for RF: {grid.best_params_}")
        elif name == 'Gradient Boosting':
            print("Tuning Gradient Boosting...")
            param_grid = {'model__n_estimators': [100, 150], 'model__learning_rate': [0.05, 0.1]}
            grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            pipeline = grid.best_estimator_
            print(f"Best params for GB: {grid.best_params_}")
        else:
            pipeline.fit(X_train, y_train)
            
        trained_pipelines[name] = pipeline
        
        # Evaluate
        print(f"Evaluating {name}...")
        metrics = utils.evaluate_regression_model(pipeline, X_test, y_test, name, PLOTS_DIR)
        results.append(metrics)
        
        if metrics['RMSE'] < best_rmse:
            best_rmse = metrics['RMSE']
            best_model_name = name
            best_pipeline = pipeline
            
    # Compare
    results_df = pd.DataFrame(results)
    print("\\nModel Performance:")
    print(results_df.to_string(index=False))
    
    utils.plot_model_comparison(results_df, PLOTS_DIR)
    
    print(f"\\nBest Model: {best_model_name} with RMSE {best_rmse:.2f}")
    save_path = os.path.join(MODEL_DIR, 'best_house_model.pkl')
    utils.save_model(best_pipeline, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
