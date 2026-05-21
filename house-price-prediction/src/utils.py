import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def load_data(filepath):
    return pd.read_csv(filepath)

def build_preprocessor(numeric_features, categorical_features):
    """Builds a scikit-learn preprocessing pipeline."""
    # Numeric pipeline: Impute missing values with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute missing with mode, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def evaluate_regression_model(model, X_test, y_test, model_name, output_dir):
    """Evaluates the model and generates residual and actual vs predicted plots."""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    # Actual vs Predicted Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'Actual vs Predicted: {model_name}')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predicted Sale Price')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.replace(" ", "_").lower()}_actual_vs_predicted.png'))
    plt.close()
    
    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='purple', bins=30)
    plt.title(f'Residuals Distribution: {model_name}')
    plt.xlabel('Residual Error')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.replace(" ", "_").lower()}_residuals.png'))
    plt.close()
    
    return metrics

def plot_model_comparison(metrics_df, output_dir):
    """Plot bar chart comparing model performances (RMSE & R2)."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot RMSE
    sns.barplot(x='Model', y='RMSE', data=metrics_df, ax=ax1, color='skyblue', label='RMSE')
    ax1.set_ylabel('RMSE', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot R2 on secondary axis
    ax2 = ax1.twinx()
    sns.lineplot(x='Model', y='R2', data=metrics_df, ax=ax2, color='red', marker='o', label='R2 Score', linewidth=2)
    ax2.set_ylabel('R2 Score', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1)
    
    plt.title('Model Performance Comparison (RMSE and R²)')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)
