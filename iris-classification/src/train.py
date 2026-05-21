import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import utils

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Iris.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')
REPORTS_DIR = os.path.join(BASE_DIR, 'outputs', 'reports')

# Create dirs if not exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_eda_plots(df):
    """Generate and save basic EDA plots."""
    print("Generating EDA plots...")
    # Pairplot
    plt.figure(figsize=(10, 8))
    sns.pairplot(df.drop(columns=['Id'], errors='ignore'), hue='species', palette='Dark2', diag_kind='kde')
    plt.savefig(os.path.join(PLOTS_DIR, 'pairplot.png'))
    plt.close()
    
    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    numeric_df = df.drop(columns=['Id', 'species'], errors='ignore')
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
    plt.close()

def plot_feature_importance(model, feature_names):
    """Plot feature importance for Decision Tree."""
    plt.figure(figsize=(8, 5))
    importance = model.feature_importances_
    sns.barplot(x=importance, y=feature_names, palette='viridis')
    plt.title('Feature Importance (Decision Tree)')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance_dt.png'))
    plt.close()

def main():
    print("Loading data...")
    df = utils.load_data(DATA_PATH)
    
    # Generate EDA plots
    generate_eda_plots(df)
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, le = utils.preprocess_data(df, random_state=42)
    class_names = le.classes_
    feature_names = df.drop(columns=['Id', 'species'], errors='ignore').columns.tolist()
    
    # Initialize models
    print("Training models...")
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=200),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    # Hyperparameter tuning for k-NN
    print("Tuning k-NN hyperparameters...")
    knn_param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}
    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring='accuracy')
    knn_grid.fit(X_train, y_train)
    models['k-NN (Tuned)'] = knn_grid.best_estimator_
    print(f"Best k-NN params: {knn_grid.best_params_}")
    
    # Train and evaluate all models
    results = []
    best_model = None
    best_acc = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        if name != 'k-NN (Tuned)':
            model.fit(X_train, y_train)
            
        metrics, report = utils.evaluate_model(model, X_test, y_test, name, class_names, PLOTS_DIR)
        metrics['Model'] = name
        results.append(metrics)
        
        # Save feature importance for DT
        if name == 'Decision Tree':
            plot_feature_importance(model, feature_names)
            
        # Check if best model
        if metrics['Accuracy'] > best_acc:
            best_acc = metrics['Accuracy']
            best_model = model
            best_model_name = name
            
    # Compile results
    results_df = pd.DataFrame(results)
    print("\nModel Performance Comparison:")
    print(results_df.to_string(index=False))
    
    # Plot model comparison
    utils.plot_model_comparison(results_df, PLOTS_DIR)
    
    # Save the best model
    print(f"\nBest Model: {best_model_name} with Accuracy {best_acc:.4f}")
    save_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    utils.save_model(best_model, scaler, le, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
