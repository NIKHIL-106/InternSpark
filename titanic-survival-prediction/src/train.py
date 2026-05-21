import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import utils
import shap

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'titanic.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')
REPORTS_DIR = os.path.join(BASE_DIR, 'outputs', 'reports')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_eda_plots(df):
    """Generate professional EDA plots for Titanic."""
    print("Generating EDA plots...")
    
    # 1. Survival Count Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Survived', data=df, palette='Set2')
    plt.title('Survival Count (0 = No, 1 = Yes)')
    plt.savefig(os.path.join(PLOTS_DIR, 'survival_count.png'))
    plt.close()
    
    # 2. Gender vs Survival
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Sex', hue='Survived', data=df, palette='Set1')
    plt.title('Gender vs Survival')
    plt.savefig(os.path.join(PLOTS_DIR, 'gender_vs_survival.png'))
    plt.close()
    
    # 3. Pclass vs Survival
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Pclass', hue='Survived', data=df, palette='viridis')
    plt.title('Passenger Class vs Survival')
    plt.savefig(os.path.join(PLOTS_DIR, 'pclass_vs_survival.png'))
    plt.close()
    
    # 4. Age Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'].dropna(), kde=True, bins=30, color='skyblue')
    plt.title('Age Distribution')
    plt.savefig(os.path.join(PLOTS_DIR, 'age_distribution.png'))
    plt.close()
    
    # 5. Fare Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Fare'].dropna(), kde=True, bins=30, color='salmon')
    plt.title('Fare Distribution')
    plt.savefig(os.path.join(PLOTS_DIR, 'fare_distribution.png'))
    plt.close()
    
    # 6. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
    plt.close()

def plot_model_comparison(metrics_df):
    """Generate model comparison chart."""
    plt.figure(figsize=(10, 6))
    metrics_df.set_index('Model')[['Accuracy', 'F1', 'ROC_AUC']].plot(kind='bar', figsize=(10, 6), colormap='viridis')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'))
    plt.close()

def explain_model_shap(model_pipeline, X_train, preprocessor):
    """Generate SHAP feature importance plot for the best tree model."""
    print("Generating SHAP Explainability plots...")
    # Extract the underlying model and transformed data
    model = model_pipeline.named_steps['model']
    X_train_transformed = preprocessor.transform(X_train)
    
    # Retrieve feature names from OneHotEncoder and others
    numeric_features = preprocessor.transformers_[0][2]
    categorical_features = preprocessor.transformers_[1][2]
    passthrough_features = preprocessor.transformers_[2][2]
    
    # Get categorical names
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    
    all_feature_names = numeric_features + list(cat_feature_names) + passthrough_features
    
    # If the model is tree-based, use TreeExplainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_transformed)
        
        # Binary classification SHAP returns a list of arrays (one per class) in some versions, or single array
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[1] # positive class
        else:
            shap_values_plot = shap_values
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_plot, X_train_transformed, feature_names=all_feature_names, show=False)
        plt.title('SHAP Feature Importance Summary')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'shap_summary.png'))
        plt.close()
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        # Fallback to standard feature importance
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importance = model.feature_importances_
            sns.barplot(x=importance, y=all_feature_names, palette='viridis')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'))
            plt.close()

def main():
    print("Loading data...")
    df = utils.load_data(DATA_PATH)
    
    # EDA
    generate_eda_plots(df)
    
    print("Performing feature engineering...")
    df_engineered = utils.feature_engineering(df)
    
    # Split features and target
    X = df_engineered.drop(columns=['Survived'])
    y = df_engineered['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor, num_feats, cat_feats, pass_feats = utils.build_preprocessor()
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = []
    best_roc = 0
    best_model_name = ""
    best_pipeline = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        
        # Hyperparameter tuning
        if name == 'Random Forest':
            print("Tuning Random Forest...")
            param_grid = {'model__n_estimators': [100, 200], 'model__max_depth': [5, 10, None]}
            grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            pipeline = grid.best_estimator_
            print(f"Best RF params: {grid.best_params_}")
            
        elif name == 'Gradient Boosting':
            print("Tuning Gradient Boosting...")
            param_grid = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1]}
            grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            pipeline = grid.best_estimator_
            print(f"Best GB params: {grid.best_params_}")
            
        else:
            pipeline.fit(X_train, y_train)
            
        # Evaluate
        metrics = utils.evaluate_classification_model(pipeline, X_test, y_test, name, PLOTS_DIR)
        results.append(metrics)
        
        if metrics['ROC_AUC'] > best_roc:
            best_roc = metrics['ROC_AUC']
            best_model_name = name
            best_pipeline = pipeline

    results_df = pd.DataFrame(results)
    print("\\nModel Performance:")
    print(results_df.to_string(index=False))
    
    # Save metrics to reports
    results_df.to_csv(os.path.join(REPORTS_DIR, 'model_metrics.csv'), index=False)
    
    plot_model_comparison(results_df)
    
    print(f"\\nBest Model: {best_model_name} with ROC-AUC {best_roc:.4f}")
    
    # Explain Best Model
    if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
        # preprocessor is already fitted as part of pipeline
        fitted_preprocessor = best_pipeline.named_steps['preprocessor']
        explain_model_shap(best_pipeline, X_train, fitted_preprocessor)
    
    save_path = os.path.join(MODEL_DIR, 'best_titanic_model.pkl')
    utils.save_model(best_pipeline, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
