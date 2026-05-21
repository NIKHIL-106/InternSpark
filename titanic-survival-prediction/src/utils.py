import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import os
import shap

def load_data(filepath):
    return pd.read_csv(filepath)

def feature_engineering(df):
    """Perform feature engineering on Titanic dataset."""
    df = df.copy()
    
    # 1. Title Extraction
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # 2. Family Size
    # OpenML columns might be string, need to convert to float/int
    df['SibSp'] = pd.to_numeric(df['SibSp'], errors='coerce').fillna(0)
    df['Parch'] = pd.to_numeric(df['Parch'], errors='coerce').fillna(0)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # 3. IsAlone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 4. Cabin Presence
    df['HasCabin'] = df['Cabin'].notnull().astype(int)
    
    # Drop unneeded columns
    df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], errors='ignore', inplace=True)
    
    return df

def build_preprocessor():
    """Builds a scikit-learn preprocessing pipeline for Titanic dataset."""
    numeric_features = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch']
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title']
    passthrough_features = ['IsAlone', 'HasCabin']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('pass', 'passthrough', passthrough_features)
        ])
    
    return preprocessor, numeric_features, categorical_features, passthrough_features

def evaluate_classification_model(model, X_test, y_test, model_name, output_dir):
    """Evaluates model and generates metrics, confusion matrix plot."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'F1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.replace(" ", "_").lower()}_confusion_matrix.png'))
    plt.close()
    
    return metrics

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)
