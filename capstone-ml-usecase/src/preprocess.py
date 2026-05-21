import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

def load_data(filepath):
    return pd.read_csv(filepath)

def feature_engineering(df):
    """Create new meaningful business features."""
    df = df.copy()
    
    # 1. Customer Tenure Groups
    bins = [0, 12, 24, 48, 60, 100]
    labels = ['0-1 Year', '1-2 Years', '2-4 Years', '4-5 Years', '5+ Years']
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=bins, labels=labels, right=False).astype(str)
    
    # 2. Spending Category
    spend_bins = [0, 40, 70, 90, 200]
    df['SpendingCategory'] = pd.cut(df['MonthlyCharges'], bins=spend_bins, labels=['Low', 'Medium', 'High', 'Premium'], right=False).astype(str)
    
    # 3. High Risk Indicator (Month-to-month + Fiber optic + No Tech Support)
    is_high_risk = (df['Contract'] == 'Month-to-month') & (df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 'No')
    df['IsHighRisk'] = is_high_risk.astype(int)
    
    return df

def get_preprocessor(df):
    """Build preprocessing pipeline."""
    # We will exclude Target, ID
    numeric_features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['Gender', 'Partner', 'InternetService', 'TechSupport', 'Contract', 'TenureGroup', 'SpendingCategory']
    passthrough_features = ['SeniorCitizen', 'IsHighRisk']
    
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
