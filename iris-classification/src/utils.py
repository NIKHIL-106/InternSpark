import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import joblib
import os

def load_data(filepath):
    """Load the dataset from the given filepath."""
    return pd.read_csv(filepath)

def preprocess_data(df, target_col='species', test_size=0.2, random_state=42):
    """Preprocess data: feature-target split, train-test split, scaling, encoding."""
    X = df.drop(columns=[target_col])
    if 'Id' in X.columns:
        X = X.drop(columns=['Id'])
    y = df[target_col]
    
    # Label encoding for target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

def evaluate_model(model, X_test, y_test, model_name, class_names, output_dir):
    """Evaluate model and save confusion matrix plot."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-score': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.replace(" ", "_").lower()}_confusion_matrix.png'))
    plt.close()
    
    # Save classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    with open(os.path.join(output_dir, f'{model_name.replace(" ", "_").lower()}_report.txt'), 'w') as f:
        f.write(f"Classification Report for {model_name}\n")
        f.write("="*50 + "\n")
        f.write(report)
        
    return metrics, report

def plot_model_comparison(metrics_df, output_dir):
    """Plot bar chart comparing model performances."""
    plt.figure(figsize=(10, 6))
    metrics_df.set_index('Model').plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()

def save_model(model, scaler, le, filepath):
    """Save the best model, scaler, and label encoder."""
    joblib.dump({'model': model, 'scaler': scaler, 'le': le}, filepath)

def load_model(filepath):
    """Load a saved model pipeline."""
    return joblib.load(filepath)
