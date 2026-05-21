import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import shap

def evaluate_model(model, X_test, y_test, model_name, plot_dir):
    """Evaluates a classification model and saves plots."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, pos_label='Yes', zero_division=0),
        'Recall': recall_score(y_test, y_pred, pos_label='Yes', zero_division=0),
        'F1-score': f1_score(y_test, y_pred, pos_label='Yes', zero_division=0),
        'ROC-AUC': roc_auc_score(y_test.map({'No':0, 'Yes':1}), y_proba) if y_proba is not None else np.nan
    }
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['No', 'Yes'])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Churned'], yticklabels=['Stayed', 'Churned'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{model_name.replace(" ", "_")}_cm.png'))
    plt.close()
    
    # 2. ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test.map({'No':0, 'Yes':1}), y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {metrics["ROC-AUC"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title(f'ROC Curve: {model_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{model_name.replace(" ", "_")}_roc.png'))
        plt.close()
        
    return metrics

def plot_model_comparison(metrics_df, plot_dir):
    """Plot model comparison charts."""
    plt.figure(figsize=(10, 6))
    metrics_df.set_index('Model')[['Accuracy', 'ROC-AUC', 'F1-score']].plot(kind='bar', figsize=(12, 6), colormap='Set2')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'model_comparison.png'))
    plt.close()

def generate_shap_explainer(model_pipeline, X_sample, plot_dir, num_feats, cat_feats, pass_feats):
    """Generate SHAP feature importance plot."""
    try:
        model = model_pipeline.named_steps['model']
        preprocessor = model_pipeline.named_steps['preprocessor']
        
        X_trans = preprocessor.transform(X_sample)
        
        # Determine feature names
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_names = cat_encoder.get_feature_names_out(cat_feats)
        all_features = num_feats + list(cat_names) + pass_feats
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)
        
        # For classification, take positive class
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_trans, feature_names=all_features, show=False)
        plt.title('SHAP Feature Importance (Drivers of Churn)')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'shap_summary.png'))
        plt.close()
    except Exception as e:
        print(f"SHAP failed: {e}")
