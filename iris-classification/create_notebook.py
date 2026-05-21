import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # 1. Introduction
    cells.append(nbf.v4.new_markdown_cell("""# Iris Classification Project 🌸

## Introduction Section
Welcome to the **Iris Classification Project**. This project is designed as an end-to-end Machine Learning pipeline to classify Iris flower species based on their morphological characteristics.

### The Iris Dataset
The Iris dataset is a classic and widely used dataset in the machine learning community. It contains 150 samples from three species of Iris flowers:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

For each sample, four features were measured:
1. **Sepal Length** (in cm)
2. **Sepal Width** (in cm)
3. **Petal Length** (in cm)
4. **Petal Width** (in cm)

### The Classification Problem
This is a **Multiclass Classification** problem. Our goal is to map the input features (the dimensions of the petals and sepals) to discrete output categories (the three species). By training a machine learning model on historical data where the species are known, the model will learn to distinguish the patterns that define each species.

### Objective
The main objectives of this project are:
- To perform comprehensive Exploratory Data Analysis (EDA) to understand feature separability and patterns.
- To preprocess the data correctly using feature scaling and label encoding.
- To train, evaluate, and compare multiple classification algorithms (k-NN, Logistic Regression, Decision Tree).
- To select and save the best performing model for future inference.
"""))
    
    # 2. Data Loading & Exploration
    cells.append(nbf.v4.new_markdown_cell("""## Data Loading & Exploration

In this section, we import the necessary libraries, load the dataset, and perform basic initial inspections like checking the data shape, data types, missing values, and class distribution.
"""))
    
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set professional visualization styles
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the dataset
df = pd.read_csv('../data/Iris.csv')

# Display first 5 rows
df.head()"""))

    cells.append(nbf.v4.new_code_cell("""# Show data shape
print(f"Dataset Shape: {df.shape}")

# Show data types
print("\\nData Types:\\n", df.dtypes)

# Check missing values
print("\\nMissing Values:\\n", df.isnull().sum())

# Display summary statistics
display(df.describe())

# Show class distribution
print("\\nClass Distribution:\\n", df['species'].value_counts())
"""))

    # 3. Exploratory Data Analysis (EDA)
    cells.append(nbf.v4.new_markdown_cell("""## Exploratory Data Analysis (EDA)

EDA allows us to discover patterns, spot anomalies, and understand feature relationships. We will use `matplotlib` and `seaborn` to create high-quality visualizations.

**Key things to observe:**
- **Class Separability**: Are the different species distinguishable based on these features?
- **Feature Importance Intuition**: Which features seem most useful for separating the classes?
- **Patterns**: We'll likely see that *Iris-setosa* is easily linearly separable from the other two based on petal dimensions.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Pairplot colored by species
plt.figure(figsize=(10, 8))
sns.pairplot(df.drop(columns=['Id'], errors='ignore'), hue='species', palette='Dark2', diag_kind='kde')
plt.suptitle('Pairplot of Iris Features', y=1.02, fontsize=16)
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 2. Correlation heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.drop(columns=['Id', 'species'], errors='ignore')
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 3. Histograms and 4. Boxplots
features = numeric_df.columns
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feature Distributions Comparison', fontsize=16)

for i, feature in enumerate(features):
    row, col = i // 2, i % 2
    sns.boxplot(x='species', y=feature, data=df, ax=axes[row, col], palette='Set2')
    axes[row, col].set_title(f'{feature} by Species')
    
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 5. Scatter plot of Petal dimensions
plt.figure(figsize=(10, 6))
sns.scatterplot(x='petal length', y='petal width', hue='species', data=df, s=100, palette='Dark2', edgecolor='w')
plt.title('Petal Length vs Petal Width', fontsize=14)
plt.show()"""))

    # 4. Data Preprocessing
    cells.append(nbf.v4.new_markdown_cell("""## Data Preprocessing

Before feeding data to our machine learning models, we need to prepare it:
1. **Feature Selection**: Separate the features (X) from the target label (y). We also drop the 'Id' column.
2. **Label Encoding**: Convert target string labels into numeric formats (0, 1, 2) since models require numerical inputs.
3. **Train-Test Split**: We'll hold out 20% of the data to evaluate our model's performance on unseen data.
4. **Feature Scaling**: Algorithms like k-NN are distance-based and require features to be on the same scale. We use `StandardScaler`.
"""))

    cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Feature Selection
X = df.drop(columns=['Id', 'species'], errors='ignore')
y = df['species']

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Train-Test Split (with stratification to preserve class distributions)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train shape: {X_train_scaled.shape}")
print(f"X_test shape: {X_test_scaled.shape}")
"""))

    # 5. Model Training
    cells.append(nbf.v4.new_markdown_cell("""## Model Training

We will train and compare three distinct classification algorithms:
1. **k-Nearest Neighbors (k-NN)**: A non-parametric method used for classification based on distance.
2. **Logistic Regression**: A linear model for classification.
3. **Decision Tree Classifier**: A tree-like model of decisions that captures non-linear relationships.

We also perform **Hyperparameter Tuning** for k-NN using GridSearchCV to find the optimal number of neighbors.
"""))

    cells.append(nbf.v4.new_code_cell("""from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# 1. Hyperparameter Tuning for k-NN
knn_param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring='accuracy')
knn_grid.fit(X_train_scaled, y_train)

print(f"Best k-NN Parameters: {knn_grid.best_params_}")
best_knn = knn_grid.best_estimator_

# 2. Initialize Models
models = {
    'k-NN (Tuned)': best_knn,
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=200),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# 3. Train all models
for name, model in models.items():
    if name != 'k-NN (Tuned)': # Already trained during GridSearch
        model.fit(X_train_scaled, y_train)
    print(f"{name} trained successfully.")
"""))

    # 6. Evaluation Metrics
    cells.append(nbf.v4.new_markdown_cell("""## Evaluation Metrics

We evaluate our trained models using various metrics:
- **Accuracy Score**: Overall correctness.
- **Precision, Recall, F1-Score**: To understand class-specific performance.
- **Confusion Matrix**: Visualizing true vs predicted classifications.
"""))

    cells.append(nbf.v4.new_code_cell("""from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

results = []

for name, model in models.items():
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1
    })
    
    print(f"\\n{'='*40}\\n{name} Evaluation\\n{'='*40}")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Feature Importance for Decision Tree
Let's see which features the Decision Tree relied on the most.
"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(8, 5))
importance = models['Decision Tree'].feature_importances_
sns.barplot(x=importance, y=X.columns, palette='viridis')
plt.title('Feature Importance (Decision Tree)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Model Comparison"""))
    
    cells.append(nbf.v4.new_code_cell("""# Compile results and display
results_df = pd.DataFrame(results)
display(results_df)

# Plot Model Comparison Bar Chart
results_df.set_index('Model').plot(kind='bar', figsize=(10, 6), colormap='Set2')
plt.title('Model Performance Comparison', fontsize=16)
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
"""))

    # 7. Model Saving
    cells.append(nbf.v4.new_markdown_cell("""## Model Saving

We identify the best performing model based on accuracy and save it, along with the scaler and label encoder, so it can be deployed or used in inference scripts.
"""))

    cells.append(nbf.v4.new_code_cell("""import joblib
import os

# Identify best model
best_model_name = results_df.loc[results_df['Accuracy'].idxmax()]['Model']
best_model = models[best_model_name]

print(f"Best Performing Model: {best_model_name}")

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Save the model pipeline (model, scaler, label encoder)
pipeline = {
    'model': best_model,
    'scaler': scaler,
    'le': le
}

model_path = '../models/best_model.pkl'
joblib.dump(pipeline, model_path)
print(f"Model successfully saved to {model_path}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Conclusion
In this project, we successfully built an end-to-end machine learning pipeline for classifying Iris flowers.
1. **Insights from EDA**: We observed that *Iris-setosa* is easily distinguishable from the other two species, particularly based on petal dimensions. Petal length and petal width are highly correlated and serve as strong predictors.
2. **Model Comparison**: Most models performed exceptionally well on this clean dataset. The simple algorithms were sufficient, and hyperparameter tuning slightly stabilized k-NN.
3. **Future Scope**: For future improvements, we could train deep learning models, deploy the `predict.py` script as a Flask or FastAPI web endpoint, or experiment with more complex ensembles. The current performance is extremely solid for deployment.
"""))

    nb['cells'] = cells
    
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/iris_classification.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook generated.")

if __name__ == "__main__":
    create_notebook()
