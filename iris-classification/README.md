# 🌸 Iris Classification Project

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## Project Overview
This project is an end-to-end Machine Learning pipeline to classify Iris flower species based on their morphological characteristics. It was built as a professional, beginner-friendly internship project. The workflow encompasses Data Exploration (EDA), Preprocessing, Model Training, Evaluation, and Inference.

## Dataset Information
The dataset used is the classic **Iris Dataset**. It contains 150 samples across 3 species:
- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`

**Features:**
1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

## Technologies Used
- **Python**: Core programming language.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **Matplotlib & Seaborn**: High-quality data visualization.
- **Scikit-Learn**: Machine learning modeling and evaluation.
- **Joblib**: Model saving and loading.
- **Jupyter Notebook**: Interactive data exploration.

## Project Structure
```
iris-classification/
│
├── data/
│   └── Iris.csv                   # The dataset
│
├── notebooks/
│   └── iris_classification.ipynb  # Interactive EDA & Modeling notebook
│
├── models/
│   └── best_model.pkl             # Saved best model
│
├── outputs/
│   ├── plots/                     # Saved EDA and Evaluation plots
│   └── reports/                   # Classification text reports
│
├── src/
│   ├── train.py                   # Script to train and evaluate models
│   ├── predict.py                 # Inference script for predictions
│   └── utils.py                   # Helper functions
│
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
└── .gitignore                     # Git ignored files
```

## Installation Steps
1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd iris-classification
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\\Scripts\\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run Notebook
To explore the data, view the visualizations, and see the step-by-step modeling process, open the Jupyter Notebook:
```bash
jupyter notebook notebooks/iris_classification.ipynb
```

## How to Run Prediction Script
The `predict.py` script allows you to make predictions on new data via the command line.

**Example command:**
```bash
cd src
python predict.py --features 5.1 3.5 1.4 0.2
```

## Model Performance
Three models were evaluated: **Logistic Regression**, **Decision Tree**, and **k-NN (Tuned)**. All models performed excellently on the clean dataset, achieving over 93% accuracy on the test set. 

**Logistic Regression** was selected as the best performing model based on accuracy and stability. A full performance comparison chart and confusion matrices can be found in `outputs/plots/`.

## Example Prediction
```bash
Input features: [[5.1 3.5 1.4 0.2]]

Predicted Species: Iris-setosa
```

## Future Improvements
- **Web App Integration**: Wrap the inference script in a Flask or FastAPI backend to serve predictions over an API.
- **Deep Learning**: Experiment with simple PyTorch/TensorFlow Neural Networks for educational purposes.
- **Advanced Ensembles**: Use Random Forests or Gradient Boosting classifiers.
- **CI/CD Pipeline**: Add GitHub Actions for automated testing and linting.
