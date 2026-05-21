import pandas as pd
from sklearn.datasets import load_iris

def create_kaggle_iris():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
    
    # Map target to names to match Kaggle
    species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    df['species'] = [species_map[target] for target in data.target]
    
    df.to_csv('Iris.csv', index=False)
    print("Iris.csv created successfully.")

if __name__ == "__main__":
    create_kaggle_iris()
