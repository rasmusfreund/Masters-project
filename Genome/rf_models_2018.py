import os
from typing import List, Dict, NoReturn
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class DataLoader:
    def __init__(self, csv_file_path: str, data_dir_path: str, one_hot_encode: bool = True, num_workers: int = 5):
        """
        Initializes the DataLoader with paths to the csv file and the directory containing the actual data files.

        Parameters:
        csv_file_path: File path to the CSV file containing the references and labels.
        data_dir_path: Directory path that contains the actual data files.
        """
        self.csv_file_path = csv_file_path
        self.data_dir_path = data_dir_path
        self.data_references = pd.read_csv(csv_file_path)
        self.num_workers = num_workers

        # Control whether to one-hot encode labels
        self.one_hot_encode = one_hot_encode
        
        # Mapping for one-hot encoding of label-vector
        self.label_mapping = {label: idx for idx, label in enumerate(self.data_references['species'].unique())}
        self.num_labels = len(self.label_mapping)
        

    def load_data(self, index: int):
        """
        Loads a single data file based on the index provided and returns the contents and label.

        Parameters:
        index: The index of the data file to load, as referenced in the CSV file.

        Returns:
        data, label pair         
        """

        file_reference = self.data_references.iloc[index]['code']
        label = self.data_references.iloc[index]['species']
        file_path = f"{self.data_dir_path}/{file_reference}.txt"

        data = self.convert_file_to_floats(file_path)
        label = self.label_mapping[label] if not self.one_hot_encode else self.one_hot_encode_label(label)
        
        return data, label


    def convert_file_to_floats(self, file_path: str):
        data = [] # Container for converted data

        with open(file_path, 'r') as file:
            next(file) # Skip the header
            for line in file:
                string_values = line.strip().split(' ')
                string_values = [value for value in string_values if value]
                for value in string_values[1:]:
                    data.append(float(value))
        return data

    
    def one_hot_encode_label(self, label: str):
        one_hot_vector = np.zeros(self.num_labels)
        label_index = self.label_mapping[label]
        one_hot_vector[label_index] = 1
        return one_hot_vector

    
    def data_generator(self):
        """
        Generator function that yields one data-label pair at a time.
        """
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_idx = {executor.submit(self.load_data, idx): idx for idx in range(len(self.data_references))}
            for future in as_completed(future_to_idx):
                yield future.result()
                

def save_cv_results(cv_results, save_path):
    """
    Saves the mean test score and standard deviation from cross-validation results to a CSV-file.

    Parameters:
    cv_results: The cv_results_ attribute from a GridSearchCV or RandomizedSearchCV object.
    save_path: Path to the file where the CV results will be saved.
    """
    # Select relevant information
    results_to_save = {
        'params': cv_results['params'],
        'mean_test_score': cv_results['mean_test_score'],
        'std_test_score': cv_results['std_test_score']
    }

    # Convert to DataFrame
    results_df = pd.DataFrame(results_to_save)

    # Save to CSV
    results_df.to_csv(save_path, index=False)
    print(f"CV results saved to {save_path}")


def main():
    data_loader = DataLoader(csv_file_path='/faststorage/project/amr_driams/rasmus/data/DRIAMS-A/id/2018_clean_train_test.csv',
                         data_dir_path='/faststorage/project/amr_driams/data/DRIAMS-A/binned_6000/2018/',
                        num_workers=10)


    total_data_points = len(data_loader.data_references)

    # Load dataset
    X, y = [], []
    for data, label in tqdm(data_loader.data_generator(), total=total_data_points, desc="Loading Data"):
        X.append(data)
        y.append(label)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42
    )
    
    # Initialise model
    rf = RandomForestClassifier(random_state=42)

    # Define parameter grid for CV
    param_grid = {
        'n_estimators': [250, 300],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
        'max_features': ['sqrt']
    }

    # Set up grid search
    grid_search_rf = GridSearchCV(estimator=rf,
                            param_grid=param_grid,
                            cv=5,
                            n_jobs=-1,
                            verbose=2)

    # Run fit
    grid_search_rf.fit(X_train, y_train)

    # Best parameters and score
    best_params = grid_search_rf.best_params_
    best_score = grid_search_rf.best_score_

    # Save the best parameters and score to a text file
    results_dir = os.path.dirname(data_loader.csv_file_path)  # Get the directory of the CSV file
    results_file_name = "rf_best_params_2018.txt"
    results_file_path = os.path.join(results_dir, results_file_name)
    with open(results_file_path, 'w') as file:
        file.write(f"Best Parameters: {best_params}\n")
        file.write(f"Best Score: {best_score}\n")

    # Extract and save the CV results
    save_path = os.path.join(results_dir, 'rf_cv_results_2018.csv')
    save_cv_results(grid_search_rf.cv_results_, save_path)

if __name__ == "__main__":
    main()