import os
from typing import List, Dict, NoReturn
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, csv_file_path: str | list[str], data_dir_path: str | list[str], one_hot_encode: bool = True, num_workers: int = 5):
        """
        Initializes the DataLoader with paths to the csv file and the directory containing the actual data files.

        Parameters:
        csv_file_path: File path (or list of paths) to the CSV file(s) containing the references and labels.
        data_dir_path: Directory path (or list of paths) that contains the actual data files.
        one_hot_encode: True / False statement on whether to one-hot encode labels or leave them as is
        num_workers: Integer defining how many CPU cores are available for the data generator
        """
        
        self.num_workers = num_workers
        
        # Control whether to one-hot encode labels
        self.one_hot_encode = one_hot_encode

        # Handle multiple CSV file paths 
        if isinstance(csv_file_path, list):
            self.data_references = pd.concat([pd.read_csv(path, low_memory=False) for path in csv_file_path], ignore_index=True)
        else:
            self.data_references = pd.read_csv(csv_file_path, low_memory=False)

        # Handle multiple data directories
        if isinstance(data_dir_path, list):
            if not isinstance(csv_file_path, list) or len(data_dir_path) != len(csv_file_path):
                raise ValueError('If data_dir_path is a list, it must match the length of csv_file_path.')
            self.data_dir_paths = data_dir_path
        else:
            self.data_dir_paths = [data_dir_path] * len(self.data_references)

        # Adjust file paths in data_references to include the correct directory
        self.data_references['file_path'] = self.data_references.apply(lambda x: self.assign_file_path(x['code']), axis=1)
        
        # Mapping for one-hot encoding of label-vector
        self.label_mapping = {label: idx for idx, label in enumerate(self.data_references['species'].unique())}
        self.num_labels = len(self.label_mapping)
        
    
    def assign_file_path(self, code):
        """
        Assigns the correct file path based on the code and directory paths.
        """

        for path in self.data_dir_paths:
            file_path = f'{path}/{code}.txt'
            if os.path.exists(file_path):
                return file_path
        raise FileNotFoundError(f'File {code}.txt not found in provided directories.')

    
    def load_data(self, index: int):
        """
        Loads a single data file based on the index provided and returns the contents and label.

        Parameters:
        index: The index of the data file to load, as referenced in the CSV file.

        Returns:
        data, label pair         
        """

        file_path = self.data_references.iloc[index]['file_path']
        label = self.data_references.iloc[index]['species']

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
        Generator function that yields one data-label pair at a time; parallelised.
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

    DATA_PATH = "/faststorage/project/amr_driams/rasmus/data/"
    ID_PATHS = []
    for root, dir, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith('clean.csv'):
                ID_PATHS.append(os.path.join(root, file))
                
    # Loader for full data set models
    paths_A = [os.path.split(path)[0] for path in sorted(ID_PATHS[:4])]
    years = ['2015', '2016', '2017', '2018']
    
    for i in range(len(paths_A)):
        paths_A[i] = os.path.join(paths_A[i], f'{years[i]}_clean_train_test.csv')
    data_A = [os.path.join('/faststorage/project/amr_driams/data/DRIAMS-A/binned_6000', year) for year in years]
    
    data_loader = DataLoader(csv_file_path=paths_A, 
                             data_dir_path= data_A,
                             one_hot_encode=True,
                             num_workers=5)
    total_data_points = len(data_loader.data_references)
    
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
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
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
    results_dir = '/faststorage/project/amr_driams/rasmus/data/DRIAMS-A/ml_models/aggregated' # Get the directory of the CSV file
    results_file_name = 'rf_best_params_full.txt'
    results_file_path = os.path.join(results_dir, results_file_name)
    with open(results_file_path, 'w') as file:
        file.write(f"Best Parameters: {best_params}\n")
        file.write(f"Best Score: {best_score}\n")

    # Extract and save the CV results
    save_path = os.path.join(results_dir, 'rf_cv_results_full.csv')
    save_cv_results(grid_search_rf.cv_results_, save_path)

if __name__ == "__main__":
    main()