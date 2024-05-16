import os
from typing import List, Dict, NoReturn
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class DataLoader:
    def __init__(self, csv_file_path: str | list[str], data_dir_path: str | list[str], one_hot_encode: bool = True, training_target: str = 'species', num_workers: int = 5):
        """
        Initializes the DataLoader with paths to the csv file and the directory containing the actual data files.

        Parameters:
        csv_file_path: File path (or list of paths) to the CSV file(s) containing the references and labels.
        data_dir_path: Directory path (or list of paths) that contains the actual data files.
        one_hot_encode: True / False statement on whether to one-hot encode species labels
        training_target: Sets label output to either 'species' or 'resistance' depending on target of the model
        num_workers: Integer defining how many CPU cores are available for the data generator
        """
        self.csv_file_path = csv_file_path
        self.num_workers = num_workers
        
        # Control whether to one-hot encode labels
        self.one_hot_encode = one_hot_encode

        # Check target of the model
        if training_target not in ['species', 'resistance']:
            raise ValueError('Argument "training_target" should either be "species" or "resistance".')
        else:
            self.training_target = training_target
        
        # Get data references
        self.data_references = self._load_data_references(csv_file_path, data_dir_path)
        
        # Adjust file paths in data_references to include the correct directory
        self.data_references['file_path'] = self.data_references.apply(self.assign_file_path, axis=1)
            
        # Mapping for one-hot encoding of label-vectorx
        # Case: species
        if self.training_target == 'species':
            self.label_columns = ['species']
            self.label_mapping = {label: idx for idx, label in enumerate(self.data_references['species'].unique())}
        # Case: resistance
        else:
            self.label_columns = [col for col in self.data_references.columns if col not in ['code', 'species', 'source_csv', 'data_dir', 'file_path']]
        
        self.num_labels = len(self.data_references)

    def __len__(self):
        return len(self.data_references)
        
    
    def __getitem__(self, idx):
        data, label = self.load_data(idx)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    
    def _load_data_references(self, csv_file_path, data_dir_path):
        data_frames = []
        if isinstance(csv_file_path, list):
            for path, dir_path in zip(csv_file_path, data_dir_path):
                df = pd.read_csv(path, low_memory=False)
                df['source_csv'] = path
                df['data_dir'] = dir_path
                data_frames.append(df)
        else:
            df = pd.read_csv(csv_file_path, low_memory=False)
            df['source_csv'] = csv_file_path
            df['data_dir'] = data_dir_path
            data_frames.append(df)
        
        return pd.concat(data_frames, ignore_index=True)

    
    def assign_file_path(self, row):
        """
        Assigns the correct file path based on the code and directory paths.
        """
        code = row['code']
        data_dir = row['data_dir']
        file_path = f'{data_dir}/{code}.txt'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File {code}.txt not found in provided directories.')
        return file_path

    
    def load_data(self, idx: int):
        """
        Loads a single data file based on the index provided and returns the contents and label.
        If target of the model is resistances, (R)esistant and (I)ntermediate will both be marked by 1, indicating resistance.

        Parameters:
        index: The index of the data file to load, as referenced in the CSV file.

        Returns:
        data, label pair         
        """
        row = self.data_references.iloc[idx]
        data = self.convert_file_to_floats(row['file_path'])
        if not data:
            print(f'No loaded data from: {file_path}')
        label = None
        
        if self.training_target == 'resistance':
            label = row[self.label_columns].values.astype(int)
        else:        
            species_label = row['species']
            if self.one_hot_encode:
                label = np.zeros(self.num_labels)
                label[self.label_mapping[species_label]] = 1
            else:
                label = self.label_mapping[species_label]
            
        return data, label


    def convert_file_to_floats(self, file_path: str):
        data = [] # Container for converted data
        try:
            with open(file_path, 'r') as file:
                next(file) # Skip the header
                for line in file:
                    string_values = line.strip().split(' ')[1]
                    data.append(float(string_values))
        except Exception as e:
            print(f'Error processing file {file_path}: {e}')
        return data

    
    def one_hot_encode_label(self, label: str):
        one_hot_vector = np.zeros(self.num_labels)
        label_index = self.label_mapping[label]
        one_hot_vector[label_index] = 1
        
        return one_hot_vector


    def get_resistances_from_vector(self, k_hot_vector):
        if self.training_target != 'resistance':
            raise ValueError('This function is only applicable for resistance data.')
        active_indeces = [i for i, val in enumerate(k_hot_vector) if val == 1]
        active_resistances = [self.label_columns[i] for i in active_indeces]

        return active_resistances
    

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
    data_loader = DataLoader(csv_file_path='/faststorage/project/amr_driams/rasmus/data/DRIAMS-A/id/resistance/2016_clean_train_val.csv',
                             data_dir_path='/faststorage/project/amr_driams/data/DRIAMS-A/binned_6000/2016/',
                             training_target='resistance',
                             num_workers=1)

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
    
    def f1_macro_scorer(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro', zero_division=0)

    scorer = make_scorer(f1_macro_scorer)
    
    # Scale data for better linear modelling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeClassifier(random_state=42))
    ])
    
    # Define parameter grid for CV
    param_grid = {
        'ridge__alpha': [0.1, 1, 10, 100, 1000]
    }
    
    # Set up the grid search
    grid_search_ridge = GridSearchCV(estimator=pipeline,
                                     param_grid=param_grid,
                                     cv=5,
                                     scoring=scorer,
                                     n_jobs=-1,
                                     verbose=2)
    # Run fit
    grid_search_ridge.fit(X_train, y_train)
    
    # Best parameters and score
    best_params = grid_search_ridge.best_params_
    best_score = grid_search_ridge.best_score_
    
    # Save the best parameters and score to a text file
    results_dir = os.path.dirname(data_loader.csv_file_path)  # Get the directory of the CSV file
    results_file_name = "2016_ridge_resistance_best_params.txt"
    results_file_path = os.path.join(results_dir, results_file_name)
    with open(results_file_path, 'w') as file:
        file.write(f"Best Parameters: {best_params}\n")
        file.write(f"Best Score: {best_score}\n")
    
    # Extract and save the CV results
    save_path = os.path.join(results_dir, '2016_ridge_resistance_cv_results.csv')
    save_cv_results(grid_search_ridge.cv_results_, save_path)
    print(f"Results saved to {results_file_path}")
    
    # Print best parameters and score
    print("Best Parameters:", grid_search_ridge.best_params_)
    print("Best Score:", grid_search_ridge.best_score_)
    print("Best Model:", grid_search_ridge.best_estimator_)

if __name__ == "__main__":
    main()