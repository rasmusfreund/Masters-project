import numpy as np
import pandas as pd
from static_frame import FrameGO
import csv
import cupy as cp


def compute_beta(A, y, weights, lambda_reg):
    # Convert to CuPy array for GPU power(!)
    A = cp.asarray(A)
    y = cp.asarray(y)
    weights = cp.asarray(weights)

    # Apply weights to A and y
    weighted_A = A * weights[:, cp.newaxis]
    weighted_y = y * weights

    # Compute the normal equation
    A_dot_W_dot_A_T = cp.dot(weighted_A.T, A)
    
    # Add regularization
    n_features = A.shape[1]
    A_dot_W_dot_A_T += lambda_reg * cp.eye(n_features)

    A_dot_W_dot_y = cp.dot(weighted_A.T, weighted_y)

    # Solve linear system
    beta = cp.linalg.solve(A_dot_W_dot_A_T, A_dot_W_dot_y)
    return beta


def weighted_linear_regression(y, x, frac, xvals=None, lambda_reg=0):
    # x, y are data points
    # frac is the fraction of data points used to evaluate each local regression
    # xvals can be set to allow for an array of x-values for which to predict smoothed y-values
    # lambda_reg is the penalty term for L2 regularization

    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)

    # If xvals are not provided, use x for prediction
    if xvals is None:
        xvals_gpu = x_gpu
    else:
        xvals_gpu = cp.asarray(xvals)

    n = len(x_gpu)
    k = int(frac*n) # Number of points to include in each local regression
    y_smoothed_gpu = cp.zeros_like(xvals_gpu)

    # Construct design matrix for linear fit
    A_gpu = cp.vstack([cp.ones_like(x_gpu), x_gpu]).T

    for i in range(len(xvals_gpu)):
        # Calculate distances from the current point
        distances_gpu =  cp.abs(x_gpu - xvals_gpu[i])

        # Determine the kth nearest distance
        cutoff_distance_gpu = cp.partition(distances_gpu, k)[k]

        # Scale distances by the cutoff distance
        scaled_distances_gpu = distances_gpu / cutoff_distance_gpu

        # Apply tricube weight function
        weights_gpu = (1 - cp.clip(scaled_distances_gpu, 0, 1) ** 3) ** 3

        # Solve the weighted linear regression problem
        # beta = int(A.T @ W @ A) @ A.T @ W @ y
        beta_gpu = compute_beta(A_gpu, y_gpu, weights_gpu, lambda_reg)

        y_smoothed_gpu[i] = beta_gpu[0] + beta_gpu[1] * xvals_gpu[i]

    # Transfer only the final result back to CPU
    y_smoothed = cp.asnumpy(y_smoothed_gpu)
    return y_smoothed


def process_file(input_path, output_path):
    # Read input file, skipping first 3 lines
    df = pd.read_csv(input_path, delim_whitespace=True, skiprows=3, names=['mass', 'int'])
    
    # Convert to StaticFrame for immutability and performance (?!)
    sf = FrameGO.from_pandas(df)

    # Log-transform
    sf = sf.assign['int'](np.log(sf['int'].values + 1))

    # First LOWESS
    smoothed = weighted_linear_regression(sf['int'].values, sf['mass'].values, frac=0.0008, lambda_reg=40)
    sf = sf.assign['int'](smoothed)
    
    # Second LOWESS
    smoothed = weighted_linear_regression(sf['int'].values, sf['mass'].values, frac=0.3, lambda_reg=40)
    normalized_int = sf['int'].values / smoothed
    sf = sf.assign['int'](normalized_int)
    
    # Rescale int column
    int_min, int_max = sf['int'].min(), sf['int'].max()
    sf = sf.assign['int'](2 * (sf['int'] - int_min) / (int_max - int_min) - 1)

    # Map to new scale
    mass_new = np.arange(2000, 20001, 0.5)
    smoothed = weighted_linear_regression(sf['int'].values, sf['mass'].values, frac=0.0003, xvals=mass_new, lambda_reg=40)

    # New StaticFrame with smoothed data
    sf_final = FrameGO.from_dict({'mass':mass_new, 'int': smoothed})

    # Write to CSV
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['int', 'mass'])
        for intensity, mass in zip(sf_final['int'].values, sf_final['mass'].values):
            writer.writerow([intensity, mass])
            
            
input_path = 'ff3751e9-a2c0-4902-9d8d-43dd5f7b26a6.txt'
output_path = 'Python_ff3751e9-a2c0-4902-9d8d-43dd5f7b26a6.txt'
process_file(input_path, output_path)