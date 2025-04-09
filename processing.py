import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

# combine bootstrap resuts into a single dictionary
def combine_results(results_list):
    """
    Combines a list of result dictionaries into a single dictionary with lists of values.

    Parameters:
    - results_list: list of dictionaries
        A list where each element is a dictionary of results.

    Returns:
    - results: dictionary
        A dictionary where each key corresponds to a list of values from all dictionaries in the results_list.
    """
    # Initialize a defaultdict to hold the combined results
    results = defaultdict(list)
    
    # Loop through each dictionary in the results_list
    for i in results_list:
        for key, value in i.items():
            # Append each value to the corresponding list in the defaultdict
            results[key].append(value)
    
    # Convert the defaultdict back to a regular dictionary
    results = dict(results)
    
    return results

# modification with procrustes rotation:
def post_process(results, sample_id, data):
    
    # Use the first bootstrap result as the reference
    reference_loadings = results['loadings'][0]
    variables = data.columns  # Assuming 'data' has columns corresponding to variables
    
    # Iterate through each bootstrap sample
    for idx in range(1, len(results['loadings'])):
        current_loadings = results['loadings'][idx]
        current_scores = results['scores'][idx]
        
        # Ensure that the shapes match
        if current_loadings.shape != reference_loadings.shape:
            # Adjust matrices to have the same shape if necessary
            # For this example, we'll assume they have the same shape
            pass
        
        # Perform Procrustes rotation to align current loadings to the reference
        R, scale = orthogonal_procrustes(current_loadings, reference_loadings)
        aligned_loadings = current_loadings @ R
        
        # Apply the rotation to the factor scores as well
        aligned_scores = current_scores @ R
        
        # Update the results with the aligned loadings and scores
        results['loadings'][idx] = aligned_loadings
        results['scores'][idx] = aligned_scores
    
    # After alignment, proceed to compute averages and variability
    # Stack the aligned loadings
    loadings_array = np.stack(results['loadings'])
    results['loadings_ave'] = np.mean(loadings_array, axis=0)
    results['loadings_2sd'] = 2 * np.std(loadings_array, axis=0)
    
    # Calculate average uniquenesses
    uniquenesses_array = np.stack(results['uniquenesses'])
    results['uniquenesses_ave'] = np.mean(uniquenesses_array, axis=0)
    results['uniquenesses_2sd'] = 2 * np.std(uniquenesses_array, axis=0)
    
    # Ensure sum_squared_loadings and proportion_variance are correctly shaped
    for idx in range(len(results['sum_squared_loadings'])):
        results['sum_squared_loadings'][idx] = results['sum_squared_loadings'][idx].reshape(1, -1)
        results['proportion_variance'][idx] = results['proportion_variance'][idx].reshape(1, -1)
    
    # Compute averages and variability
    sum_squared_loadings_array = np.vstack(results['sum_squared_loadings'])
    results['sum_squared_loadings_ave'] = np.mean(sum_squared_loadings_array, axis=0)
    results['sum_squared_loadings_2sd'] = 2 * np.std(sum_squared_loadings_array, axis=0)
    
    proportion_variance_array = np.vstack(results['proportion_variance'])
    results['proportion_variance_ave'] = np.mean(proportion_variance_array, axis=0)
    results['proportion_variance_2sd'] = 2 * np.std(proportion_variance_array, axis=0)
    
    # Calculate average score value, using original sample IDs for sorting (indices)
    scores_df_list = []
    for arr, idx in zip(results['scores'], sample_id):
        df = pd.DataFrame(arr, index=idx)
        scores_df_list.append(df)
    scores_df = pd.concat(scores_df_list)
    
    # Group by index and calculate average
    scores_per_sample = scores_df.groupby(scores_df.index)
    scores_ave = scores_per_sample.mean()
    
    # Reorder to match the original input data (for plotting and sample ID consistency)
    scores_ave = scores_ave.reindex(data.index)
    
    # Reconvert to array and add to results dictionary
    scores_ave_array = scores_ave.to_numpy()
    results['scores_ave'] = scores_ave_array
    results['scores_per_sample'] = scores_per_sample
    
    # Return post-processed, final results
    return results

# manually swap indices in results, to ensure factors are ordered by explained variance
def swap_indices(data_dict, swap_indices=(1, 2), skip_keys=None):
    """
    Swaps specified indices in the values of a dictionary based on their dimensionality.
    
    Parameters:
    - data_dict (dict): The dictionary containing the data to modify.
    - swap_indices (tuple): A tuple of two indices to swap. Default is (1, 2).
    - skip_keys (list): A list of keys to skip. Default is ["uniquenesses", "uniquenesses_ave", "uniquenesses_2sd"].
    
    Returns:
    - None: The function modifies the dictionary in place.
    """
    if skip_keys is None:
        skip_keys = ["uniquenesses", "uniquenesses_ave", "uniquenesses_2sd"]
    
    idx1, idx2 = swap_indices
    
    for key, value in data_dict.items():
        if key in skip_keys:
            continue
        
        try:
            if isinstance(value, np.ndarray):
                if value.ndim == 3:
                    # Swap along the last axis for 3D arrays
                    value[..., [idx1, idx2]] = value[..., [idx2, idx1]]
                elif value.ndim == 2:
                    # Swap along the second axis for 2D arrays
                    value[:, [idx1, idx2]] = value[:, [idx2, idx1]]
                elif value.ndim == 1:
                    # Swap elements in 1D arrays
                    if len(value) > max(idx1, idx2):
                        value[idx1], value[idx2] = value[idx2], value[idx1]
                else:
                    print(f"Unsupported ndarray dimensions for key: {key}")
            
            elif isinstance(value, list):
                # Determine the depth of the list
                if all(isinstance(elem, list) for elem in value):
                    # Assume 2D list
                    for sublist in value:
                        if len(sublist) > max(idx1, idx2):
                            sublist[idx1], sublist[idx2] = sublist[idx2], sublist[idx1]
                else:
                    # Assume 1D list
                    if len(value) > max(idx1, idx2):
                        value[idx1], value[idx2] = value[idx2], value[idx1]
            
            elif isinstance(value, pd.core.groupby.generic.DataFrameGroupBy):
                # Skipping pandas DataFrameGroupBy objects
                continue
            
            else:
                print(f"Unsupported type for key: {key}, type: {type(value)}")
        
        except Exception as e:
            print(f"Error processing key: {key}. Error: {e}")