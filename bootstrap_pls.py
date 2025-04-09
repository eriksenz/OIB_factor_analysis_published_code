import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- define helper functions ---

# bootstrap resampling function (individual resample)
def bootstrap(data, n=40):
    """
    Parameters:
    - data: DataFrame including a 'Group' column.
    - n: Maximum number of samples to draw per group.
    """
    sampled_data = []
    groups = data['Group'].unique()
    for grp in groups:
        grp_data = data[data['Group'] == grp]
        n_samples = min(len(grp_data), n)
        sampled_grp_data = grp_data.sample(n=n_samples, replace=True, random_state=None)
        sampled_data.append(sampled_grp_data)
    return pd.concat(sampled_data)
    
# vip calculation function
def calculate_vip(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(-1, 1)
    total_s = np.sum(s)
    vip = np.zeros((p,))
    for j in range(p):
        weight = np.array([(w[j, k] ** 2) * s[k] for k in range(h)])
        vip[j] = np.sqrt(p * np.sum(weight) / total_s)
    return vip

# --- define bootstrap pls function ---

def pls_bootstrap(predictors, response, n_bootstraps=500):
    # Dictionary to store final results for each factor
    pls_bootstrap_results = {}
    # Use the response dataframe's columns as the factors to analyze
    factor_names = response.columns

    # Loop over each factor
    for factor_name in factor_names:
        print(f"Processing {factor_name}...")
        r2_scores = []
        mse_scores = []
        vip_scores_all_bootstraps = []
        predictions_per_sample = {}
        # List to collect best number of components selected in each fold
        n_components_all = []

        # Loop over the number of bootstrap iterations
        for i in range(n_bootstraps):
            # Bootstrap sampling
            boot_data = bootstrap(predictors, n=15)
            # Extract predictors (dropping 'Group') and align responses
            X_boot = boot_data.drop(columns=['Group']).values
            y_boot = response.loc[boot_data.index, factor_name].values
            sample_indices_boot = boot_data.index
            groups_boot = predictors.loc[boot_data.index, 'Group'].values

            # 5-fold cross-validation for each bootstrap sample
            kf = KFold(n_splits=5, shuffle=True, random_state=i)
            for train_index, test_index in kf.split(X_boot):
                X_train, X_test = X_boot[train_index], X_boot[test_index]
                y_train, y_test = y_boot[train_index], y_boot[test_index]
                sample_indices_test = sample_indices_boot[test_index]
                groups_test = groups_boot[test_index]

                # Scale predictors based on training data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Scale response variable based on training data
                y_scaler = StandardScaler()
                y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
                y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

                # Set up the PLS regression model in a pipeline for hyperparameter tuning
                pipeline = Pipeline([('pls', PLSRegression())])
                param_grid = {'pls__n_components': list(range(1, min(10, X_train_scaled.shape[1]) + 1))}
                inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
                grid_search = GridSearchCV(estimator=pipeline,
                                           param_grid=param_grid,
                                           cv=inner_cv,
                                           scoring='neg_mean_squared_error',
                                           n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train_scaled)
                best_model = grid_search.best_estimator_

                # Record the best number of components for this fold
                best_n_components = best_model.named_steps['pls'].n_components
                n_components_all.append(best_n_components)

                # Predict on the test set and convert back to the original scale
                y_pred_scaled = best_model.predict(X_test_scaled).ravel()
                y_test_original = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
                y_pred_original = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

                # Collect predictions per sample
                for idx, sample_idx in enumerate(sample_indices_test):
                    if sample_idx not in predictions_per_sample:
                        predictions_per_sample[sample_idx] = {
                            'y_true': y_test_original[idx],
                            'y_pred_list': [y_pred_original[idx]],
                            'group': groups_test[idx]
                        }
                    else:
                        predictions_per_sample[sample_idx]['y_pred_list'].append(y_pred_original[idx])

                # Calculate and store R² and MSE for this fold
                r2_iter = r2_score(y_test_original, y_pred_original)
                mse_iter = mean_squared_error(y_test_original, y_pred_original)
                r2_scores.append(r2_iter)
                mse_scores.append(mse_iter)

                # Calculate VIP scores for the current model
                pls_model = best_model.named_steps['pls']
                vip_scores = calculate_vip(pls_model)
                vip_scores_all_bootstraps.append(vip_scores)

        # Aggregate VIP scores over all bootstraps
        vip_scores_all_bootstraps = np.array(vip_scores_all_bootstraps)
        vip_scores_mean = np.mean(vip_scores_all_bootstraps, axis=0)
        vip_scores_std = np.std(vip_scores_all_bootstraps, axis=0)

        # Calculate overall R² and MSE statistics for the factor
        r2_mean = np.mean(r2_scores)
        mse_mean = np.mean(mse_scores)
        r2_std = np.std(r2_scores)
        mse_std = np.std(mse_scores)

        print(f"Average R-squared for {factor_name}: {r2_mean:.2f} ± {r2_std:.2f}")

        # Aggregate the best number of components over all folds
        n_components_all = np.array(n_components_all)
        n_components_mean = np.mean(n_components_all)
        n_components_std = np.std(n_components_all)
        # For the mode, we'll use pandas mode
        n_components_mode = pd.Series(n_components_all).mode().iloc[0]

        # Create a DataFrame for VIP scores (using predictor names, excluding 'Group')
        vip_df = pd.DataFrame({
            'VIP': vip_scores_mean,
            'STD': vip_scores_std
        }, index=predictors.drop(columns=['Group']).columns)

        # Create a DataFrame for the average predictions per sample
        sample_data = []
        for sample_id, data_dict in predictions_per_sample.items():
            sample_data.append({
                'SampleIndex': sample_id,
                'y_true': data_dict['y_true'],
                'y_pred': np.mean(data_dict['y_pred_list']),
                'Group': data_dict['group']
            })
        results_df = pd.DataFrame(sample_data)

        # Store the results for this factor, including n_components information
        pls_bootstrap_results[factor_name] = {
            'r2_mean': r2_mean,
            'r2_std': r2_std,
            'mse_mean': mse_mean,
            'mse_std': mse_std,
            'vip_df': vip_df,
            'results_df': results_df,
            'n_components': {
                'mean': n_components_mean,
                'std': n_components_std,
                'mode': n_components_mode
            }
        }

    return pls_bootstrap_results