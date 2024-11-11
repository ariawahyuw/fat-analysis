import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

"""
Glossary:
- PCA: Principal Component Analysis
- KNN: K-Nearest Neighbors
- MSE: Mean Squared Error
- GridSearchCV: Exhaustive search over specified parameter values for an estimator
- n_neighbors: Number of neighbors to use by default for kneighbors queries
- Pure: Data that contains only one type of fatty acid (100% concentration)
- Mixed: Data that contains a mixture of two types of fatty acids
- Cleaned: Data that has been processed to remove unwanted data
- Mask: A filter to remove unwanted data from the dataset
"""

def main():
    import pandas as pd
    import numpy as np
    from fatty_acid.load_spectra_data import load_config
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error

    from example.load_preprocessed_data import clean

    config = load_config('file.yaml')
    wavenumbers, pure_fatty_cols, mixed_fatty_cols, cleaned_pure_spectra_data, cleaned_mixed_spectra_data, all_data = clean(config)

    pca = PCA(n_components=5)
    pca.fit(all_data.iloc[:, 0:len(wavenumbers)], all_data.loc[:, pure_fatty_cols])
    cleaned_pure_pca_data = pca.transform(cleaned_pure_spectra_data.iloc[:, 0:len(wavenumbers)])
    cleaned_mixed_pca_data = pca.transform(cleaned_mixed_spectra_data.iloc[:, 0:len(wavenumbers)])
    
    
    X = np.concatenate([cleaned_pure_pca_data, cleaned_mixed_pca_data], axis=0)
    y_mixed = cleaned_mixed_spectra_data.loc[:, mixed_fatty_cols]
    # Combine all types of pork fatty acids into one column
    y_pure = np.concatenate([
        cleaned_pure_spectra_data.loc[:, pure_fatty_cols[:4]].values, 
        cleaned_pure_spectra_data.loc[:, pure_fatty_cols[4:]].values.sum(axis=1).reshape(-1, 1)
    ], axis=1)
    y = np.concatenate([y_pure, y_mixed], axis=0)        

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    knn = KNeighborsRegressor()
    # Define the parameter grid
    param_grid = {'n_neighbors': [i for i in range(3, 30, 2)]}

    # Create GridSearchCV object
    knn_grid = GridSearchCV(knn, param_grid)

    knn_grid.fit(X_train, y_train)
    knn = knn_grid.best_estimator_
    
    y_pred_train = knn_grid.predict(X_train)
    y_pred_test = knn_grid.predict(X_test)
    y_pred_train = np.clip(y_pred_train, 0, 1)
    y_pred_test = np.clip(y_pred_test, 0, 1)

    mse_train = mean_squared_error(y_train, y_pred_train, multioutput='raw_values')
    mse_test = mean_squared_error(y_test, y_pred_test, multioutput='raw_values')
    
    class_names = ['Chicken', 'Cow', 'Duck', 'Goat', 'Pig']
        
    print(f"Best model is KNN with n_neighbors = {knn_grid.best_params_['n_neighbors']}")
    
    print(
        pd.DataFrame(
            [
                np.concatenate([mse_train, [mse_train.mean()]]),
                np.concatenate([mse_test, [mse_test.mean()]]
            )],
            columns=class_names + ['Total'],
            index=['Train', 'Test']
        )
    )

    print(
        pd.DataFrame(
            np.concatenate([y_test[:5], y_pred_test[:5]], axis=1)
            , columns=class_names + [f'{class_name}_pred' for class_name in class_names]
        )
    )

if __name__ == '__main__':
    main()