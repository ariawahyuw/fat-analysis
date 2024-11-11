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
- Pred: Predicted
- True: Actual
"""

def main():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from fat_analysis.load_spectra_data import load_config
    from fat_analysis.helper.plot_3d import make_meshgrid, plot_contours
    from scipy.interpolate import griddata
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.neighbors import KNeighborsRegressor

    from example.load_preprocessed_data import clean

    config = load_config('file.yaml')
    wavenumbers, pure_fatty_cols, mixed_fatty_cols, cleaned_pure_spectra_data, cleaned_mixed_spectra_data, all_data = clean(config) 

    # Apply PCA to all data
    pca = PCA(n_components=5)
    pca.fit(all_data.iloc[:, 0:len(wavenumbers)], all_data.loc[:, pure_fatty_cols])
    cleaned_pure_pca_data = pca.transform(cleaned_pure_spectra_data.iloc[:, 0:len(wavenumbers)])
    cleaned_mixed_pca_data = pca.transform(cleaned_mixed_spectra_data.iloc[:, 0:len(wavenumbers)])
    
    # Create the training and testing dataset
    X = np.concatenate([cleaned_pure_pca_data, cleaned_mixed_pca_data], axis=0)
    y_mixed = cleaned_mixed_spectra_data.loc[:, mixed_fatty_cols]
    # Combine all types of pork fatty acids into one column
    y_pure = np.concatenate([
        cleaned_pure_spectra_data.loc[:, pure_fatty_cols[:4]].values, 
        cleaned_pure_spectra_data.loc[:, pure_fatty_cols[4:]].values.sum(axis=1).reshape(-1, 1)
    ], axis=1)
    y = np.concatenate([y_pure, y_mixed], axis=0)
    y_pure_label = cleaned_pure_spectra_data["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    # Train using K-Nearest Neighbors Regressor
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

    print(f"Best model is KNN with n_neighbors = {knn_grid.best_params_['n_neighbors']}")
    

    # Evaluate the model using the test dataset
    test_false_halal = np.where((y_test[:, 4] > 0.0) & (y_pred_test[:, 4] == 0.0))[0]
    test_false_haram = np.where((y_test[:, 4] == 0.0) & (y_pred_test[:, 4] > 0.0))[0]
    test_true_halal = np.where((y_test[:, 4] == 0.0) & (y_pred_test[:, 4] == 0.0))[0]
    test_true_haram = np.where((y_test[:, 4] > 0.0) & (y_pred_test[:, 4] > 0.0))[0]
    cm = np.array([
        [len(test_true_halal), len(test_false_haram)],
        [len(test_false_halal), len(test_true_haram)]
    ])
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    f1 = 2 * (precision * recall) / (precision + recall)
    print(recall, precision, f1)
    print(
        pd.DataFrame(cm, columns=['Predicted Halal', 'Predicted Haram'], index=['Actual Halal', 'Actual Haram'])
    )

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(y_pure_label)
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange']
    # Plot each label category separately
    for i, label in enumerate(unique_labels):
        mask = y_pure_label == label
        plt.scatter(
            cleaned_pure_pca_data[mask, 0], 
            cleaned_pure_pca_data[mask, 1], 
            label=label, 
            marker='s', 
            c=colors[i],
            alpha=0.2
        )
    plt.scatter(X_test[test_false_haram, 0], X_test[test_false_haram, 1], c='r', label='Wrong haram', marker='x')
    plt.scatter(X_test[test_false_halal, 0], X_test[test_false_halal, 1], c='g', label='Wrong halal', marker='^')
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.title('Test Dataset Model Evaluation', fontname='Times New Roman', fontsize=15)
    plt.legend(prop={'family': 'Times New Roman', 'size': 10}, loc='upper right')
    plt.show()


    # Evaluate the model using the pure dataset
    pure_false_halal = np.where((y_pure[:, 4] > 0.0) & (knn.predict(cleaned_pure_pca_data).clip(0, 1)[:, 4] == 0.0))[0]
    pure_false_haram = np.where((y_pure[:, 4] == 0.0) & (knn.predict(cleaned_pure_pca_data).clip(0, 1)[:, 4] > 0.0))[0]
    pure_true_halal = np.where((y_pure[:, 4] == 0.0) & (knn.predict(cleaned_pure_pca_data).clip(0, 1)[:, 4] == 0.0))[0]
    pure_true_haram = np.where((y_pure[:, 4] > 0.0) & (knn.predict(cleaned_pure_pca_data).clip(0, 1)[:, 4] > 0.0))[0]
    cm = np.array([
        [len(pure_true_halal), len(pure_false_haram)],
        [len(pure_false_halal), len(pure_true_haram)]
    ])
    print(pd.DataFrame(cm, columns=['Predicted Halal', 'Predicted Haram'], index=['Actual Halal', 'Actual Haram']))

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(y_pure_label)
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange']
    # Plot each label category separately
    for i, label in enumerate(unique_labels):
        mask = y_pure_label == label
        plt.scatter(
            cleaned_pure_pca_data[mask, 0], 
            cleaned_pure_pca_data[mask, 1], 
            label=label, 
            marker='s', 
            c=colors[i],
            alpha=0.2
        )
    plt.scatter(cleaned_pure_pca_data[pure_false_haram, 0], cleaned_pure_pca_data[pure_false_haram, 1], c='r', label='Wrong haram', marker='x')
    plt.scatter(cleaned_pure_pca_data[pure_false_halal, 0], cleaned_pure_pca_data[pure_false_halal, 1], c='g', label='Wrong halal', marker='^')
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.title('Pure Dataset Model Evaluation', fontname='Times New Roman', fontsize=15)
    plt.legend(prop={'family': 'Times New Roman', 'size': 10}, loc='upper right')
    plt.show()
    

    # Plot the decision boundary for halal-haram classification
    X0, X1 = cleaned_pure_pca_data[:, 0], cleaned_pure_pca_data[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    X2, X3, X4 = cleaned_pure_pca_data[:, 2], cleaned_pure_pca_data[:, 3], cleaned_pure_pca_data[:, 4]
    X_closest = np.c_[xx.ravel(), yy.ravel(), griddata((X0, X1), X2, (xx, yy), method='nearest').ravel(), griddata((X0, X1), X3, (xx, yy), method='nearest').ravel(), griddata((X0, X1), X4, (xx, yy), method='nearest').ravel()]

    fig, ax =  plt.subplots(figsize=(10, 10))
    colors = ['gray', 'g', 'b', 'y', 'c', 'm', 'k', 'orange']
    plot_contours(ax, knn, X_closest, xx, yy, type="binary", cmap=plt.cm.coolwarm, alpha=0.5)
    for i, label in enumerate(unique_labels):
        mask = y_pure_label == label
        ax.scatter(
            cleaned_pure_pca_data[mask, 0], 
            cleaned_pure_pca_data[mask, 1], 
            label=label, 
            marker='o', 
            c=colors[i],
            alpha=0.4,
            linewidths=0.5,
            edgecolors='black'
        )
    ax.scatter(cleaned_pure_pca_data[pure_false_halal, 0], cleaned_pure_pca_data[pure_false_halal, 1], c='g', marker='^', label='Wrong halal')
    ax.scatter(cleaned_pure_pca_data[pure_false_haram, 0], cleaned_pure_pca_data[pure_false_haram, 1], c='r', marker='x', label='Wrong haram')
    legend_elements = [
        plt.scatter([], [], c=colors[i], marker='o', alpha=0.4, linewidths=0.5, edgecolors='k', label=label)
        for i, label in enumerate(unique_labels)
    ]
    legend_elements += [plt.Line2D([0], [0], color='w', marker='s', alpha=0.5, markerfacecolor=plt.cm.coolwarm(0), markersize=10, label='Halal'),
                        plt.Line2D([0], [0], color='w', marker='s', alpha=0.5, markerfacecolor=plt.cm.coolwarm(255), markersize=10, label='Haram')]
    legend_elements += [plt.scatter([], [], c='g', marker='^', label='Wrong halal'),
                        plt.scatter([], [], c='r', marker='x', label='Wrong haram')]
    ax.set_xlabel('Principal component 1')
    ax.set_ylabel('Principal component 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('Decision Boundary for Halal-Haram Classification', fontname='Times New Roman', fontsize=15)
    ax.legend(handles=legend_elements)
    plt.show()

if __name__ == '__main__':
    main()