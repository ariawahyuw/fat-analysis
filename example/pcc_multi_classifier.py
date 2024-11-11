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
    import matplotlib.colors as mcolors
    from fat_analysis.fatty_acid.load_spectra_data import load_config
    from helper.plot_3d import make_meshgrid, plot_contours
    from scipy.interpolate import griddata
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, recall_score, precision_score, fbeta_score

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
    y_type = np.argmax(y, axis=1)
    y_pure_label = cleaned_pure_spectra_data["label"].values
    
    # Use y_haram as the target variable for binary classification, instead of y (which contains the fatty acid concentrations)
    X_train, X_test, y_train, y_test = train_test_split(X, y_type, test_size=0.15, random_state=0)

    # Train using K-Nearest Neighbors
    knn = KNeighborsClassifier()
    # Define the parameter grid
    param_grid = {'n_neighbors': [i for i in range(3, 30, 2)]}

    # Create GridSearchCV object
    knn_grid = GridSearchCV(knn, param_grid)
    knn_grid.fit(X_train, y_train)
    knn = knn_grid.best_estimator_

    y_pred_train = knn_grid.predict(X_train)
    y_pred_test = knn_grid.predict(X_test)
    
    class_names = ['Chicken', 'Cow', 'Duck', 'Goat', 'Pig']
    unique_labels = np.unique(y_pure_label)
    print(f"Best model is KNN with n_neighbors = {knn_grid.best_params_['n_neighbors']}")

    y_pure_type = np.argmax(y_pure, axis=1)
    y_pred_pure_type = knn.predict(cleaned_pure_pca_data)
    y_pure_class = np.array([class_names[i] for i in y_pure_type])
    y_pred_pure_class = np.array([class_names[i] for i in y_pred_pure_type])
    
    recall = recall_score(y_pure_type, y_pred_pure_type, average='weighted')
    precision = precision_score(y_pure_type, y_pred_pure_type, average='weighted')
    f2 = fbeta_score(y_pure_type, y_pred_pure_type, beta=2, average='weighted')
    
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F2: ", f2)
    print(classification_report(y_pure_type, y_pred_pure_type, target_names=class_names))

    X0, X1 = cleaned_pure_pca_data[:, 0], cleaned_pure_pca_data[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    X2, X3, X4 = cleaned_pure_pca_data[:, 2], cleaned_pure_pca_data[:, 3], cleaned_pure_pca_data[:, 4]
    X_closest = np.c_[
        xx.ravel(), yy.ravel(), 
        griddata((X0, X1), X2, (xx, yy), method='nearest').ravel(),
        griddata((X0, X1), X3, (xx, yy), method='nearest').ravel(),
        griddata((X0, X1), X4, (xx, yy), method='nearest').ravel()
    ]

    fig, ax =  plt.subplots(figsize=(10, 10))
    
    norm = plt.Normalize()
    colors = plt.cm.viridis(norm(np.arange(5)))
    added_colors = ['gold', 'tan', 'orange']
    added_colors_rgba = [mcolors.to_rgba(color) for color in added_colors]
    all_colors = np.vstack([colors, added_colors_rgba])
    markers = ['X', 'v', 's', 'o', '^', '^', '^', '^']
    
    plot_contours(ax, knn, X_closest, xx, yy, cmap='viridis', alpha=0.8, extend='max')
    for i, label in enumerate(unique_labels):
        mask = y_pure_label == label
        ax.scatter(
            cleaned_pure_pca_data[mask, 0], 
            cleaned_pure_pca_data[mask, 1], 
            label=label, 
            marker=markers[i],
            color=all_colors[i],
            s=70,
            linewidths=0.25,
            edgecolors='k'
        )
    legend_elements = [
        plt.scatter([], [], color=all_colors[i], marker=markers[i], label=label)
        for i, label in enumerate(unique_labels)
    ]
    legend_elements += [plt.Line2D([0], [0], marker='s', label=class_names[i], color='w', markerfacecolor=colors[i], markersize=10) for i in range(5)]
    ax.set_xlabel('Principal component 1')
    ax.set_ylabel('Principal component 2')
    ax.legend(handles=legend_elements, loc='upper right')
    plt.show()

    fig, ax =  plt.subplots(figsize=(10, 10))
    plot_contours(ax, knn, X_closest, xx, yy, cmap='viridis', alpha=0.3, extend='max')
    for i, class_ in enumerate(class_names):
        mask = y_pure_class == class_
        pred_mask = y_pred_pure_class == class_
        misclassified_mask = y_pure_class != y_pred_pure_class
        ax.scatter(
            cleaned_pure_pca_data[(mask & ~misclassified_mask), 0],
            cleaned_pure_pca_data[(mask & ~misclassified_mask), 1],
            label=label, 
            marker=markers[min(i, 4)],
            color=colors[min(i, 4)],
            s=70,
            linewidths=0.25,
            edgecolors='k',
            alpha=0.3
        )
        ax.scatter(
            cleaned_pure_pca_data[(pred_mask & misclassified_mask), 0],
            cleaned_pure_pca_data[(pred_mask & misclassified_mask), 1],
            label="false " + label,
            marker=markers[min(i, 4)],
            color=colors[min(i, 4)],
            s=70,
            linewidths=0.5,
            edgecolors='r'
        )
    legend_elements = [
        plt.scatter([], [], color=colors[i], marker=markers[i], edgecolors='k', linewidths=0.25, label=class_names[i])
        for i in range(5)
    ]
    legend_elements += [
        plt.scatter([], [], color=colors[i], marker=markers[i], edgecolors='r', linewidths=0.5, label='False ' + class_names[i])
        for i in range(5)
    ]
    legend_elements += [plt.Line2D([0], [0], color='w', marker='s', alpha=0.3, markerfacecolor=colors[i], markersize=10, label=class_names[i]) for i in range(5)]
    ax.set_xlabel('Principal component 1')
    ax.set_ylabel('Principal component 2')
    ax.legend(handles=legend_elements)
    plt.show()

if __name__ == '__main__':
    main()