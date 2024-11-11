# Principal Component Classification

Principal Component Analysis (PCA) is a statistical method to reduce the dimensionality of the dataset. It is used to identify the patterns in data and express the data in such a way to highlight their similarities and differences.

## PC Binary Classification

The PCA is combined with the binary classification to classify the dataset into two classes. The following code is used to load the dataset and clean the dataset:

```python
from fatty_acid.load_spectra_data import load_config

# ... define clean method for the dataset ...

config = load_config('file.yaml')
wavenumbers, pure_fatty_cols, mixed_fatty_cols, cleaned_pure_spectra_data, cleaned_mixed_spectra_data, all_data = clean(config)
```

Note that the all_data consists of non-cleaned data.

The PCA is conducted to the dataset to reduce the dimensionality of the dataset. The following code is used to transform the cleaned dataset using the PCA with 5 principal components:

```python
pca = PCA(n_components=5)
pca.fit(all_data.iloc[:, 0:len(wavenumbers)], all_data.loc[:, pure_fatty_cols])
cleaned_pure_pca_data = pca.transform(cleaned_pure_spectra_data.iloc[:, 0:len(wavenumbers)])
cleaned_mixed_pca_data = pca.transform(cleaned_mixed_spectra_data.iloc[:, 0:len(wavenumbers)])
```

To prepare the dataset for regression, the following code is used:

```python
X = np.concatenate([cleaned_pure_pca_data, cleaned_mixed_pca_data], axis=0)
y_mixed = cleaned_mixed_spectra_data.loc[:, mixed_fatty_cols]
# Combine all types of pork fatty acids into one column (pork)
y_pure = np.concatenate([
    cleaned_pure_spectra_data.loc[:, pure_fatty_cols[:4]].values, 
    cleaned_pure_spectra_data.loc[:, pure_fatty_cols[4:]].values.sum(axis=1).reshape(-1, 1)
], axis=1)
y_pure_label = cleaned_pure_spectra_data["label"].values
y = np.concatenate([y_pure, y_mixed], axis=0)
y_haram = np.where(y[:, 4] > 0, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y_haram, test_size=0.15, random_state=0)

```

The `X` consists of the PCA of the cleaned dataset and the PCA of the mixture dataset. The `y_haram` is the label for the binary classification. The `y` consists of the fatty acid concentration in the dataset. The `y_pure` consists of the fatty acid concentration in the pure dataset. The `y_pure_label` consists of the label for the pure dataset.

We can use the K-Nearest Neighbors (KNN) classification to classify the dataset. We also use the GridSearchCV to find the best parameter for the KNN classification. The following code is used to conduct the classification:

```python
knn = KNeighborsClassifier()
# Define the parameter grid
param_grid = {'n_neighbors': [i for i in range(3, 30, 2)]}

# Create GridSearchCV object
knn_grid = GridSearchCV(knn, param_grid)
knn_grid.fit(X_train, y_train)
knn = knn_grid.best_estimator_

y_pred_train = knn_grid.predict(X_train)
y_pred_test = knn_grid.predict(X_test)

unique_labels = np.unique(y_pure_label)

test_false_halal = np.where((y_test == 0) & (y_pred_test == 1))[0]
test_false_haram = np.where((y_test == 1) & (y_pred_test == 0))[0]
train_false_halal = np.where((y_train == 0) & (y_pred_train == 1))[0]
train_false_haram = np.where((y_train == 1) & (y_pred_train == 0))[0]
```

To evaluate the classification result, we can use the classification report. The following code is used to get the classification results for halal and haram classification:

```python
print(f"Best model is KNN with n_neighbors = {knn_grid.best_params_['n_neighbors']}")
print("Classification report for training dataset:")
print(classification_report(y_train, y_pred_train, target_names=['Halal', 'Haram'], zero_division=0) + '\n')
print("Classification report for testing dataset:")
print(classification_report(y_test, y_pred_test, target_names=['Halal', 'Haram'], zero_division=0) + '\n')
```

To plot the results, we need to make the meshgrid from the PCA dataset. We also need to get the closest value for points which are not in the PCA dataset. The following code is used to make the meshgrid and get the value for points which are not in the PCA dataset:

```python
X0, X1 = cleaned_pure_pca_data[:, 0], cleaned_pure_pca_data[:, 1]
xx, yy = make_meshgrid(X0, X1)
X2, X3, X4 = cleaned_pure_pca_data[:, 2], cleaned_pure_pca_data[:, 3], cleaned_pure_pca_data[:, 4]
X_closest = np.c_[
    xx.ravel(), yy.ravel(), 
    griddata((X0, X1), X2, (xx, yy), method='nearest').ravel(),
    griddata((X0, X1), X3, (xx, yy), method='nearest').ravel(),
    griddata((X0, X1), X4, (xx, yy), method='nearest').ravel()
]
```

Then, we can plot the decision boundary and the points in the PCA dataset. The following code is used to plot the decision boundary and the points in the PCA dataset:

```python
fig, ax =  plt.subplots(figsize=(10, 10))
plot_contours(ax, knn, X_closest, xx, yy, cmap=plt.cm.coolwarm, alpha=0.2)
for i, label in enumerate(unique_labels):
    mask = y_pure_label == label
    ax.scatter(
        cleaned_pure_pca_data[mask, 0], 
        cleaned_pure_pca_data[mask, 1], 
        label=label, 
        marker='o', 
        c=colors[i],
        alpha=0.3
    )
ax.scatter(X_train[train_false_halal, 0], X_train[train_false_halal, 1], c='g', marker='^', label='Wrong halal')
ax.scatter(X_train[train_false_haram, 0], X_train[train_false_haram, 1], c='r', marker='x', label='Wrong haram')
ax.scatter(X_test[test_false_halal, 0], X_test[test_false_halal, 1], c='g', marker='^', label='Wrong halal')
ax.scatter(X_test[test_false_haram, 0], X_test[test_false_haram, 1], c='r', marker='x', label='Wrong haram')
legend_elements = [
    plt.scatter([], [], c=colors[i], marker='o', alpha=0.3, label=label)
    for i, label in enumerate(unique_labels)
]
legend_elements += [plt.Line2D([0], [0], color='w', marker='s', alpha=0.2, markerfacecolor=plt.cm.coolwarm(0), markersize=10, label='Halal'),
                    plt.Line2D([0], [0], color='w', marker='s', alpha=0.2, markerfacecolor=plt.cm.coolwarm(255), markersize=10, label='Haram')]
legend_elements += [plt.scatter([], [], c='g', marker='^', label='Wrong halal'),
                    plt.scatter([], [], c='r', marker='x', label='Wrong haram')]
ax.set_xlabel('Principal component 1')
ax.set_ylabel('Principal component 2')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('K-Nearest Neighbors Decision Boundary with Misclassification')
ax.legend(handles=legend_elements)
plt.show()
```

## PC Multiclass Classification

For the multiclass classification, we need to change the target label to the type of fatty acids. The following code is used to define the target label:

```python
# ...
y_type = np.argmax(y, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y_type, test_size=0.15, random_state=0)
```

To evaluate the classification result, we can use the classification report, recall, precision, and F2 score. The following code is used to evaluate the classification result:

```python
class_names = ['Chicken', 'Cow', 'Duck', 'Goat', 'Pig']
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
```

The `class_names` consists of the type of fatty acids. The `y_pure_type` consists of the type of fatty acids (in integer). The `y_pred_pure_type` consists of the predicted type of fatty acids (in integer). The `y_pure_class` consists of the type of fatty acids (in string). The `y_pred_pure_class` consists of the predicted type of fatty acids (in string).

To plot the decision boundary of the classification, we can use `plot_contours` function. The following code is used to plot the decision boundary and the points in the PCA dataset:

```python
X0, X1 = cleaned_pure_pca_data[:, 0], cleaned_pure_pca_data[:, 1]
xx, yy = make_meshgrid(X0, X1)
X2, X3, X4 = cleaned_pure_pca_data[:, 2], cleaned_pure_pca_data[:, 3], cleaned_pure_pca_data[:, 4]
X_closest = np.c_[
    xx.ravel(), yy.ravel(), 
    griddata((X0, X1), X2, (xx, yy), method='nearest').ravel(),
    griddata((X0, X1), X3, (xx, yy), method='nearest').ravel(),
    griddata((X0, X1), X4, (xx, yy), method='nearest').ravel()
]

# Plot the decision boundary for all types of fatty acids
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
```

For further analysis, we can get the misclassified results from the classification results. The following code is used to plot the misclassified results:

```python
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
```
