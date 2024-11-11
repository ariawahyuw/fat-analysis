# Principal Component Regression

Principal Component Analysis (PCA) is a statistical method to reduce the dimensionality of the dataset. It is used to identify the patterns in data and express the data in such a way to highlight their similarities and differences.

Table of contents:

- [PC Regression](#pc-regression)
- [PC Regression + Binary Classification](#pc-regression--binary-classification)
- [PC Regression + Multiclass Classification](#pc-regression--multiclass-classification)

## PC Regression

The PCA is combined with regression to predict the fatty acid concentration in the dataset. To load the cleaned dataset, the following code is used:

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

```

The `X` consists of the PCA of the cleaned dataset and the PCA of the mixture dataset. The `y` consists of the fatty acid concentration of the cleaned dataset and the mixture dataset.

The regression is conducted to predict the fatty acid concentration in the dataset. The regression result is evaluated using the Mean Squared Error (MSE) and Mean Absolute Error (MAE).

We can use the K-Nearest Neighbors (KNN) regression to predict the fatty acid concentration in the dataset. We also use the GridSearchCV to find the best parameter for the KNN regression.

After the regression, we need to clip the predicted value to 0 and 1 because the fatty acid concentration is between 0 and 1.

The following code is used to conduct the regression:

```python
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
```

## PC Regression + Binary Classification

To get the regression results for **halal and haram classification**, we can just map the concentration to halal or haram class by checking the concentration of the pig fat. If the concentration of the pig fat is more than 0, then it is haram. Otherwise, it is halal.

```python
test_false_halal = np.where((y_test[:, 4] > 0.0) & (y_pred_test[:, 4] == 0.0))[0]
test_false_haram = np.where((y_test[:, 4] == 0.0) & (y_pred_test[:, 4] > 0.0))[0]
test_true_halal = np.where((y_test[:, 4] == 0.0) & (y_pred_test[:, 4] == 0.0))[0]
test_true_haram = np.where((y_test[:, 4] > 0.0) & (y_pred_test[:, 4] > 0.0))[0]
```

To evaluate the classification result, we can use the confusion matrix, recall, precision, and F1 score. The following code is used to get the regression results for halal and haram classification:

```python
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
```

We can plot the PCA of the dataset to visualize the classification result. The following code is used to plot the PCA of the dataset:

```python
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
```

To plot the decision boundary of the classification, we can use `plot_contours` function. First, for plotting purposes, the following functions are used to make meshgrid and plot contours from collection of points:

```python
def make_meshgrid(x, y, h=.05):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, X_closest, xx, yy, **params):
    Z = clf.predict(X_closest)
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
```

To plot the results, we need to make the meshgrid from the PCA dataset. We also need to get the closest value for points which are not in the PCA dataset.
The following code is used to make the meshgrid and get the value for points which are not in the PCA dataset:

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

```

## PC Regression + Multiclass Classification

The classification is conducted to classify the fatty acid content in the dataset. We can use the previous regression result (from [PC Regression](#pc-regression)) to classify the fatty acid content in the dataset. The following code is used to classify the fatty acid content in the dataset:

```python
class_names = ['Chicken', 'Cow', 'Duck', 'Goat', 'Pig']
unique_labels = np.unique(y_pure_label)

# Try to predict for pure data only
y_pred_pure = knn.predict(cleaned_pure_pca_data)
y_pure_type = np.argmax(y_pure, axis=1)
y_pure_class = np.array([class_names[i] for i in y_pure_type])
y_pred_pure_type = np.argmax(y_pred_pure, axis=1)
y_pred_pure_class = np.array([class_names[i] for i in y_pred_pure_type])
```

To evaluate the classification result, we can use the confusion matrix, recall, precision, and F1 score. The following code is used to evaluate the classification result:

```python
recall = recall_score(y_pure_type, y_pred_pure_type, average='weighted')
precision = precision_score(y_pure_type, y_pred_pure_type, average='weighted')
f2 = fbeta_score(y_pure_type, y_pred_pure_type, beta=2, average='weighted')

print("Recall: ", recall)
print("Precision: ", precision)
print("F2: ", f2)
print(classification_report(y_pure_type, y_pred_pure_type, target_names=class_names))
```

To plot the decision boundary of the classification, we can use `plot_contours` function. The following code is used to plot the decision boundary and the points in the PCA dataset:

```python
X0, X1 = cleaned_pure_pca_data[:, 0], cleaned_pure_pca_data[:, 1]
xx, yy = make_meshgrid(X0, X1)
X2, X3, X4 = cleaned_pure_pca_data[:, 2], cleaned_pure_pca_data[:, 3], cleaned_pure_pca_data[:, 4]
X_closest = np.c_[xx.ravel(), yy.ravel(), griddata((X0, X1), X2, (xx, yy), method='nearest').ravel(), griddata((X0, X1), X3, (xx, yy), method='nearest').ravel(), griddata((X0, X1), X4, (xx, yy), method='nearest').ravel()]


# Plot the decision boundary for all types of fatty acids
fig, ax =  plt.subplots(figsize=(10, 10))
norm = plt.Normalize()
colors = plt.cm.viridis(norm(np.arange(5)))
added_colors = ['gold', 'tan', 'orange']
added_colors_rgba = [mcolors.to_rgba(color) for color in added_colors]
colors = np.vstack([colors, added_colors_rgba])
markers = ['X', 'v', 's', 'o', '^', '^', '^', '^']
plot_contours(ax, knn, X_closest, xx, yy, type="multi", cmap='viridis', alpha=0.8, extend='max')
for i, label in enumerate(unique_labels):
    mask = y_pure_label == label
    ax.scatter(
        cleaned_pure_pca_data[mask, 0], 
        cleaned_pure_pca_data[mask, 1], 
        label=label, 
        marker=markers[i],
        color=colors[i],
        s=70,
        linewidths=0.5,
        edgecolors='k'
    )
legend_elements = [
    plt.scatter([], [], color=colors[i], marker=markers[i], label=label)
    for i, label in enumerate(unique_labels)
]
legend_elements += [plt.Line2D([0], [0], marker='s', label=class_names[i], color='w', markerfacecolor=colors[i], markersize=10) for i in range(5)]
ax.set_xlabel('Principal component 1', fontsize=12)
ax.set_ylabel('Principal component 2', fontsize=12)
ax.set_xticks(())
ax.set_yticks(())
ax.legend(handles=legend_elements, fontsize=12, loc='upper right')
plt.show()
```

For further analysis, we can get the misclassified results from the classification results. The following code is used to plot the misclassified results:

```python
false_halal_train = np.where((y_train == 1) & (y_train_pred == 0))[0]
false_haram_train = np.where((y_train == 0) & (y_train_pred == 1))[0]
false_halal_test = np.where((y_test == 1) & (y_test_pred == 0))[0]
false_haram_test = np.where((y_test == 0) & (y_test_pred == 1))[0]
```

Then, we can plot the false halal and false haram in the PCA dataset. The following code is used to plot the false halal and false haram in the PCA dataset:

```python
fig, ax =  plt.subplots(figsize=(10, 10))
plot_contours(ax, knn, X_closest, xx, yy, type="multi", cmap='viridis', alpha=0.3, extend='max')
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
