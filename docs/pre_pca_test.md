# Pre-PCA Test

## Mixture Test

The mixture test is conducted to observe the mixture of fatty acids in the dataset. Each dataset is loaded using the following code:

```python
from fatty_acid.load_spectra_data import load_config, load_data

config = load_config('file.yaml')
pure_data = load_data(config, 'pure')
mixed_data = load_data(config, 'mixed')

pure_data, wavenumbers = fill_concentration_to_data(pure_data, config, 'pure')
mixed_data, _ = fill_concentration_to_data(mixed_data, config, 'mixed')

# ... define clean method for the dataset ...

wavenumbers, pure_fatty_cols, mixed_fatty_cols, cleaned_pure_spectra_data, cleaned_mixed_spectra_data, all_data = clean(config)
```

and for the mean dataset, the following code is used:

```python
# ...
mean_pure_data = load_data(config, 'pure', 'mean')
```

The PCA is conducted to the dataset to observe the mixture of fatty acids in the dataset. The following code is used to conduct the PCA:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=200)
pure_pca_data = pca.fit_transform(pure_data.iloc[:, 0:len(wavenumbers)], pure_data.loc[:, pure_fatty_cols])
```

The pca_data has shape of `(m, l)` where `m` is the number of wavenumbers and `l` is the number of components.
To plot the first two components of the PCA, the following code is used:

```python
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange']
for i, label in enumerate(pure_fatty_cols):
    mask = pure_data["label"] == label
    plt.scatter(
        pure_pca_data[mask, 0], 
        pure_pca_data[mask, 1], 
        label=label, 
        c=colors[i]
    )
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.legend(prop={'family': 'Times New Roman', 'size': 10}, loc='upper right')
plt.show()
```

By observing the PCA plot, we can observe if the dataset is separable or not. If the dataset is separable, we can proceed to the next step. Otherwise, we need to clean the dataset again.

## Dominant Wavenumbers

To get the dominant wavenumbers in the dataset, first the cumulative variance of the PCA is plotted. The following code is used to plot the cumulative variance of the PCA:

```python
pca = PCA(n_components=0.95)
pca.fit(all_data.iloc[:, 0:len(wavenumbers)], all_data.loc[:, cols])
pca_data = pca.transform(pure_data.iloc[:, 0:len(wavenumbers)])
```

The first 20 components are plotted to observe the cumulative variance of the PCA:

```python
x = np.arange(1, 21)
y = pca.explained_variance_ratio_[:20].cumsum()
plt.plot(x, y, marker='o', linestyle='--', color='k', linewidth=1)
plt.xlabel('Principal component')
plt.ylabel('Cumulative variance')
plt.xticks(np.arange(1, 21, 2))
plt.show()
```

We can get the dominant wavenumbers by getting the principal component vector length and sorting the length.  We can cluster the wavenumbers using the KMeans algorithm to get the dominant wavenumbers. The following code is used to get the dominant wavenumbers:

```python
com = pca.components_[:2]
norm = np.linalg.norm(np.abs(com), axis=0)
norm_idx = np.argsort(norm)[::-1]
kmeans = KMeans(n_clusters=40, random_state=0).fit(norm_idx.reshape(-1, 1))
kmeans.labels_
top_raw_idx = {}
for i, val in enumerate(norm_idx):
    if kmeans.labels_[i] not in top_raw_idx:
        top_raw_idx[kmeans.labels_[i]] = []
    top_raw_idx[kmeans.labels_[i]].append(val)
    
top_idx = {k: int(np.mean(v)) for k, v in top_raw_idx.items()}
top_idx = sorted(top_idx.items(), key=lambda x: norm[x[1]], reverse=True)
top_idx = [x[1] for x in top_idx[:20]]
```

The dominant wavenumbers are plotted to observe the dominant wavenumbers in the dataset. The following code is used to plot the top 5 dominant wavenumbers:

```python
plt.figure(figsize=(10, 8))
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange']
for i, label in enumerate(pure_fatty_cols):
    mask = pure_data["label"] == label
    plt.plot(
        wavenumbers, 
        i + mean_pure_data[label].iloc[:, 1],
        label=label,
        c=colors[i]
    )

for i, val in enumerate(top_idx[:5]):
    plt.axvline(x=wavenumbers[val], color='k', linestyle='--', alpha=0.5)
    plt.annotate(str(i+1), (wavenumbers[val] - 12, 0.9), textcoords="offset points", xytext=(0,350), ha='center')
plt.legend()
plt.xlabel(r'Wavenumber $(cm^{-1})$')
plt.ylabel(r'Relative Intensity $(arb. units)$')
plt.yticks([])
plt.show()
```

After we get the dominant wavenumbers, we can plot the PCA of the dataset to observe the dominant wavenumbers in the dataset. The following code is used to transform the dataset using the PCA with 5 principal components:

```python
pca = PCA(n_components=5)
pca.fit(all_data.iloc[:, 0:1012], all_data.loc[:, cols])
pca_data = pca.transform(pure_data.iloc[:, 0:1012])
```
