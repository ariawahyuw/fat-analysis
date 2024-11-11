import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    from fat_analysis.load_spectra_data import load_config, load_data, fill_concentration_to_data
    from example.load_preprocessed_data import clean

    config = load_config('file.yaml')
    pure_data = load_data(config, 'pure')
    mixed_data = load_data(config, 'mixed')
    mean_pure_data = load_data(config, 'pure', 'mean')

    pure_data, wavenumbers = fill_concentration_to_data(pure_data, config, 'pure')
    mixed_data, _ = fill_concentration_to_data(mixed_data, config, 'mixed')

    wavenumbers, pure_fatty_cols, _, _, _, all_data = clean(config)


    # Mixture Test

    pca = PCA(n_components=200)
    pure_pca_data = pca.fit_transform(pure_data.iloc[:, 0:len(wavenumbers)], pure_data.loc[:, pure_fatty_cols])

    print(f"Explained variance ratio: {np.round(pca.explained_variance_ratio_[0:8], 3)}")

    mixed_pca_data = pca.transform(mixed_data.iloc[:, 0:len(wavenumbers)])

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


    # Dominant Wavenumber Test
    pca = PCA(n_components=0.95)
    pca.fit(all_data.iloc[:, 0:len(wavenumbers)], all_data.loc[:, pure_fatty_cols])
    pca_data = pca.transform(pure_data.iloc[:, 0:len(wavenumbers)])

    x = np.arange(1, 21)
    y = pca.explained_variance_ratio_[:20].cumsum()
    plt.plot(x, y, marker='o', linestyle='--', color='k', linewidth=1)
    plt.xlabel('Principal component')
    plt.ylabel('Cumulative variance')
    plt.xticks(np.arange(1, 21, 2))
    plt.show()

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

if __name__ == '__main__':
    main()