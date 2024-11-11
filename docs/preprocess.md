# Preprocess Spectra Data

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Preprocessing](#2-preprocessing)
- [3. Mean Analysis](#3-mean-analysis)

## 1. Introduction

The preprocessing steps are divided into several steps:

- [Data Acquisition and Loading](#21-data-acquisition-and-loading)
- [Despiking](#22-despiking)
- [Baseline correction and Normalization](#23-baseline-correction-and-normalization)
- [Noise Filtering](#24-noise-filtering)
- [Generating Output](#25-generating-output)

Refer to this article for further information about preprocessing steps: \[[link](https://towardsdatascience.com/data-science-for-raman-spectroscopy-a-practical-example-e81c56cf25f)\]\[[alternative](https://github.com/nicocopez/Data-Science-for-Raman-spectroscopy-a-practical-example/blob/main/Workshop%20ML%20with%20Python%20-%20DS%20for%20Raman%20spectroscopy%20-%20An%20example%20by%20NCL.ipynb)\]

## 2. Preprocessing

### 2.1. Data Acquisition and Loading

The datasets are loaded from the selected path. For mixed spectra data, the data is needed to be loaded by each variations of concentrations combination.
To load the data, the following code is used:

```python
import os
from modules.preprocess import read_data

chicken_path = '../data/Data Lemak/pure fat/chicken/'
data_chicken = []
for root, dirs, files in os.walk(chicken_path):
    data = read_data(root+'/')
    if len(data) > 0:
        data_chicken += data
```

For the mixed spectra data, the data is loaded by each variations of concentrations combination as follows:

```python
pig_goat_path = '../data/Data Lemak/contamination/pig-goat/'
data_pig_goat_25_75 = []
data_pig_goat_50_50 = []
data_pig_goat_75_25 = []
for root, dirs, files in os.walk(pig_goat_path):
    if root.split('/')[-1].startswith('25-75'):
        data = read_data(root+'/')
        if len(data) > 0:
            data_pig_goat_25_75 += data
    elif root.split('/')[-1].startswith('50-50'):
        data = read_data(root+'/')
        if len(data) > 0:
            data_pig_goat_50_50 += data
    elif root.split('/')[-1].startswith('75-25'):
        data = read_data(root+'/')
        if len(data) > 0:
            data_pig_goat_75_25 += data
```

### 2.2. Despiking

Despiking is conducted to remove the spikes in the dataset. The method used in this simulation is by observing the modified Z-score of the spectra. These processes are done manually.

#### 2.2.1. Identify the Spectra with Spikes

The identifying process is done by plotting the spectra and observing the spikes in the dataset. The following code is used to plot the spectra and observe the spikes in the dataset:

```python
import matplotlib.pyplot as plt

spectra_with_spikes = [4] # The index of the spectra with spikes. Use [] to observe all spectra.

fig, ax = plt.subplots(figsize=(10,5))
for i in range(0, len(data_chicken)):
    if i not in spectra_with_spikes:
        ax.plot(data_chicken[i][:,0], data_chicken[i][:,1], label='Spectra ' + str(i+1))
ax.set_xlabel('Wavenumber (cm-1)')
ax.set_ylabel('Absorbance')
ax.set_title('Chicken')
plt.show()
```

#### 2.2.2. Despike the Spectra

Once the spectra with spikes are identified, the despiking process is conducted by using the `fixer` function. The following code is used to despiking the spectra:

```python
from modules.preprocess import fixer

despike_spectra_4 = fixer(data_chicken[4][:,1], 10)
data_chicken[4][:,1] = despike_spectra_4
```

The despiking process might be conducted multiple times to ensure the spikes are removed from the dataset.

```python
from modules.preprocess import fixer

despike_spectra_4 = fixer(fixer(data_chicken[4][:,1], 15, threshold=4), 10, threshold=4)
data_chicken[4][:,1] = despike_spectra_4
```

Refer to the [introduction](#1-introduction) section above for further information about the despiking process.

### 2.3. Baseline correction and Normalization

Baseline correction is conducted to remove the fluorescence background in the dataset. The method used in this simulation is the **Asymmetric Least Squares Smoothing** method. The parameters used in this method are selected manually, based on the observation of the dataset.

```python
from modules.preprocess import baseline_als

df_chicken = data_chicken[0][:,0]
for i in range(0, len(data_chicken)):
    # Baseline correction
    baseline_= baseline_als(data_chicken[i][:,1], 100000, 0.0001)
    df_chicken = np.vstack((df_chicken, data_chicken[i][:,1] - baseline_))
df_chicken = np.transpose(df_chicken)
scaler = MinMaxScaler()
df_chicken_scaled = scaler.fit_transform(df_chicken[:,1:])
df_chicken_scaled = np.hstack((df_chicken[:,0].reshape(-1,1), df_chicken_scaled))
```

Refer to the [introduction](#1-introduction) section above for further information about the despiking process.

### 2.4. Noise Filtering

Filtering is conducted to remove the noise in the dataset. The method used in this simulation is the `Savitzky-Golay` method. The parameters used in this method are selected manually, based on the observation of the dataset.
These filtering processes are done only to get the mean spectra of the dataset.

```python
from scipy.signal import savgol_filter

chicken = np.mean(df_chicken_scaled[:, 1:], axis=1)
chicken_filtered = savgol_filter(chicken, 20, 3)
```

Refer to the [introduction](#1-introduction) section above for further information about the despiking process.

### 2.5. Generating Output

The preprocessed data is saved in the selected path. For pure fat, the dataset consists of the whole spectra with shape of `(m, n+1)` where `n+1` consists of the wavenumbers and the intensity of `n` spectra, and `m` is the number of wavenumbers.

```python
wavenumber = chicken[:, 0].values
chicken_spectra = chicken.iloc[:, 1:].T
```

The mean spectra consists of the wavenumbers and the intensity of the mean spectra `(m, 2)`. The following code is used to save the preprocessed data:

```python
if not os.path.exists('../data/Data Lemak/preprocessed/pure fat/chicken/'):
    os.makedirs('../data/Data Lemak/preprocessed/pure fat/chicken/')

pd.DataFrame(df_chicken_scaled).to_csv(
    '../data/Data Lemak/preprocessed/pure fat/chicken/chicken_all.csv',
    index=False, header=False
)
pd.DataFrame([df_chicken[:,0], chicken_filtered]).T.to_csv(
    '../data/Data Lemak/preprocessed/pure fat/chicken/chicken_mean.csv',
    index=False, header=False
)
```

## 3. Mean Analysis

> This section is used to observe the mean of each fatty acid in the dataset and to observe the differences between the fatty acids. **This section is not used and related to the preprocessing steps.**

The mean analysis is conducted to observe the mean of each fatty acid in the dataset and to observe the differences between the fatty acids. The mean spectra is obtained from the preprocessed dataset. The following code is used to plot the mean spectra:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(chicken['Wavenumbers'], chicken['Intensity'], label='Chicken')
plt.plot(cow['Wavenumbers'], cow['Intensity'], label='Cow')
plt.plot(duck['Wavenumbers'], duck['Intensity'], label='Duck')
plt.plot(goat['Wavenumbers'], goat['Intensity'], label='Goat')
plt.plot(pig_b['Wavenumbers'], pig_b['Intensity'], label='Pig B')
plt.plot(pig_p['Wavenumbers'], pig_p['Intensity'], label='Pig P')
plt.plot(pig_rj['Wavenumbers'], pig_rj['Intensity'], label='Pig RJ')
plt.plot(pig_s['Wavenumbers'], pig_s['Intensity'], label='Pig S')
plt.legend()
plt.title('Pure Fat')
plt.xlabel('Wavenumbers')
plt.ylabel('Intensity')
plt.show()
```
