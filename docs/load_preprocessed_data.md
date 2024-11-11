# Load Preprocessed Data

Table of Contents:
- [Configuration File](#1-configuration-file)
- [Load Preprocessed Data](#2-load-preprocessed-data)
- [Fill Concentration Label to Data](#3-fill-concentration-label-to-data)
- [Clean the Data](#4-clean-the-data)
- [Output](#5-output)

Glossary:

- Pure: Data that contains only one type of fatty acid (100% concentration)
- Mixed: Data that contains a mixture of two types of fatty acids
- Mask: A filter to remove unwanted data from the dataset

## 1. Configuration File

After the data has been preprocessed (using [preprocess module](/docs/preprocess.md)), the next step is to load the preprocessed data into the Python environment. The configuration file is used to load the preprocessed data. The configuration file contains the file path of the preprocessed data and the mask to remove unwanted data from the dataset.

The following yaml snippet shows the structure of the preprocessed data configuration file:

```yaml
data:
  pure:
    types:
      - type of fatty acid, e.g. chicken
      - ...
    fatty_acid:
      files: file_path
    ...

  mixed:
    types:
      - type of all primary fatty acid, e.g. chicken
    fatty_acid_1,fatty_acid_2,percentage_1,percentage_2:
      files: file_path
    ...

mask_to_remove:
  pure:
    fatty_acid: [index]
    ...
  mixed:
    fatty_acid_1,fatty_acid_2,percentage_1,percentage_2: [index]
    ...
```

you can see the full example of the configuration file [here](/example/file.yaml).

To load the configuration file, you can use the following code:

```python
from fatty_acid.load_spectra_data import load_config

config = load_config('file.yaml')
```

## 2. Load Preprocessed Data

The preprocessed data is loaded using the `load_data` function. The function takes two arguments: the configuration file and the type of data (pure or mixed). The function returns the data and the wavenumbers.

The following code shows how to load the preprocessed data:

```python
from fatty_acid.load_spectra_data import load_data

pure_data = load_data(config, 'pure')
mixed_data = load_data(config, 'mixed')
```

To get the list of fatty acids, you can use the following code:

```python
pure_fatty_cols = config['data']['pure']['types']
mixed_fatty_cols = config['data']['mixed']['types']
```

## 3. Fill Concentration Label to Data

The `fill_concentration_to_data` function is used to fill the concentration label to the data. The function takes three arguments: the data, the configuration file, and the type of data (pure or mixed). The function returns the data and the wavenumbers.

The following code shows how to fill the concentration label to the data:

```python
from fatty_acid.load_spectra_data import fill_concentration_to_data

pure_spectra_data, wavenumbers = fill_concentration_to_data(pure_data, config, 'pure')
mixed_spectra_data, wavenumbers = fill_concentration_to_data(mixed_data, config, 'mixed')
```

## 4. Clean the Data

By using the mask to remove unwanted data from the dataset, the data can be cleaned. The mask is a filter to remove unwanted data from the dataset by its index. The following code shows how to clean the data:

```python
try:
    mask_to_remove = config['mask_to_remove']
    cleaned_pure_spectra_data = pure_spectra_data.copy()
    cleaned_mixed_spectra_data = mixed_spectra_data.copy()
    for key, value in mask_to_remove['pure'].items():
        cleaned_pure_spectra_data = cleaned_pure_spectra_data.loc[~((cleaned_pure_spectra_data["label"] == key) & (cleaned_pure_spectra_data.index.isin(value)))]
    for key, value in mask_to_remove['mixed'].items():
        cleaned_mixed_spectra_data = cleaned_mixed_spectra_data.loc[~((cleaned_mixed_spectra_data["label"] == key) & (cleaned_mixed_spectra_data.index.isin(value)))]
except KeyError:
    cleaned_pure_spectra_data = pure_spectra_data
    cleaned_mixed_spectra_data = mixed_spectra_data
```

## 5. Output

The output of the data loading process is the wavenumbers, the list of fatty acids, the cleaned pure data, the cleaned mixed data, and the total data. The following code shows how to output the data:

```python
def clean(config):
    # ...
    all_data = pd.concat([pure_spectra_data, mixed_spectra_data], axis=0)
    return (
        wavenumbers, 
        pure_fatty_cols, 
        mixed_fatty_cols, 
        cleaned_pure_spectra_data, 
        cleaned_mixed_spectra_data, 
        all_data
    )
