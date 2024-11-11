import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulasi_data_lemak.fatty_acid.load_spectra_data import load_config, load_data, fill_concentration_to_data

"""
Glossary:
- Pure: Data that contains only one type of fatty acid (100% concentration)
- Mixed: Data that contains a mixture of two types of fatty acids
- Cleaned: Data that has been processed to remove unwanted data
- Mask: A filter to remove unwanted data from the dataset
- Pred: Predicted
- True: Actual
"""

def clean(config):
    #Load the configuration and file locations from the YAML file
    pure_data = load_data(config, 'pure')
    mixed_data = load_data(config, 'mixed')

    pure_fatty_cols = config['data']['pure']['types']
    mixed_fatty_cols = config['data']['mixed']['types']

    pure_spectra_data, wavenumbers = fill_concentration_to_data(pure_data, config, 'pure')
    mixed_spectra_data, _ = fill_concentration_to_data(mixed_data, config, 'mixed')

    try:
        mask_to_remove = config['mask_to_remove']
        # Cleaning pure_spectra_data and mixed_spectra_data by its relative fatty index
        cleaned_pure_spectra_data = pure_spectra_data.copy()
        cleaned_mixed_spectra_data = mixed_spectra_data.copy()
        for key, value in mask_to_remove['pure'].items():
            cleaned_pure_spectra_data = cleaned_pure_spectra_data.loc[~((cleaned_pure_spectra_data["label"] == key) & (cleaned_pure_spectra_data.index.isin(value)))]
        for key, value in mask_to_remove['mixed'].items():
            cleaned_mixed_spectra_data = cleaned_mixed_spectra_data.loc[~((cleaned_mixed_spectra_data["label"] == key) & (cleaned_mixed_spectra_data.index.isin(value)))]
    except KeyError:
        cleaned_pure_spectra_data = pure_spectra_data
        cleaned_mixed_spectra_data = mixed_spectra_data

    all_data = pd.concat([pure_spectra_data, mixed_spectra_data], axis=0)
    return wavenumbers, pure_fatty_cols, mixed_fatty_cols, cleaned_pure_spectra_data, cleaned_mixed_spectra_data, all_data


def main():
    # Mask for cleaning the data by its index
    config = load_config('file.yaml')
    wavenumbers, pure_fatty_cols, mixed_fatty_cols, cleaned_pure_spectra_data, cleaned_mixed_spectra_data, all_data = clean(config)
    print(f"Wavenumbers starts from {wavenumbers[-1]} cm^(-1) to {wavenumbers[0]} cm^(-1), with a total of {len(wavenumbers)} wavenumbers")
    print(f"Pure fatty acids: {pure_fatty_cols}")
    print(f"Mixed fatty acids: {mixed_fatty_cols}")
    print(f"Cleaned pure data: {cleaned_pure_spectra_data.shape}")
    print(f"Cleaned mixed data: {cleaned_mixed_spectra_data.shape}")
    print(f"Total data: {all_data.shape}")

if __name__ == '__main__':
    main()