import pandas as pd
import yaml
import warnings

# Example: Load data from the config
def load_spectra(config, key, data_type="all"):
    if data_type == "all":
        file_path = config[key]['all_spectrum']
    elif data_type == "mean":
        file_path = config[key]['mean_spectrum']
    return pd.read_csv(file_path, header=None)

def load_config(file_path):
    try:
        with open(file_path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        warnings.warn('Config file not found', UserWarning)
        exit()
    except yaml.YAMLError:
        warnings.warn('Error reading the config file', UserWarning)
        exit()
    return None

def fill_concentration(df, name, names, name2=None, val=1):
    for n in names:
        if n == name:
            df[n] = val
        elif name2 != None and n == name2:
            df[n] = 1-val
        else:
            df[n] = 0

def load_data(config, type='pure', data_type='all'):
    data = {}
    try:
        for k in config['data'][type].keys():
            if k == 'types':
                continue
            data[k] = load_spectra(config['data'][type], k, data_type)
    except KeyError:
        warnings.warn('Error in the config file', UserWarning)
        exit()
    return data

def fill_concentration_to_data(data, config, key='pure'):
    data_cols = config['data'][key]['types']
    new_data = data.copy()

    if key == 'pure':
        wavenumbers = data[data_cols[0]].iloc[:, 0].values
        for key in new_data.keys():
            new_data[key] = new_data[key].iloc[:, 1:].T
            fill_concentration(new_data[key], key, data_cols)
    elif key == 'mixed':
        first_key = list(data.keys())[0]
        wavenumbers = data[first_key].iloc[:, 0].values
        for key in new_data.keys():
            pure_fats_1, pure_fats_2, c_1, c_2 = key.split(',')
            c_1, c_2 = float(c_1)/100, float(c_2)/100
            new_data[key] = new_data[key].iloc[:, 1:].T
            fill_concentration(new_data[key], pure_fats_1, data_cols, pure_fats_2, c_1)
    else:
        raise ValueError('Invalid key')
    
    # new_data = pd.concat(new_data.values(), axis=0)
    df = pd.DataFrame()
    for k, v in new_data.items():
        added_df = pd.DataFrame(v)
        added_df["label"] = k
        df = pd.concat([df, added_df], axis=0)
    return df, wavenumbers

if __name__ == '__main__':
    config = load_config('fatty_acid/file.yaml')
    pure_data = load_data(config, key='pure')
    print("Pure data loaded with keys: ", pure_data.keys())