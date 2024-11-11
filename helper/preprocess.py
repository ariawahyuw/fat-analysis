from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import pandas as pd
import os

# Baseline estimation function:
def baseline_als(y, lam, p, niter=100):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def modified_z_score(ys):
    ysb = np.diff(ys) # Differentiated intensity values
    median_y = np.median(ysb) # Median of the intensity values
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ysb]) # median_absolute_deviation of the differentiated intensity values
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in ysb] # median_absolute_deviationmodified z scores
    return modified_z_scores
    
# The next function calculates the average values around the point to be replaced.
def fixer(y,ma, threshold=7, epsilon=np.finfo(float).eps):
    spikes = np.abs(np.array(modified_z_score(y))) > threshold
    y_out = y.copy()
    for i in np.arange(len(spikes)):
        if spikes[i] != 0:
            min_idx = max(0,i-ma)
            max_idx = min(len(spikes),i+ma)
            w = np.arange(min_idx,max_idx)
            we = w[(spikes[w] < epsilon)]
            y_out[i] = np.mean(y[we])
    return y_out

# 
def split_data(data, name, path, subpath, first_idx=1804.50391):
    '''
        Example:
        - name = 'chicken'
        - path = './Data Lemak/pure fat/'
        - subpath = '/area '
    '''
    for k in range(3):
        df_split = pd.DataFrame(data[k])
        index_split = df_split[df_split[0] == first_idx].index
        for i in range(0, len(index_split)):
            if i == len(index_split)-1:
                file = df_split.iloc[index_split[i]:]
            else:
                file = df_split.iloc[index_split[i]:index_split[i+1]]
            if os.path.exists(path + name + subpath + str(k+1)) == False:
                os.mkdir(path + name + subpath + str(k+1))
            file.to_csv(path + name + subpath + str(k+1) + '/' + str(i+1) + '.txt', index=None, header=None, sep='\t')
        
def read_data(path, file_name=''):
    # Get all files in the folder
    files = os.listdir(path)
    # Read all files
    data = []
    for file in files:
        if file.endswith('.txt') and file.startswith(file_name):
            data.append(np.genfromtxt(path + file, delimiter='\t'))
    return data