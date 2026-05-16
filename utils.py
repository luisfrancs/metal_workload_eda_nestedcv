## VERSION 17 February 2026
## AUTHOR Luis Sigcha

import numpy as np
import pandas as pd
import seaborn as sns
from numpy import where
#from tensorflow import keras
#from keras.utils import to_categorical
from sklearn import metrics
from sklearn.utils import class_weight
from collections import Counter
#import tensorflow as tf
#from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import sklearn
from sklearn.model_selection import train_test_split
from numpy import argmax
from sklearn.metrics import (roc_auc_score,accuracy_score,confusion_matrix, precision_recall_curve, auc,roc_curve, recall_score, classification_report, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold

#Data procesing

def sliding_window_triaxial(signal, window_size, step_size):
    """
    Generate sliding windows from a triaxial time series signal.
    
    Parameters
    ----------
    signal : array-like, shape (N, 3)
        Input triaxial signal (e.g., accelerometer data with x, y, z axes).
    window_size : int
        Length of each window in samples.
    step_size : int
        Step size (stride) between the starting indices of consecutive windows.

    Returns
    -------
    windows : ndarray, shape (num_windows, window_size, 3)
        Array of sliding windows extracted from the input signal.
        Only fully contained windows are returned (no padding).
    """
    # Calculate the number of windows
    num_windows = (signal.shape[0] - window_size) // step_size + 1
    if num_windows <= 0:
        raise ValueError("Invalid combination of window size and step size for the given signal length.")
    # Create sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(signal, (window_size, 3))
    # Extract and reshape windows to remove the singleton dimension
    windows = windows[::step_size, :, :]  # Select windows with the specified step size
    windows = windows.reshape(-1, window_size, 3)  # Reshape to (num_windows, window_size, 3)
    return windows

def sliding_window_vector(signal, window_size, step_size):
    """
    Generate sliding windows from a unidimensional time series signal (no padding is applied).

    Parameters
    ----------
    signal : array-like, shape (N,) or (N, 1)
        Input unidimensional signal.
    window_size : int
        Length of each window in samples.
    step_size : int
        Step size (stride) between the starting indices of consecutive windows.

    Returns
    -------
    windows : ndarray, shape (num_windows, window_size)
        Array of sliding windows extracted from the input signal.
    """
    # Ensure input is a NumPy array (accepts lists, tuples, pandas Series, etc.)
    signal = np.asarray(signal)
    # If input is a 2D column vector with shape (N, 1), squeeze it to a 1D vector (N,)
    if signal.ndim == 2 and signal.shape[1] == 1:
        signal = signal[:, 0]
    # Validate that the result is a 1D vector; otherwise, the windowing logic is not well-defined
    if signal.ndim != 1:
        raise ValueError(f"Expected vector of shape (N,) or (N,1). Got {signal.shape}.")
    # Compute how many windows of length `window_size` fit when moving by `step_size`
    num_windows = (signal.shape[0] - window_size) // step_size + 1
    # If no valid windows fit, raise an error (e.g., window_size > signal length, or invalid parameters)
    if num_windows <= 0:
        raise ValueError("Invalid combination of window size and step size for the given signal length." )
    # Create a view of all contiguous windows of length `window_size` with step 1
    # Output shape: (N - window_size + 1, window_size)
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_size) # This is a *view* (no copy) when possible.
    # Subsample the windowed view so that consecutive windows start `step_size` samples apart,
    #effectively enforcing the desired stride between overlapping windows.
    windows = windows[::step_size]
    # Return array of windows with shape: (num_windows, window_size)
    return windows

#Data handling

def multilabel_selector(index_array, selected_ids):
    res = []
    for currentindex in selected_ids:
        ind = np.where(index_array == currentindex)
        res.append(ind[0])
    selectecIndexes = np.concatenate(res,axis=0)
    selectecIndexes = np.sort(selectecIndexes)#sort list to preserve idx order
    return selectecIndexes

def valid_data(valid_data):
    mask = (valid_data[:, 0] >= 0) #ADAPTED TO DAPTNET
    return mask

def select_valid_data(X_3D,y,id):
    valid_mask=valid_data(y)
    X_nu=X_3D[valid_mask,:,:]#
    y_nu=y[valid_mask,:] #
    id_nu=id[valid_mask,:]
    return X_nu,y_nu,id_nu
   
def printLabel_distribution(y, plot_title="Class distribution", label_map=None):
    df = pd.DataFrame(y)
    count_classes = df[0].value_counts().sort_index()
    # If a label map is provided, apply it to the index (class labels)
    if label_map:
        count_classes.index = count_classes.index.map(label_map)
    count_classes.plot(kind='bar', rot=0)
    plt.title(plot_title)
    plt.xlabel("Physical activity")
    plt.ylabel("Number of observations")
    plt.show()    

def get_best_model(main_results_repeated, models, metric_column=3):
    #get best model in a list of models based in a metric 3=Fscore
    scores=main_results_repeated['Test'][:,metric_column]#get fscores
    max_index = np.argmax(scores)
    print('Best model index:', max_index)
    print('Best F-score:',scores[max_index])
    return models[max_index]