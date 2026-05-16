# TRAINING ML BINARY Date: 29 January 2026
# Author: Luis Sigcha

import numpy as np
import pandas as pd
import seaborn as sns
from numpy import where
#from keras.utils import to_categorical
from sklearn import metrics
from collections import Counter
from sklearn.model_selection import GroupKFold
from collections import Counter

import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from numpy import argmax
from metrics_ML_binary import performance_evaluation, prcThreshold, eerThreshold, predictAndMetrics

#Metrics on subsets

def metricsOnSubsets_EER(model,dataTrain, labelTrain, dataTrain_val, labelTrain_val, X_test3D, y_test):
    '''Evaluate Performance'''
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")    
    print('EER Threshold metrics')
    print('\nTRAIN')
    y_predict_train = model.predict_proba(dataTrain)
    print('AQUI_y_predict_train.shape', y_predict_train.shape)
    y_predict_train=y_predict_train[:,1]#get only positive predictions
    optimalThresold_TRAIN=eerThreshold(labelTrain, y_predict_train)
    _,results_train = performance_evaluation(labelTrain, y_predict_train,optimalThresold_TRAIN)
    '''Test performance'''
    print('\nVALIDATION')
    y_predict_val= model.predict_proba(dataTrain_val)
    y_predict_val=y_predict_val[:,1]#get only positive predictions
    optimalThresold_VAL = eerThreshold(labelTrain_val, y_predict_val)
    _,results_val = performance_evaluation(labelTrain_val, y_predict_val,optimalThresold_VAL)
    print('\nTEST')
    results_test=predictAndMetrics(model,X_test3D,y_test,optimalThresold_TRAIN)
    '''Save results'''
    results_metrics= {'Train':results_train , 'Validation': results_val, 'Test': results_test}
    return optimalThresold_TRAIN,optimalThresold_VAL,results_metrics

def metricsOnSubsets_PRC(model,dataTrain, labelTrain, dataTrain_val, labelTrain_val, X_test3D, y_test):
    '''Evaluate Performance'''
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")    
    print('PRC Threshold metrics')
    print('\nTRAIN')
    y_predict_train = model.predict_proba(dataTrain)
    y_predict_train=y_predict_train[:,1]#get only positive predictions
    optimalThresold_TRAIN=prcThreshold(labelTrain, y_predict_train)
    _,results_train = performance_evaluation(labelTrain, y_predict_train,optimalThresold_TRAIN)
    '''Test performance'''
    print('\nVALIDATION')
    y_predict_val= model.predict_proba(dataTrain_val)
    y_predict_val=y_predict_val[:,1]#get only positive predictions
    optimalThresold_VAL = prcThreshold(labelTrain_val, y_predict_val)
    _,results_val = performance_evaluation(labelTrain_val, y_predict_val,optimalThresold_VAL)
    print('\nTEST')
    results_test=predictAndMetrics(model,X_test3D,y_test,optimalThresold_TRAIN)
    '''Save results'''
    results_metrics= {'Train':results_train , 'Validation': results_val, 'Test': results_test}
    return optimalThresold_TRAIN,optimalThresold_VAL,results_metrics

# Train model

def trainModel_ML(model, X_train3D, y_train,X_train3D_val,y_train_val, randomState=0):
    model.set_params(random_state=randomState)#set a different random state per iteration
    model.fit(X_train3D, y_train.ravel())
    val_accuracy=model.score(X_train3D_val,y_train_val)
    return model,val_accuracy
    
## Group K-fold
def _safe_index(X, idx):
    """Index numpy arrays or pandas DataFrame/Series safely."""
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]

def _to_1d_groups(groups):
    """Force groups into a 1D numpy array of hashable subject IDs."""
    if hasattr(groups, "to_numpy"):
        g = groups.to_numpy()
    else:
        g = np.asarray(groups)

    if g.ndim == 2 and g.shape[1] == 1:
        g = g[:, 0]
    if g.ndim != 1:
        raise ValueError(f"`groups` must be 1D, got shape {g.shape}")
    return g

def kfold_group_test_EE_PR(model_base, X, y, groups, n_splits=5, base_random_state=5):
    """
    GroupKFold CV with non-overlapping subjects.
    Only TRAIN and TEST sets (no validation).
    Both EER and PRC metrics are computed and stored.
    """
    y_arr = _safe_index(y, np.arange(len(y))) if hasattr(y, "iloc") else np.asarray(y)
    groups_1d = _to_1d_groups(groups)

    gkf = GroupKFold(n_splits=n_splits)

    results_train_EER = np.zeros((n_splits, 7))
    results_test_EER  = np.zeros((n_splits, 7))
    results_train_PRC = np.zeros((n_splits, 7))
    results_test_PRC  = np.zeros((n_splits, 7))

    trained_models = []

    for fold, (tr_idx, te_idx) in enumerate(
        gkf.split(np.zeros(len(groups_1d)), y_arr, groups_1d)
    ):
        print("---------------------------------------------------------------")
        print(f"Fold {fold + 1}/{n_splits}")

        # === TEST SUBJECT SUMMARY ===
        test_subjects = np.unique(groups_1d[te_idx])
        print(f"Test subjects ({len(test_subjects)}): {test_subjects.tolist()}")
        # This code shows subject IDs and sample indices used in TEST for each fold.
        #print("Test subject → sample indices:")
        #for subj in test_subjects:
        #    subj_idx = te_idx[groups_1d[te_idx] == subj]
        #    print(f"  Subject {subj}: {subj_idx.tolist()}")
        # Optional compact count summary
        counts = Counter(groups_1d[te_idx])
        print("Observations per test subject:", dict(counts))

        X_train = _safe_index(X, tr_idx)
        y_train = _safe_index(y, tr_idx)
        X_test  = _safe_index(X, te_idx)
        y_test  = _safe_index(y, te_idx)

        # Fresh model instance (same logic as repeated holdout)
        model_instance = model_base.__class__(**model_base.get_params())

        # Train model (dummy validation = train)
        m_model, _ = trainModel_ML( model_instance,  X_train, y_train, X_train, y_train, randomState=base_random_state + fold )
        trained_models.append(m_model)

        # ---- METRICS ----
        _, _, metricsEER = metricsOnSubsets_EER( m_model, X_train, y_train, X_train, y_train,  X_test, y_test )
        _, _, metricsPRC = metricsOnSubsets_PRC( m_model, X_train, y_train, X_train, y_train,  X_test, y_test )

        results_train_EER[fold] = metricsEER["Train"]
        results_test_EER[fold]  = metricsEER["Test"]
        results_train_PRC[fold] = metricsPRC["Train"]
        results_test_PRC[fold]  = metricsPRC["Test"]

    mainresults = {
        "EER": {
            "Train": results_train_EER,
            "Test":  results_test_EER },
        "PRC": {
            "Train": results_train_PRC,
            "Test":  results_test_PRC}
        }

    return mainresults, trained_models
