# ML METRICS BINARY Date: 29 January 2026
# Author: Luis Sigcha
#Funtion to support "training_ML_binary"

import sklearn
import pandas as pd
from sklearn.metrics import auc,roc_curve,f1_score,confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_auc_score,accuracy_score, recall_score, classification_report, precision_recall_fscore_support

import numpy as np
from numpy import argmax

from scipy.signal import medfilt #TEST


def prcThreshold(labels, predictions):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, predictions)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    return thresholds[ix]

def eerThreshold(labels, y_predict_prob):
    fpr, tpr, thresholds = roc_curve(labels, y_predict_prob)
    roc_auc = auc(fpr, tpr)
    print('roc_auc=',  roc_auc)
    fnr = 1 - tpr
    EER = 100*fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    print('Best EER Threshold=',  eer_threshold,' ','EER=', EER)
    return eer_threshold
    
def performance_evaluation(y_test,y_predict_prob,decision_threshold):
    print("Y-shape ",y_test.shape )
    fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(y_test, y_predict_prob)
    auc_pr = auc(recall, precision)
    print(f'Optimal threshold : {decision_threshold:3f}')
    predicted_class = (y_predict_prob >= decision_threshold).astype(int)
    #TEST median filter
    #window_size = 3 
    #print('median filter applied: ',window_size )
    #predicted_class = medfilt(predicted_class, kernel_size=window_size)
    #TEST median filter
    accu=accuracy_score(y_test, predicted_class)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_class).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    precision = tp/(tp+fp)
    GM = (specificity*sensitivity)**(1/2)
    f1=f1_score(y_test, predicted_class)
    print('\n')
    print("accuracy: ",accu)
    print(confusion_matrix(y_test, predicted_class))
    print(" ")
    print("sensitivity: ",sensitivity)
    print("specificity: ",specificity)
    #print("GM: ",GM)
    print('precision: ',precision)
    print("F1-score: ",f1)
    print("AUC: ",roc_auc)
    print("AUC-PR: ",auc_pr)
    results=[accu,sensitivity,specificity,precision,f1,roc_auc,auc_pr]
    return decision_threshold,results
    
def predictAndMetrics(model,X, y, threshold=0.5):
    '''Evaluate Performance'''
    y_predict_train = model.predict_proba(X)
    y_predict_train=y_predict_train[:,1]#get only positive predictions
    _,results_metrics=performance_evaluation(y,y_predict_train,threshold)
    return results_metrics

# Results Metrics summary    
def printsummary_Results(results_np_matrix,print_mean=True):
    df = pd.DataFrame(results_np_matrix,columns=['Accuracy','Sensitivity', 'Specificity', 'Precision','Fscore','AUC','AUCPRC'])#, results_val,results_test)
    summary_mean=df.mean()
    summary_std=df.std()
    if print_mean==True:
        res=summary_mean
    else:
        res=summary_std
    df=res.to_frame().T
    df = df.reset_index(drop=True)
    return df

def print_Results_Table(main_results_repeated, show_std=True):
    """
    Parameters
    ----------
    main_results_repeated : dict
        Dictionary with 'Train', and 'Test' results.
    show_std : bool, default=True
        If True, display mean (std). If False, display only mean.
    """
    # --- mean values ---
    df1 = printsummary_Results(main_results_repeated['Train'], print_mean=True)
    df2 = printsummary_Results(main_results_repeated['Test'], print_mean=True)
    dfRes_mean = pd.concat([df1, df2])

    if show_std:
        # --- std values ---
        df1 = printsummary_Results(main_results_repeated['Train'], print_mean=False)
        df2 = printsummary_Results(main_results_repeated['Test'], print_mean=False)
        dfRes_std = pd.concat([df1, df2])
        # --- combine mean and std ---
        df_mean, df_std = dfRes_mean.align(dfRes_std, join="outer")
        df_combined = df_mean.applymap(lambda x: "{:.3f}".format(x)) + " (" + df_std.applymap(lambda x: "{:.5f}".format(x)) + ")"
    else:
        # Only mean
        df_combined = dfRes_mean.applymap(lambda x: f"{x:.3f}")
    # Insert Subset column
    df_combined.insert(0, "Subset", ["Train", "Test"])
    # Style for display
    df_combined = df_combined.style.hide(axis="index")
    return df_combined
