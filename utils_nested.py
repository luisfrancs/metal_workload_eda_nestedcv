## VERSION 13 March
## AUTHOR Luis Sigcha

import pandas as pd
import numpy as np

from metrics_ML_binary import print_Results_Table,performance_evaluation,eerThreshold

import scipy.io
from scipy import stats
from scipy import signal
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.model_selection import GroupShuffleSplit

def evaluate_model(model, Xt, y_target, fold):
    probas = model.predict_proba(Xt)
    optimal_threshold = eerThreshold(y_target, probas[:, 1])
    _, results_metrics = performance_evaluation( y_target, probas[:, 1], optimal_threshold )
    pred = (probas[:, 1] >= optimal_threshold).astype(int)
    print(f"\nFold {fold}")
    print(classification_report(y_target, pred))
    metrics = {"fold": fold,   "accuracy": results_metrics[0],  "balanced_accuracy": balanced_accuracy_score(y_target, pred),
            "sensitivity": results_metrics[1], "specificity": results_metrics[2], "precision": results_metrics[3],
            "macro_f1": results_metrics[4], "AUC": results_metrics[5], "AUC_PR": results_metrics[4] }
    return metrics

def evaluate_best_model(best_model, best_index, X_target, y_target):
    # Generate predictions
    probas = best_model.predict_proba(X_target)
    optimal_threshold = eerThreshold(y_target, probas[:, 1])
    _, results_metrics = performance_evaluation(y_target, probas[:, 1], optimal_threshold)
    return optimal_threshold, results_metrics, probas

def mean_sd_table(df, exclude_cols=None, decimals=3):
    if exclude_cols is not None:
        df = df.drop(columns=exclude_cols)
    stats = df.agg(['mean', 'std'])
    formatted = stats.loc['mean'].combine(
        stats.loc['std'],
        lambda m, s: f"{m:.{decimals}f} ({s:.{decimals}f})"    )
    return formatted.to_frame().T

def get_metrics_df(best_metrics):
    data = {"accuracy": best_metrics[0], "balanced_accuracy": np.nan, "sensitivity": best_metrics[1], "specificity": best_metrics[2],
            "precision": best_metrics[3], "macro_f1": best_metrics[4],"AUC": best_metrics[5], "AUC_PR": best_metrics[6]  }
    return pd.DataFrame([data])


def nested_grouped_training_pipeline(
    X_source, y_source, idLOSO_source,
    X_target, y_target, idLOSO_target,
    pipeline_factory,
    param_grid,
    eval_model_fn=None,
    n_splits_outer=10,
    n_splits_inner=3,
    target_test_size=0.3):

    outer_results = []
    models, metrics_source, metrics_target = [], [], []

    # -----------------------------
    # 1. Split target dataset ONCE
    # -----------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=target_test_size, random_state=42)
    align_idx, test_idx = next(gss.split(X_target, y_target, groups=idLOSO_target))
    target_split = { "align_idx": align_idx,  "test_idx": test_idx} #Store Targets idxs
    Xt_align = X_target[align_idx]
    Xt_test  = X_target[test_idx]
    yt_test  = y_target[test_idx]

    print(f"Target alignment samples: {Xt_align.shape[0]}")
    print(f"Target test samples: {Xt_test.shape[0]}")

    # -----------------------------
    # 2. Nested CV on source
    # -----------------------------
    outer_gkf = GroupKFold(n_splits=n_splits_outer)

    for fold, (train_idx, test_idx_source) in enumerate(
        outer_gkf.split(X_source, y_source, groups=idLOSO_source)   ):
        # Source split
        Xs_train_outer = X_source[train_idx]
        Xs_test_outer  = X_source[test_idx_source]

        ys_train_outer = y_source[train_idx]
        ys_test_outer  = y_source[test_idx_source]

        groups_inner = idLOSO_source[train_idx]

        # ------------------------------------
        # 3. Create pipeline
        # ------------------------------------
        pipeline = pipeline_factory()

        inner_cv = GroupKFold(n_splits=n_splits_inner)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="f1_macro",
            n_jobs=-1  )

        grid_search.fit(Xs_train_outer, ys_train_outer, groups=groups_inner)

        best_model = grid_search.best_estimator_

        if eval_model_fn is None:
            eval_model = best_model
        else:
            eval_model = eval_model_fn(best_model)

        # -----------------------------
        # 4. Source evaluation
        # -----------------------------
        print("Source metrics:")
        source_metrics = evaluate_model(
            eval_model,
            Xs_test_outer,
            ys_test_outer,
            fold        )

        # -----------------------------
        # 5. Target evaluation
        # -----------------------------
        print("Target metrics:")
        target_metrics = evaluate_model(
            eval_model,
            Xt_test,
            yt_test,
            fold        )

        outer_results.append({
            "fold": fold,
            "best_params": grid_search.best_params_,
            "source_f1": source_metrics.get("macro_f1"),
            "target_f1": target_metrics.get("macro_f1"),
            "source_acc": source_metrics.get("accuracy"),
            "target_acc": target_metrics.get("accuracy")        })

        models.append(best_model)
        metrics_source.append(source_metrics)
        metrics_target.append(target_metrics)

        print(f"Fold {fold} complete. Best Params: {grid_search.best_params_}")

    return pd.DataFrame(outer_results), models, metrics_source, metrics_target, target_split

