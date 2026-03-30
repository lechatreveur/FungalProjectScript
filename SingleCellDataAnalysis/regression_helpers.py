#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 13:51:34 2025

@author: user
"""

import os
import pandas as pd
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import classification_report

def run_lasso_on_clades(df_norm, target_series, clade_cell_ids, features, output_path):
    """
    Run Lasso regression separately for each clade.
    """
    rows = []

    for clade_id, cell_ids in clade_cell_ids.items():
        X = df_norm.loc[cell_ids, features]
        y = target_series.loc[cell_ids]

        model = LassoCV(cv=5, random_state=42).fit(X, y)

        row = {
            'clade_id': clade_id,
            'intercept': model.intercept_,
            'r_squared': model.score(X, y),
            'alpha': model.alpha_
        }
        row.update({feat: coef for feat, coef in zip(features, model.coef_)})
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df = results_df[['clade_id', 'r_squared', 'alpha', 'intercept'] + features]
    results_df.to_csv(output_path, index=False)
    print(f"✅ Saved clade-wise Lasso regression results to:\n{output_path}")
    return results_df


def run_global_lasso(df_norm, target_series, features, output_path):
    """
    Run Lasso regression across all cells.
    """
    common_cells = df_norm.index.intersection(target_series.index)
    X = df_norm.loc[common_cells, features]
    y = target_series.loc[common_cells]

    model = LassoCV(cv=5, random_state=42).fit(X, y)

    row = {
        'intercept': model.intercept_,
        'r_squared': model.score(X, y),
        'alpha': model.alpha_,
        **{feat: coef for feat, coef in zip(features, model.coef_)}
    }

    results_df = pd.DataFrame([row])
    results_df = results_df[['r_squared', 'alpha', 'intercept'] + features]
    results_df.to_csv(output_path, index=False)
    print(f"✅ Saved global Lasso regression result to:\n{output_path}")
    return results_df


def run_logistic_l1(df_norm, binary_labels, features, output_path):
    """
    Run L1-penalized logistic regression and save coefficients.
    """
    common_cells = df_norm.index.intersection(binary_labels.index)
    X = df_norm.loc[common_cells, features]
    y = binary_labels.loc[common_cells]

    model = LogisticRegressionCV(
        cv=5, penalty='l1', solver='saga',
        scoring='accuracy', random_state=42, max_iter=10000
    ).fit(X, y)

    coefs = pd.Series(model.coef_[0], index=features)
    results_df = pd.DataFrame([{
        'intercept': model.intercept_[0],
        'accuracy': model.score(X, y),
        **coefs.to_dict()
    }])
    results_df.to_csv(output_path, index=False)

    print(f"✅ Saved logistic regression results to:\n{output_path}")
    print("\n📄 Classification Report:")
    print(classification_report(y, model.predict(X)))

    return results_df, model
