#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 19:06:24 2025

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import pandas as pd

def fit_constant(t, y):
    c = np.mean(y)
    y_pred = np.full_like(t, c)
    var = np.var(y - y_pred) * np.ones_like(t)
    return y_pred, var, {'c': float(c)}


def fit_linear(t, y):
    reg = LinearRegression().fit(t, y)
    a = reg.coef_[0][0]
    b = reg.intercept_[0]
    y_pred = reg.predict(t)
    var = np.var(y - y_pred) * np.ones_like(y_pred)
    return y_pred, var, {'a': float(a), 'b': float(b)}


def fit_step_discrete(t, y):
    t = t.flatten()
    y = y.flatten()

    best_loss = np.inf
    best_step = None
    best_c1 = None
    best_c2 = None
    best_pred = None

    for step_idx in range(2, len(t) - 2):  # avoid edges
        step_time = t[step_idx]
        before = y[t < step_time]
        after = y[t >= step_time]
        if len(before) == 0 or len(after) == 0:
            continue
        c1 = np.mean(before)
        c2 = np.mean(after)
        pred = np.where(t < step_time, c1, c2)
        loss = np.mean((y - pred) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_step = step_time
            best_c1 = c1
            best_c2 = c2
            best_pred = pred

    y_pred = best_pred.reshape(-1, 1)
    var = np.var(y - y_pred) * np.ones_like(y_pred)
    return y_pred, var, {'step_time': float(best_step), 'c1': float(best_c1), 'c2': float(best_c2)}


def fmt_params(param_dict):
    return ", ".join(f"{k}={v:.2f}" for k, v in param_dict.items())

def fmt_limited_params(param_dict, max_items=6):
    formatted = []
    for k, v in list(param_dict.items())[:max_items]:
        if isinstance(v, (int, float)):
            formatted.append(f"{k}={v:.2f}")
        else:
            formatted.append(f"{k}={v}")
    return ", ".join(formatted)

def fmt_limited_nested(params, max_items=6):
    lines = [f"{params['model']} (AIC={params['AIC']:.1f})"]
    for group in ['trend_params', 'osc_params']:
        if group in params:
            sub = params[group]
            formatted = []
            for k, v in list(sub.items())[:max_items]:
                if isinstance(v, (int, float)):
                    formatted.append(f"{k}={v:.2f}")
                else:
                    formatted.append(f"{k}={v}")
            lines.append(", ".join(formatted))
    return "\n".join(lines)


from scipy.optimize import curve_fit
# simple sine wave
def sine_func(t, A, f, phi):
    return A * np.sin(2 * np.pi * f * t + phi)

def fit_sine_wave(t, y):
    t = t.flatten()
    y = y.flatten()
    A_guess = (y.max() - y.min()) / 2
    f_guess = 1.0 / (t[-1] - t[0])  # one cycle
    phi_guess = 0

    try:
        popt, _ = curve_fit(sine_func, t, y, p0=[A_guess, f_guess, phi_guess])
        y_pred = sine_func(t, *popt).reshape(-1, 1)
        residuals = y.reshape(-1, 1) - y_pred
        var = np.var(residuals) * np.ones_like(y_pred)
        return y_pred, var, {'A': popt[0], 'f': popt[1], 'phi': popt[2]}
    except RuntimeError as e:
        print(f"❌ Sine fit failed: {e}")
        return np.zeros_like(y).reshape(-1, 1), np.ones_like(y).reshape(-1, 1), {'A': 0, 'f': 0, 'phi': 0}
    
    
def fit_detrend_then_sine(t, y, trend_type='linear'):
    # Map to detrending function
    trend_map = {
        'constant': fit_constant,
        'linear': fit_linear,
        'step': fit_step_discrete
    }
    assert trend_type in trend_map, "trend_type must be 'constant', 'linear', or 'step'"

    # Detrend
    mu_trend, _, trend_params = trend_map[trend_type](t, y)
    y_detrended = y - mu_trend

    # Fit sine to residual
    mu_sine, var_sine, sine_params = fit_sine_wave(t, y_detrended)

    # Recombine
    full_mu = mu_trend + mu_sine
    return full_mu, var_sine, {
        'trend': trend_params,
        'sine': sine_params
    }

# harmonic sine wave
def harmonic_sine(t, A1, phi1, A2, phi2, f):
    t = t.flatten()
    return (
        A1 * np.sin(2 * np.pi * f * t + phi1) +
        A2 * np.sin(2 * np.pi * 2 * f * t + phi2)
    )

def fit_harmonic_sine_wave(t, y):
    t = t.flatten()
    y = y.flatten()

    # Initial parameter guesses
    A1_guess = (np.percentile(y, 95) - np.percentile(y, 5)) / 2
    A2_guess = A1_guess / 2
    f_guess = 1 / (t[-1] - t[0])  # one cycle
    phi1_guess = 0
    phi2_guess = 0

    try:
        popt, _ = curve_fit(
            harmonic_sine,
            t,
            y,
            p0=[A1_guess, phi1_guess, A2_guess, phi2_guess, f_guess],
            maxfev=10000
        )
        y_pred = harmonic_sine(t, *popt).reshape(-1, 1)
        residuals = y.reshape(-1, 1) - y_pred
        var = np.var(residuals) * np.ones_like(y_pred)
        params = {
            'A1': popt[0], 'phi1': popt[1],
            'A2': popt[2], 'phi2': popt[3],
            'f': popt[4]
        }
        return y_pred, var, params
    except RuntimeError as e:
        print(f"❌ Harmonic sine fit failed: {e}")
        return np.zeros_like(y).reshape(-1, 1), np.ones_like(y).reshape(-1, 1), {'A1': 0, 'phi1': 0, 'A2': 0, 'phi2': 0, 'f': 0}


def fit_detrend_then_harmonic(t, y, trend_type='linear'):
    trend_map = {
        'constant': fit_constant,
        'linear': fit_linear,
        'step': fit_step_discrete
    }
    assert trend_type in trend_map

    mu_trend, _, trend_params = trend_map[trend_type](t, y)
    y_detrended = y - mu_trend

    mu_harm, var_harm, harm_params = fit_harmonic_sine_wave(t, y_detrended)
    full_mu = mu_trend + mu_harm

    return full_mu, var_harm, {
        'trend': trend_params,
        'harmonic': harm_params
    }

# model selection with n harmonic
def harmonic_sine_n(t, A1, phi1, A2, phi2, f, n):
    t = t.flatten()
    return (
        A1 * np.sin(2 * np.pi * f * t + phi1) +
        A2 * np.sin(2 * np.pi * n * f * t + phi2)
    )

def fit_harmonic_sine_given_n(t, y, n):
    t = t.flatten()
    y = y.flatten()

    A1_guess = (np.percentile(y, 95) - np.percentile(y, 5)) / 2
    A2_guess = A1_guess / 2
    f_guess = 1 / (t[-1] - t[0])
    phi1_guess = 0
    phi2_guess = 0

    def wrapped_sine(t, A1, phi1, A2, phi2, f):
        return harmonic_sine_n(t, A1, phi1, A2, phi2, f, n)

    try:
        popt, _ = curve_fit(
            wrapped_sine,
            t,
            y,
            p0=[A1_guess, phi1_guess, A2_guess, phi2_guess, f_guess],
            maxfev=10000
        )
        y_pred = wrapped_sine(t, *popt).reshape(-1, 1)
        residuals = y.reshape(-1, 1) - y_pred
        mse = np.mean(residuals**2)
        aic = 2 * len(popt) + len(y) * np.log(mse)
        var = np.var(residuals) * np.ones_like(y_pred)
        return y_pred, var, {
            'A1': popt[0], 'phi1': popt[1],
            'A2': popt[2], 'phi2': popt[3],
            'f': popt[4], 'n': n, 'MSE': mse, 'AIC': aic
        }
    except RuntimeError:
        return None, None, None

def fit_best_harmonic_sine(t, y, n_list=[2, 3, 5]):
    best_aic = np.inf
    best_result = None

    for n in n_list:
        y_pred, var, params = fit_harmonic_sine_given_n(t, y, n)
        if params is not None and params['AIC'] < best_aic:
            best_aic = params['AIC']
            best_result = (y_pred, var, params)

    if best_result is None:
        y_pred = np.zeros_like(y)
        var = np.ones_like(y)
        return y_pred, var, {'A1': 0, 'phi1': 0, 'A2': 0, 'phi2': 0, 'f': 0, 'n': 0, 'MSE': np.nan, 'AIC': np.inf}
    return best_result
def harmonic_sine_3n(t, A1, phi1, A2, phi2, A3, phi3, f, n2, n3):
    t = t.flatten()
    return (
        A1 * np.sin(2 * np.pi * f * t + phi1) +
        A2 * np.sin(2 * np.pi * n2 * f * t + phi2) +
        A3 * np.sin(2 * np.pi * n3 * f * t + phi3)
    )
def fit_harmonic_sine_given_n3(t, y, n2, n3):
    t = t.flatten()
    y = y.flatten()

    amp_range = (np.percentile(y, 95) - np.percentile(y, 5)) / 2
    A1_guess = amp_range
    A2_guess = amp_range / 2
    A3_guess = amp_range / 3
    f_guess = 1 / (t[-1] - t[0])
    phi1_guess = phi2_guess = phi3_guess = 0

    def wrapped_sine(t, A1, phi1, A2, phi2, A3, phi3, f):
        return harmonic_sine_3n(t, A1, phi1, A2, phi2, A3, phi3, f, n2, n3)

    try:
        popt, _ = curve_fit(
            wrapped_sine,
            t,
            y,
            p0=[A1_guess, phi1_guess, A2_guess, phi2_guess, A3_guess, phi3_guess, f_guess],
            maxfev=20000
        )
        y_pred = wrapped_sine(t, *popt).reshape(-1, 1)
        residuals = y.reshape(-1, 1) - y_pred
        mse = np.mean(residuals**2)
        aic = 2 * len(popt) + len(y) * np.log(mse)
        var = np.var(residuals) * np.ones_like(y_pred)
        return y_pred, var, {
            'A1': popt[0], 'phi1': popt[1],
            'A2': popt[2], 'phi2': popt[3],
            'A3': popt[4], 'phi3': popt[5],
            'f': popt[6], 'n2': n2, 'n3': n3,
            'MSE': mse, 'AIC': aic
        }
    except RuntimeError:
        return None, None, None
def fit_best_harmonic_sine_3(t, y, n_pair_list=[(2, 3), (2, 5), (3, 5)]):
    best_aic = np.inf
    best_result = None

    for n2, n3 in n_pair_list:
        y_pred, var, params = fit_harmonic_sine_given_n3(t, y, n2, n3)
        if params is not None and params['AIC'] < best_aic:
            best_aic = params['AIC']
            best_result = (y_pred, var, params)

    if best_result is None:
        y_pred = np.zeros_like(y)
        var = np.ones_like(y)
        return y_pred, var, {
            'A1': 0, 'phi1': 0, 'A2': 0, 'phi2': 0,
            'A3': 0, 'phi3': 0, 'f': 0, 'n2': 0, 'n3': 0,
            'MSE': np.nan, 'AIC': np.inf
        }
    return best_result
def general_sine_3f(t, A1, phi1, f1, A2, phi2, f2, A3, phi3, f3):
    t = t.flatten()
    return (
        A1 * np.sin(2 * np.pi * f1 * t + phi1) +
        A2 * np.sin(2 * np.pi * f2 * t + phi2) +
        A3 * np.sin(2 * np.pi * f3 * t + phi3)
    )
def fit_general_sine_3f(t, y):
    t = t.flatten()
    y = y.flatten()

    amp_range = (np.percentile(y, 95) - np.percentile(y, 5)) / 2
    A1_guess = amp_range
    A2_guess = amp_range / 2
    A3_guess = amp_range / 3
    f_base = 1 / (t[-1] - t[0])
    f1_guess, f2_guess, f3_guess = f_base, 2 * f_base, 3 * f_base
    phi1_guess = phi2_guess = phi3_guess = 0

    def wrapped_sine(t, A1, phi1, f1, A2, phi2, f2, A3, phi3, f3):
        return general_sine_3f(t, A1, phi1, f1, A2, phi2, f2, A3, phi3, f3)

    try:
        popt, _ = curve_fit(
            wrapped_sine,
            t,
            y,
            p0=[A1_guess, phi1_guess, f1_guess,
                A2_guess, phi2_guess, f2_guess,
                A3_guess, phi3_guess, f3_guess],
            bounds=(
                [0, -np.pi, 0, 0, -np.pi, 0, 0, -np.pi, 0],
                [np.inf, np.pi, np.inf, np.inf, np.pi, np.inf, np.inf, np.pi, np.inf]
            ),
            maxfev=30000
        )
        y_pred = wrapped_sine(t, *popt).reshape(-1, 1)
        residuals = y.reshape(-1, 1) - y_pred
        mse = np.mean(residuals**2)
        aic = 2 * len(popt) + len(y) * np.log(mse)
        var = np.var(residuals) * np.ones_like(y_pred)
        return y_pred, var, {
            'A1': popt[0], 'phi1': popt[1], 'f1': popt[2],
            'A2': popt[3], 'phi2': popt[4], 'f2': popt[5],
            'A3': popt[6], 'phi3': popt[7], 'f3': popt[8],
            'MSE': mse, 'AIC': aic
        }
    except RuntimeError:
        return None, None, None


def harmonic_sine_n_terms(t, *params):
    """
    Harmonic sine model with N terms:
    params = [A1, phi1, A2, phi2, ..., AN, phiN, f]
    Frequencies are n_i * f, where n_i = 1, 2, ..., N
    """
    t = t.flatten()
    N = (len(params) - 1) // 2
    f = params[-1]
    result = np.zeros_like(t, dtype=float)

    for i in range(N):
        A = params[2*i]
        phi = params[2*i + 1]
        n = i + 1
        result += A * np.sin(2 * np.pi * n * f * t + phi)

    return result
def fit_harmonic_sine_N(t, y, N=6):
    """
    Fit harmonic sine model with N harmonics: n=1 to N.
    Shared base frequency f is fitted, amplitudes/phases per harmonic.
    """
    t = t.flatten()
    y = y.flatten()

    amp_range = (np.percentile(y, 95) - np.percentile(y, 5)) / 2
    f_guess = 1 / (t[-1] - t[0])

    # Initial guesses and bounds
    p0 = []
    bounds_lower = []
    bounds_upper = []
    for i in range(N):
        A_guess = amp_range / (i + 1)
        phi_guess = 0
        p0 += [A_guess, phi_guess]
        bounds_lower += [0, -np.pi]
        bounds_upper += [np.inf, np.pi]

    p0 += [f_guess]
    bounds_lower += [0]
    bounds_upper += [np.inf]

    try:
        popt, _ = curve_fit(
            harmonic_sine_n_terms,
            t,
            y,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=30000
        )
        y_pred = harmonic_sine_n_terms(t, *popt).reshape(-1, 1)
        residuals = y.reshape(-1, 1) - y_pred
        mse = np.mean(residuals**2)
        aic = 2 * len(popt) + len(y) * np.log(mse)
        var = np.var(residuals) * np.ones_like(y_pred)

        param_dict = {'MSE': mse, 'AIC': aic, 'f': popt[-1]}
        for i in range(N):
            param_dict[f'A{i+1}'] = popt[2*i]
            param_dict[f'phi{i+1}'] = popt[2*i + 1]
            param_dict[f'n{i+1}'] = i + 1

        return y_pred, var, param_dict

    except RuntimeError:
        y_pred = np.zeros_like(y).reshape(-1, 1)
        var = np.ones_like(y_pred)
        return y_pred, var, None
def fit_best_harmonic_sine_by_aic(t, y, N_max=10, N_min=1):
    """
    Try harmonic models with N harmonics from N_min to N_max.
    Select the best one based on AIC.
    """
    best_aic = np.inf
    best_result = None

    for N in range(N_min, N_max + 1):
        y_pred, var, params = fit_harmonic_sine_N(t, y, N=N)
        if params is not None and params['AIC'] < best_aic:
            best_aic = params['AIC']
            best_result = (y_pred, var, params)

    if best_result is None:
        y_pred = np.zeros_like(y).reshape(-1, 1)
        var = np.ones_like(y_pred)
        return y_pred, var, {'MSE': np.nan, 'AIC': np.inf, 'f': 0, 'N': 0}

    y_pred, var, params = best_result
    params['N'] = (len(params) - 3) // 3  # estimate N from #params
    return y_pred, var, params


def fit_detrend_then_best_harmonic(t, y, trend_type='linear', harmonics=[2, 3, 5]):
    trend_map = {
        'constant': fit_constant,
        'linear': fit_linear,
        'step': fit_step_discrete
    }
    assert trend_type in trend_map

    mu_trend, _, trend_params = trend_map[trend_type](t, y)
    y_detrended = y - mu_trend

    #mu_harm, var_harm, harm_params = fit_best_harmonic_sine(t, y_detrended, harmonics)
    mu_harm, var_harm, harm_params = fit_best_harmonic_sine_3(t, y_detrended)#, harmonics)
    full_mu = mu_trend + mu_harm

    return full_mu, var_harm, {
        'trend': trend_params,
        'harmonic': harm_params
    }

# full model selection
def compute_aic(y_true, y_pred, n_params):
    resid = y_true - y_pred
    mse = np.mean(resid**2)
    aic = 2 * n_params + len(y_true) * np.log(mse)
    return aic

def fit_model_constant(t, y):
    c = np.mean(y)
    y_pred = np.full_like(y, c)
    aic = compute_aic(y, y_pred, n_params=1)
    var = np.var(y - y_pred) * np.ones_like(y_pred)
    return y_pred, var, {'model': 'constant', 'c': c, 'AIC': aic}

def fit_model_linear(t, y):
    reg = LinearRegression().fit(t, y)
    a, b = reg.coef_[0][0], reg.intercept_[0]
    y_pred = reg.predict(t)
    aic = compute_aic(y, y_pred, n_params=2)
    var = np.var(y - y_pred) * np.ones_like(y_pred)
    return y_pred, var, {'model': 'linear', 'a': a, 'b': b, 'AIC': aic}

def fit_model_linear_plus_sine(t, y):
    y_trend, _, trend_params = fit_model_linear(t, y)
    y_detrended = y - y_trend
    y_sine, _, sine_params = fit_sine_wave(t, y_detrended)
    y_pred = y_trend + y_sine
    aic = compute_aic(y, y_pred, n_params=5)
    var = np.var(y - y_pred) * np.ones_like(y_pred)
    return y_pred, var, {
        'model': 'linear+sine',
        'AIC': aic,
        'trend_params': trend_params,
        'osc_params': sine_params
    }

def fit_model_linear_plus_harmonic(t, y, N_max=10):
    y_trend, _, trend_params = fit_model_linear(t, y)
    y_detrended = y - y_trend
    #y_harm, _, harm_params = fit_best_harmonic_sine(t, y_detrended, harmonics)
    #y_harm, _, harm_params = fit_best_harmonic_sine_3(t, y_detrended)#, harmonics)
    #y_harm, _, harm_params = fit_general_sine_N(t, y_detrended)
    y_harm, _, harm_params = fit_best_harmonic_sine_by_aic(t, y_detrended, N_max=N_max)#, N=10)

    y_pred = y_trend + y_harm
    aic = compute_aic(y, y_pred, n_params=7)
    var = np.var(y - y_pred) * np.ones_like(y_pred)
    return y_pred, var, {
        'model': 'linear+harmonic',
        'AIC': aic,
        'trend_params': trend_params,
        'osc_params': harm_params
    }


def model_selector_with_threshold(t, y, N_max=10, delta_threshold=4):
    from functools import partial

    # Create partial and manually set name
    fit_harmonic_partial = partial(fit_model_linear_plus_harmonic, N_max=N_max)
    fit_harmonic_partial.__name__ = 'fit_model_linear_plus_harmonic'
    
    models = [
        fit_model_constant,
        fit_model_linear,
        fit_model_linear_plus_sine,
        fit_harmonic_partial
    ]

    results = []
    for fit_func in models:
        try:
            y_pred, var, params = fit_func(t, y)
            results.append({
                'func': fit_func.__name__,
                'y_pred': y_pred,
                'var': var,
                'params': params,
                'AIC': params['AIC'],
                'complexity': len(params.get('trend_params', {})) + len(params.get('osc_params', {}))
            })
        except Exception as e:
            print(f"{fit_func.__name__} failed: {e}")

    if not results:
        raise RuntimeError("All model fits failed.")

    # Sort by AIC
    results.sort(key=lambda r: r['AIC'])

    best = results[0]
    for r in results[1:]:
        delta_aic = r['AIC'] - best['AIC']
        if r['complexity'] < best['complexity'] and delta_aic < delta_threshold:
            best = r  # prefer simpler model if AIC is close

    return best['y_pred'], best['var'], best['params']

from SingleCellDataAnalysis.config import WORKING_DIR
import os
def phi_to_frame_offset(phi, f, n):
    """
    Convert phase phi (radians) to delay in time frames.
    Always returns a positive delay within the harmonic period.
    """
    T_n = 1 / (n * f)  # period in frames
    delay = (-phi / (2 * np.pi * n * f)) % T_n
    return delay

def plot_simple_model_grid(df_all, cell_ids, time_points, feature1='pol1_int_corr', feature2='pol2_int_corr',
                            model_type='linear', start_idx=0, n_cells=25):
    model_fn_map = {
        'constant': fit_constant,
        'linear': fit_linear,
        'step': fit_step_discrete
    }
    
    assert model_type in model_fn_map, "Choose from 'constant', 'linear', or 'step'."
    
    fit_fn = model_fn_map[model_type]
    fig, axs = plt.subplots(5, 5, figsize=(20, 20), sharey=True)
    axs = axs.flatten()
    
    end_idx = start_idx + n_cells

    for i, cell_id in enumerate(cell_ids[start_idx:end_idx]):
        ax = axs[i]
        df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")

        t = df_cell["time_point"].values.reshape(-1, 1)
        y1 = df_cell[feature1].values.reshape(-1, 1)
        y2 = df_cell[feature2].values.reshape(-1, 1)

        valid = ~np.isnan(y1[:, 0]) & ~np.isnan(y2[:, 0])
        t, y1, y2 = t[valid], y1[valid], y2[valid]

        if len(t) < 5:
            ax.set_visible(False)
            continue

        try:
            mu1, var1, params1 = model_selector_with_threshold(t, y1, delta_threshold=10)
            mu2, var2, params2 = model_selector_with_threshold(t, y2, delta_threshold=10)
        except Exception as e:
            print(f"❌ Model failed on Cell {cell_id}: {e}")
            ax.set_visible(False)
            continue

        # Plot raw data and model fits
        
        ax.plot(t, y1, 'o', markersize=1, color='blue')
        ax.plot(t, mu1, 'b-')
        ax.plot(t, y2, 'o', markersize=1, color='darkorange')
        ax.plot(t, mu2, 'orange')
        
        for params, color in zip([params1, params2], ['blue', 'darkorange']):
            osc = params.get('osc_params', {})
            f = osc.get('f')
            trend = params.get('trend_params', {})

            if not f or 'model' not in trend:
               continue
            
            # Compute trend baseline
            if trend['model'] == 'linear':
                a = trend.get('a', 0)
                b = trend.get('b', 0)
                baseline = a * t[:, 0] + b
                #print(baseline)
            elif trend['model'] == 'constant':
                baseline = np.full_like(t[:, 0], trend.get('a', 0))
            else:
                baseline = np.zeros_like(t[:, 0])

            # --- Single dashed line: sum of first 3 harmonics + baseline ---
            harmonic_sum = np.zeros_like(t[:, 0], dtype=float)
            
            for j in range(1, 4):
                key_phi = f'phi{j}'
                key_amp = f'A{j}'
                key_n = f'n{j}'
            
                if key_phi in osc and key_amp in osc and key_n in osc:
                    A = osc[key_amp]
                    phi = osc[key_phi]
                    n = osc[key_n]
                    try:
                        harmonic_sum += A * np.sin(2 * np.pi * n * f * t[:, 0] + phi)
                    except Exception as e:
                        print(f"⚠️ Error computing harmonic {j}: {e}")
            
            # Add to baseline and plot one line
            y_wave_total = baseline + harmonic_sum
            ax.plot(t[:, 0], y_wave_total, linestyle='dashed', linewidth=1, color=color, alpha=0.7)


            # --- Text labels for phase offsets ---
            marker_positions = []
            
            for j in range(1, 21):
                key_phi = f'phi{j}'
                key_amp = f'A{j}'
                key_n = f'n{j}'
            
                if key_phi in osc and key_amp in osc and key_n in osc:
                    phi = osc[key_phi]
                    A = osc[key_amp]
                    n = osc[key_n]
                    try:
                        x_offset = phi_to_frame_offset(phi, f, n)
            
                        if trend['model'] == 'linear':
                            y_base = a * x_offset + b
                        elif trend['model'] == 'constant':
                            y_base = trend.get('a', 0)
                        else:
                            y_base = 0
                        y_marker = y_base + A
            
                        # Collision avoidance
                        spacing_threshold = 5
                        vertical_shift = -2
                        shift = sum(abs(existing_x - x_offset) < spacing_threshold for existing_x, _ in marker_positions)
                        y_shifted = y_marker + shift * vertical_shift * ((-1) ** shift)
            
                        marker_positions.append((x_offset, y_shifted))
            
                        # Scale fontsize with amplitude A (clip to avoid extremes)
                        font_min, font_max = 6, 20
                        A_clipped = max(min(A, 3), 0)  # restrict A to [0, 30]
                        font_size = font_min + (font_max - font_min) * (A_clipped / 3)
            
                        ax.text(x_offset, y_shifted, f'{j}', color=color, fontsize=font_size,
                                verticalalignment='bottom', horizontalalignment='center')
                    except Exception as e:
                        print(f"⚠️ Error placing φ{j}: {e}")


        ax.set_ylim(-10, 30)
        ax.grid(True)
        if i % 5 == 0:
            ax.set_ylabel("Signal")
        if i >= 20:
            ax.set_xlabel("Time")
        ax.set_title(
            f"Cell {cell_id}\npol1:\n{fmt_limited_nested(params1)}\npol2:\n{fmt_limited_nested(params2)}",
            fontsize=7
        )

    for j in range(n_cells, 25):
        axs[j].set_visible(False)

    fig.suptitle(f"{model_type.capitalize()} Fit for pol1 and pol2", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    



def quantify_all_cells(df_all, cell_ids, feature1='pol1_int_corr', feature2='pol2_int_corr', delta_threshold=4, filename='model_fits_by_cell.csv'):
    records = []

    for cell_id in cell_ids:
        df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
        t = df_cell["time_point"].values.reshape(-1, 1)
        y1 = df_cell[feature1].values.reshape(-1, 1)
        y2 = df_cell[feature2].values.reshape(-1, 1)

        valid = ~np.isnan(y1[:, 0]) & ~np.isnan(y2[:, 0])
        t_valid, y1_valid, y2_valid = t[valid], y1[valid], y2[valid]

        if len(t_valid) < 5:
            continue

        for feature, y in zip(['pol1', 'pol2'], [y1_valid, y2_valid]):
            try:
                _, _, params = model_selector_with_threshold(t_valid, y,delta_threshold=delta_threshold)
                flat_params = {
                    'cell_id': cell_id,
                    'feature': feature,
                    'model': params.get('model'),
                    'AIC': params.get('AIC')
                }

                for prefix in ['trend_params', 'osc_params']:
                    if prefix in params:
                        for k, v in params[prefix].items():
                            flat_params[f"{prefix}.{k}"] = v

                # ➕ Add phi-to-frame offset conversions
                # ➕ Compute phi-to-frame offset after flattening
                if 'osc_params.f' in flat_params:
                    f = flat_params['osc_params.f']
                    for i in range(1, 21):  # scan for up to 20 harmonics
                        phi_key = f'osc_params.phi{i}'
                        n_key = f'osc_params.n{i}'
                        if phi_key in flat_params and n_key in flat_params:
                            phi = flat_params[phi_key]
                            n = flat_params[n_key]
                            try:
                                offset = phi_to_frame_offset(phi, f, n)
                                flat_params[f'{phi_key}_offset'] = offset
                            except Exception as e:
                                print(f"⚠️ Failed to convert phi{i} for cell {cell_id}, feature {feature}: {e}")


                records.append(flat_params)

            except Exception as e:
                print(f"Model failed for Cell {cell_id}, {feature}: {e}")
                continue

    df_result = pd.DataFrame.from_records(records)

    output_path = os.path.join(WORKING_DIR, filename)
    df_result.to_csv(output_path, index=False)
    print(f"✔️ Results saved to: {output_path}")

    return df_result

from scipy.signal import correlate

def quantify_all_cells_xcor(df_all, cell_ids, feature1='pol1_int_corr', feature2='pol2_int_corr',
                             delta_threshold=4, filename='xcor_detrended_results.csv'):
    records = []

    for cell_id in cell_ids:
        df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
        t = df_cell["time_point"].values.reshape(-1, 1)
        y1 = df_cell[feature1].values.reshape(-1, 1)
        y2 = df_cell[feature2].values.reshape(-1, 1)

        valid = ~np.isnan(y1[:, 0]) & ~np.isnan(y2[:, 0])
        t_valid, y1_valid, y2_valid = t[valid], y1[valid], y2[valid]

        if len(t_valid) < 5:
            continue

        try:
            # Detrend both signals
            y1_trend, _, _ = model_selector_with_threshold(t_valid, y1_valid, N_max=3, delta_threshold=delta_threshold)
            y2_trend, _, _ = model_selector_with_threshold(t_valid, y2_valid, N_max=3, delta_threshold=delta_threshold)
            y1_detrended = (y1_valid - y1_trend)[:, 0]
            y2_detrended = (y2_valid - y2_trend)[:, 0]

            # Cross-correlation
            xcor = correlate(y1_detrended, y2_detrended, mode='full')
            lags = np.arange(-len(y1_detrended) + 1, len(y1_detrended))

            max_idx = np.argmax(xcor)
            xcor_max = xcor[max_idx]
            xcor_lag = lags[max_idx]

            # Zero-lag correlation is at center index
            center_idx = len(xcor) // 2
            xcor_zero_lag = xcor[center_idx]

            records.append({
                'cell_id': cell_id,
                'xcor_max': xcor_max,
                'xcor_lag': xcor_lag,
                'xcor_zero_lag': xcor_zero_lag
            })

        except Exception as e:
            print(f"❌ Failed for Cell {cell_id}: {e}")
            continue

    df_result = pd.DataFrame.from_records(records)

    output_path = os.path.join(WORKING_DIR, filename)
    df_result.to_csv(output_path, index=False)
    print(f"✔️ Cross-correlation results saved to: {output_path}")

    return df_result


def summarize_model_distribution(df_results):
    # Ensure we have exactly one row per (cell_id, feature)
    df_pivot = df_results.pivot(index="cell_id", columns="feature", values="model")

    # Count combinations
    counts = df_pivot.value_counts().reset_index()
    counts.columns = ['pol1_model', 'pol2_model', 'count']

    # Create 4x4 matrix
    model_order = ['constant', 'linear', 'linear+sine', 'linear+harmonic']
    table = pd.DataFrame(0, index=model_order, columns=model_order)

    for _, row in counts.iterrows():
        table.loc[row['pol1_model'], row['pol2_model']] = row['count']

    return table


def symmetric_model_key(model1, model2):
    return " & ".join(sorted([model1, model2]))

def build_model_category_heatmap_df(df_all, df_results, best_shifts, time_bin_width=1):
    # Pivot results to get pol1/pol2 model type per cell
    model_by_cell = df_results.pivot(index="cell_id", columns="feature", values="model").reset_index()

    # Generate symmetric model category
    model_by_cell["model_category"] = model_by_cell.apply(
        lambda row: symmetric_model_key(row["pol1"], row["pol2"]), axis=1
    )

    # Merge with time data
    df_time = df_all[["cell_id", "time_point"]].copy()
    df_time = df_time.drop_duplicates()

    # Add best shift as global time offset
    df_time["best_shift"] = df_time["cell_id"].map(best_shifts)
    df_time["global_time"] = df_time["time_point"] + df_time["best_shift"]

    # Merge with model category
    df_merged = pd.merge(df_time, model_by_cell[["cell_id", "model_category"]], on="cell_id")

    # Bin global time
    if time_bin_width > 1:
        df_merged["time_bin"] = (df_merged["global_time"] // time_bin_width) * time_bin_width
    else:
        df_merged["time_bin"] = df_merged["global_time"]

    # Count cells by model category and global time bin
    heatmap_df = df_merged.groupby(["model_category", "time_bin"]).size().reset_index(name="count")
    return heatmap_df


def pivot_heatmap_matrix(heatmap_df):
    heatmap_matrix = heatmap_df.pivot(index="model_category", columns="time_bin", values="count").fillna(0)
    return heatmap_matrix

import seaborn as sns


def plot_model_heatmap(heatmap_matrix):
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_matrix, cmap="viridis", linewidths=0.5, linecolor='grey', annot=False)
    plt.title("Cell Model Category Distribution Over Time")
    plt.xlabel("Global Time")
    plt.ylabel("Model Combination (symmetrical)")
    plt.tight_layout()
    plt.show()

def extract_oscillation_data(df_results):
    rows = []

    for _, row in df_results.iterrows():
        model = row['model']
        feature = row['feature']
        cell_id = row['cell_id']

        if model == "linear+sine":
            amp = abs(row.get('osc_params.A', np.nan))
            freq = row.get('osc_params.f', np.nan)
        elif model == "linear+harmonic":
            A1 = abs(row.get('osc_params.A1', 0))
            A2 = abs(row.get('osc_params.A2', 0))
            f = row.get('osc_params.f', np.nan)
            n = row.get('osc_params.n', 1)
            amp = A1 + A2
            freq = f * n if f is not None and n is not None else np.nan
        else:
            continue  # Skip constant and linear-only models

        if not np.isnan(freq) and not np.isnan(amp):
            rows.append({
                'cell_id': cell_id,
                'feature': feature,
                'model': model,
                'amplitude': amp,
                'frequency': freq
            })

    return pd.DataFrame(rows)



import matplotlib.gridspec as gridspec

def plot_oscillation_scatter(df_osc):
    # Initialize figure and layout
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(4, 4, figure=fig)

    # Axes: top violin (frequency), right violin (amplitude), main scatter
    ax_violin_x = fig.add_subplot(gs[0, 0:3])
    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_violin_y = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)

    # Violin plot on top (frequency distribution)
    sns.violinplot(
        data=df_osc,
        x="frequency",
        y=None,
        hue="feature",
        ax=ax_violin_x,
        dodge=True,
        linewidth=1,
        inner="box"
    )
    ax_violin_x.set_ylabel("")
    ax_violin_x.set_xlabel("")
    ax_violin_x.tick_params(bottom=False, labelbottom=False)
    ax_violin_x.legend_.remove()

    # Scatter plot in the center
    sns.scatterplot(
        data=df_osc,
        x="frequency",
        y="amplitude",
        hue="feature",
        style="model",
        alpha=0.7,
        s=60,
        ax=ax_scatter
    )
    ax_scatter.set_xlabel("Frequency (Hz or 1/time unit)")
    ax_scatter.set_ylabel("Amplitude")
    ax_scatter.set_title("Amplitude vs Frequency of Oscillatory Fits")
    ax_scatter.grid(True)

    # Violin plot on right (amplitude distribution)
    sns.violinplot(
        data=df_osc,
        y="amplitude",
        x=None,
        hue="feature",
        ax=ax_violin_y,
        dodge=True,
        linewidth=1,
        inner="box",
        orient="v"
    )
    ax_violin_y.set_xlabel("")
    ax_violin_y.set_ylabel("")
    ax_violin_y.tick_params(left=False, labelleft=False)
    ax_violin_y.legend_.remove()

    plt.tight_layout()
    plt.show()



def get_median_global_times(df_all, best_shifts):
    df = df_all[["cell_id", "time_point"]].drop_duplicates().copy()
    df["best_shift"] = df["cell_id"].map(best_shifts)
    df["global_time"] = df["time_point"] + df["best_shift"]
    median_global_time = df.groupby("cell_id")["global_time"].median()
    return median_global_time.to_dict()

def extract_frequency_with_time(df_results, median_global_time_dict, t_mid=25):
    rows = []

    for _, row in df_results.iterrows():
        model = row['model']
        feature = row['feature']
        cell_id = row['cell_id']

        # Trend values
        a = row.get("trend_params.a", 0)
        b = row.get("trend_params.b", 0)
        lin_val = abs(a * t_mid + b)

        # Oscillation component
        if model == "linear+sine":
            A = abs(row.get('osc_params.A', np.nan))
            freq = row.get('osc_params.f', np.nan)
        elif model == "linear+harmonic":
            A1 = abs(row.get('osc_params.A1', 0))
            A2 = abs(row.get('osc_params.A2', 0))
            A = A1 + A2
            f = row.get('osc_params.f', np.nan)
            n = row.get('osc_params.n', 1)
            freq = f * n if f is not None and n is not None else np.nan
        else:
            continue

        amp_total = A + lin_val
        global_time = median_global_time_dict.get(cell_id, np.nan)

        if not np.isnan(freq) and not np.isnan(global_time) and amp_total > 0:
            rows.append({
                'cell_id': cell_id,
                'feature': feature,
                'model': model,
                'frequency': freq,
                'global_time': global_time,
                'amplitude': amp_total,
                'osc_amplitude': A,
                'linear_value_at_mid': lin_val
            })

    return pd.DataFrame(rows)


def plot_frequency_timeline(df_freq_time):
    plt.figure(figsize=(10, 6))

    # Filter time window for regression
    df_reg = df_freq_time[(df_freq_time["global_time"] >= 0) & (df_freq_time["global_time"] <= 400)]

    # Scatter plot (grouped colors & styles)
    sns.scatterplot(
        data=df_freq_time,
        x="global_time",
        y="frequency",
        hue="feature",
        style="model",
        s=60,
        alpha=0.7
    )

    # Global regression line (not separated)
    sns.regplot(
        data=df_reg,
        x="global_time",
        y="frequency",
        scatter=False,
        color="black",
        #linewidth=2,
        label="Linear fit (0–400)",
        ci=None
    )

    plt.xlabel("Global Time")
    plt.ylabel("Frequency")
    plt.title("Oscillation Frequency vs Global Time (Amplitude > 5)\nWith Unified Linear Regression (0–400)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_amplitude_timeline(df_freq_time):
    plt.figure(figsize=(10, 6))

    df_reg = df_freq_time[(df_freq_time["global_time"] >= 0) & (df_freq_time["global_time"] <= 400)]

    sns.scatterplot(
        data=df_freq_time,
        x="global_time",
        y="amplitude",
        hue="feature",
        style="model",
        s=60,
        alpha=0.7
    )

    sns.regplot(
        data=df_reg,
        x="global_time",
        y="amplitude",
        scatter=False,
        color="black",
        #linewidth=2,
        label="Linear fit (0–400)",
        ci=None
    )

    plt.xlabel("Global Time")
    plt.ylabel("Amplitude")
    plt.title("Oscillation Amplitude vs Global Time\nWith Unified Linear Regression (0–400)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()



def extract_slope_with_time(df_results, df_all, best_shifts):
    # Build median global time dict
    df = df_all[["cell_id", "time_point"]].drop_duplicates().copy()
    df["best_shift"] = df["cell_id"].map(best_shifts)
    df["global_time"] = df["time_point"] + df["best_shift"]
    median_global_time = df.groupby("cell_id")["global_time"].median().to_dict()

    # Extract slope per feature
    rows = []
    for _, row in df_results.iterrows():
        cell_id = row['cell_id']
        feature = row['feature']
        model = row['model']
        global_time = median_global_time.get(cell_id, np.nan)

        # Slope determination
        if model == "constant":
            slope = 0.0
        elif "trend_params.a" in row:
            slope = row["trend_params.a"]
        else:
            slope = np.nan  # e.g., model fit failed

        if not np.isnan(slope) and not np.isnan(global_time):
            rows.append({
                "cell_id": cell_id,
                "feature": feature,
                "model": model,
                "slope": slope,
                "global_time": global_time
            })

    return pd.DataFrame(rows)


def plot_slope_timeline(df_slope_time):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_slope_time,
        x="global_time",
        y="slope",
        hue="feature",
        style="model",
        s=60,
        alpha=0.8
    )
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Global Time")
    plt.ylabel("Slope of Linear Trend")
    plt.title("Slope vs Global Timeline (0 = constant)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_first_time_distribution(df_slope_time, pol1="pol1", pol2="pol2"):
    # Pivot the slope dataframe for easier comparison
    df_pivot = df_slope_time.pivot(index="cell_id", columns="feature", values="slope")

    # Drop cells with missing slope info for either pol1 or pol2
    df_pivot = df_pivot[[pol1, pol2]].dropna()

    # Classify cells into 3 categories
    def classify_row(row):
        pos1 = row[pol1] > 0
        pos2 = row[pol2] > 0
        if pos1 and pos2:
            return "Both positive"
        elif pos1 or pos2:
            return "One positive"
        else:
            return "None positive"

    df_pivot["category"] = df_pivot.apply(classify_row, axis=1)
    df_pivot = df_pivot.reset_index()

    # Merge with global_time information (take first aligned time point)
    df_first_time = df_slope_time.groupby("cell_id")["global_time"].min().reset_index(name="first_global_time")
    df_plot = pd.merge(df_pivot, df_first_time, on="cell_id", how="left")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_plot,
        x="first_global_time",
        hue="category",
        multiple="dodge",
        bins=10,
        palette="muted"
    )
    plt.xlabel("First Aligned Global Time")
    plt.ylabel("Number of Cells")
    plt.title("Distribution of First Global Time by Slope Categories")
    plt.tight_layout()
    plt.show()


def filter_slope_data(df_slope_time, time_min=0, time_max=400):
    return df_slope_time[
        (df_slope_time["global_time"] >= time_min) &
        (df_slope_time["global_time"] <= time_max)
    ]



def fit_slope_vs_time(df_filtered):
    X = df_filtered["global_time"].values.reshape(-1, 1)
    y = df_filtered["slope"].values
    reg = LinearRegression().fit(X, y)
    return reg  # contains reg.coef_, reg.intercept_

def plot_slope_with_regression(df_filtered, reg):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_filtered,
        x="global_time",
        y="slope",
        hue="feature",
        alpha=0.7,
        s=60
    )

    # Regression line
    x_vals = np.linspace(df_filtered["global_time"].min(), df_filtered["global_time"].max(), 100).reshape(-1, 1)
    y_vals = reg.predict(x_vals)
    plt.plot(x_vals, y_vals, color='black', linestyle='--', label=f"Regression: slope={reg.coef_[0]:.3f}")

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Global Time")
    plt.ylabel("Slope of Linear Trend")
    plt.title("Linear Trend Slope vs Global Time (0–400)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def prepare_pol1_pol2_slope_with_time(df_slope_time, df_all, best_shifts):
    # Median global time per cell
    df = df_all[["cell_id", "time_point"]].drop_duplicates().copy()
    df["best_shift"] = df["cell_id"].map(best_shifts)
    df["global_time"] = df["time_point"] + df["best_shift"]
    median_global_time = df.groupby("cell_id")["global_time"].median().to_dict()

    # Pivot slope data
    df_pivot = df_slope_time.pivot(index="cell_id", columns="feature", values="slope")
    df_pivot.columns = ['pol1_slope', 'pol2_slope']
    df_pivot = df_pivot.dropna()

    # Add global time
    df_pivot["global_time"] = df_pivot.index.map(median_global_time)
    return df_pivot.reset_index()

def plot_pol1_vs_pol2_with_lines(df_pivoted):
    df_sorted = df_pivoted.sort_values("global_time")

    plt.figure(figsize=(7, 6))

    # Draw line through points in temporal order
    plt.plot(
        df_sorted["pol1_slope"],
        df_sorted["pol2_slope"],
        color="lightgray",
        linewidth=1,
        linestyle='-',
        zorder=1
    )

    # Scatter plot with color based on global time
    scatter = plt.scatter(
        df_sorted["pol1_slope"],
        df_sorted["pol2_slope"],
        c=df_sorted["global_time"],
        cmap="viridis",
        edgecolor='k',
        s=15,
        alpha=0.9,
        zorder=2
    )

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("pol1 Slope")
    plt.ylabel("pol2 Slope")
    plt.title("pol1 vs pol2 Slope Trajectory Across Global Time")
    plt.grid(True)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Global Time")
    plt.tight_layout()
    plt.show()



def compute_correlation_corrected(df_all, pol1_col="pol1_int_corr", pol2_col="pol2_int_corr"):
    df = df_all.copy()

    # Drop rows with missing values
    df = df.dropna(subset=[pol1_col, pol2_col])

    # Compute correlation for each cell
    results = []
    for cell_id, group in df.groupby("cell_id"):
        if group.shape[0] >= 2:
            corr = np.corrcoef(group[pol1_col], group[pol2_col])[0, 1]
            median_time = group["aligned_time"].median()
        else:
            corr = np.nan
            median_time = np.nan

        results.append({
            "cell_id": cell_id,
            "pearson_r": corr,
            "n_points": group.shape[0],
            "median_aligned_time": median_time
        })

    return pd.DataFrame(results)


def plot_correlation_timeline(df_corr):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_corr, x="median_aligned_time", y="pearson_r")
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.ylim(-1.05, 1.05)
    plt.xlabel("Median Aligned Time")
    plt.ylabel("Pearson Correlation (pol1 vs pol2)")
    plt.title("Per-Cell Correlation vs Aligned Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def simulate_stepwise_cells(n_cells=25, n_timepoints=51, noise_std=2.0, seed=42):
    np.random.seed(seed)
    all_records = []

    for cell_id in range(n_cells):
        step1 = np.random.randint(10, 30)  # step point for pol1
        step2 = np.random.randint(20, 40)  # step point for pol2

        low1, high1 = np.random.uniform(5, 20), np.random.uniform(25, 50)
        low2, high2 = np.random.uniform(10, 30), np.random.uniform(30, 55)

        if np.random.rand() < 0.5:
            low1, high1 = high1, low1  # allow step down
        if np.random.rand() < 0.5:
            low2, high2 = high2, low2  # allow step down

        time_points = np.arange(n_timepoints)
        pol1_values = np.where(time_points < step1, low1, high1) + np.random.normal(0, noise_std, n_timepoints)
        pol2_values = np.where(time_points < step2, low2, high2) + np.random.normal(0, noise_std, n_timepoints)

        for t, p1, p2 in zip(time_points, pol1_values, pol2_values):
            all_records.append({
                'cell_id': cell_id,
                'time_point': t,
                'pol1_int_corr': p1,
                'pol2_int_corr': p2
            })

    df = pd.DataFrame(all_records)
    return df




