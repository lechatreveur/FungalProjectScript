#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 16:23:31 2025

@author: user
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.signal import correlate
import pandas as pd
from SingleCellDataAnalysis.config import WORKING_DIR
import os as os
from functools import partial
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


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

def sine_func(t, A, f, phi):
    return A * np.sin(2 * np.pi * f * t + phi)

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

# full model selection
def compute_aic(y_true, y_pred, n_params):
    resid = y_true - y_pred
    mse = np.mean(resid**2)
    aic = 2 * n_params + len(y_true) * np.log(mse)
    return aic

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

def phi_to_frame_offset(phi, f, n):
    """
    Convert phase phi (radians) to delay in time frames.
    Always returns a positive delay within the harmonic period.
    """
    T_n = 1 / (n * f)  # period in frames
    delay = (-phi / (2 * np.pi * n * f)) % T_n
    return delay


def plot_simple_model_grid(df_all, cell_ids, time_points, feature1='pol1_int_corr', feature2='pol2_int_corr',
                            model_type='linear', start_idx=0, n_cells=25,
                            show_fit=True, pol_high_color='blue', pol_low_color='darkorange'):
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

        # Determine which signal is stronger on average
        avg_y1, avg_y2 = np.nanmean(y1), np.nanmean(y2)
        if avg_y1 >= avg_y2:
            color1, color2 = pol_high_color, pol_low_color
            stronger_label = feature1
        else:
            color1, color2 = pol_low_color, pol_high_color
            stronger_label = feature2

        # Plot raw data
        ax.plot(t, y1, 'o-', markersize=1, color=color1)
        ax.plot(t, y2, 'o-', markersize=1, color=color2)

        if show_fit:
            try:
                mu1, var1, params1 = model_selector_with_threshold(t, y1, delta_threshold=10)
                mu2, var2, params2 = model_selector_with_threshold(t, y2, delta_threshold=10)
            except Exception as e:
                print(f"❌ Model failed on Cell {cell_id}: {e}")
                ax.set_visible(False)
                continue

            ax.plot(t, mu1, '-', color=color1)
            ax.plot(t, mu2, '-', color=color2)

            for params, color in zip([params1, params2], [color1, color2]):
                osc = params.get('osc_params', {})
                f = osc.get('f')
                trend = params.get('trend_params', {})

                if not f or 'model' not in trend:
                    continue

                # Compute baseline
                if trend['model'] == 'linear':
                    a = trend.get('a', 0)
                    b = trend.get('b', 0)
                    baseline = a * t[:, 0] + b
                elif trend['model'] == 'constant':
                    baseline = np.full_like(t[:, 0], trend.get('a', 0))
                else:
                    baseline = np.zeros_like(t[:, 0])

                # Harmonic sum
                harmonic_sum = np.zeros_like(t[:, 0], dtype=float)
                for j in range(1, 4):
                    key_phi = f'phi{j}'
                    key_amp = f'A{j}'
                    key_n = f'n{j}'
                    if key_phi in osc and key_amp in osc and key_n in osc:
                        try:
                            A = osc[key_amp]
                            phi = osc[key_phi]
                            n = osc[key_n]
                            harmonic_sum += A * np.sin(2 * np.pi * n * f * t[:, 0] + phi)
                        except Exception as e:
                            print(f"⚠️ Error computing harmonic {j}: {e}")

                y_wave_total = baseline + harmonic_sum
                ax.plot(t[:, 0], y_wave_total, linestyle='dashed', linewidth=1, color=color, alpha=0.7)

                # Optional: place φj labels
                marker_positions = []
                for j in range(1, 21):
                    key_phi = f'phi{j}'
                    key_amp = f'A{j}'
                    key_n = f'n{j}'
                    if key_phi in osc and key_amp in osc and key_n in osc:
                        try:
                            phi = osc[key_phi]
                            A = osc[key_amp]
                            n = osc[key_n]
                            x_offset = phi_to_frame_offset(phi, f, n)

                            y_base = a * x_offset + b if trend['model'] == 'linear' else trend.get('a', 0)
                            y_marker = y_base + A

                            spacing_threshold = 5
                            vertical_shift = -2
                            shift = sum(abs(existing_x - x_offset) < spacing_threshold for existing_x, _ in marker_positions)
                            y_shifted = y_marker + shift * vertical_shift * ((-1) ** shift)
                            marker_positions.append((x_offset, y_shifted))

                            font_min, font_max = 6, 20
                            A_clipped = max(min(A, 3), 0)
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
        ax.set_title(f"Cell {cell_id}", fontsize=7)

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

    output_path = filename
    df_result.to_csv(output_path, index=False)
    print(f"✔️ Results saved to: {output_path}")

    return df_result




# def quantify_all_cells_xcor(df_all, cell_ids, feature1='pol1_int_corr', feature2='pol2_int_corr',
#                              delta_threshold=4, filename='xcor_detrended_results.csv',
#                              visualize=False):
#     records = []

#     for cell_id in cell_ids:
#         df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
#         t = df_cell["time_point"].values.reshape(-1, 1)
#         y1 = df_cell[feature1].values.reshape(-1, 1)
#         y2 = df_cell[feature2].values.reshape(-1, 1)

#         valid = ~np.isnan(y1[:, 0]) & ~np.isnan(y2[:, 0])
#         t_valid, y1_valid, y2_valid = t[valid], y1[valid], y2[valid]

#         if len(t_valid) < 5:
#             continue

#         try:
#             # Detrend both signals
#             y1_trend, _, _ = model_selector_with_threshold(t_valid, y1_valid, N_max=3, delta_threshold=delta_threshold)
#             y2_trend, _, _ = model_selector_with_threshold(t_valid, y2_valid, N_max=3, delta_threshold=delta_threshold)
#             y1_detrended = (y1_valid - y1_trend)[:, 0]
#             y2_detrended = (y2_valid - y2_trend)[:, 0]

#             # Cross-correlation
#             xcor = correlate(y1_detrended, y2_detrended, mode='full')
#             lags = np.arange(-len(y1_detrended) + 1, len(y1_detrended))
#             center_idx = len(xcor) // 2
#             xcor_zero_lag = xcor[center_idx]

#             # Improved peak detection (positive lags only)
#             xcor_temp = xcor.copy()
#             xcor_temp[center_idx] = -np.inf  # mask zero-lag

#             pos_lags = lags[center_idx + 1:]
#             xcor_pos = xcor[center_idx + 1:]

#             diffs = np.diff(xcor_pos)
#             std_noise = np.std(diffs)

#             peaks, props = find_peaks(xcor_pos, height=1.96 * std_noise)

#             first_dip_idx = np.argmax(diffs < 0) + 1
#             valid_peaks = [p for p in peaks if p > first_dip_idx]

#             if valid_peaks:
#                 xcor_max_idx = valid_peaks[0] + center_idx + 1
#             else:
#                 xcor_max_idx = np.argmax(xcor_temp)

#             xcor_max = xcor[xcor_max_idx]
#             xcor_lag = lags[xcor_max_idx]

#             records.append({
#                 'cell_id': cell_id,
#                 'xcor_max': xcor_max,
#                 'xcor_lag': xcor_lag,
#                 'xcor_zero_lag': xcor_zero_lag
#             })

#             # --- Optional Visualization ---
#             if visualize:
#                 fig, axs = plt.subplots(2, 2, figsize=(12, 8))

#                 axs[0, 0].plot(t_valid[:, 0], y1_valid[:, 0], label='Original')
#                 axs[0, 0].plot(t_valid[:, 0], y1_trend[:, 0], '--', label='Trend')
#                 axs[0, 0].plot(t_valid[:, 0], y1_detrended, label='Detrended')
#                 axs[0, 0].set_title(f'Cell {cell_id} - {feature1}')
#                 axs[0, 0].legend()

#                 axs[0, 1].plot(t_valid[:, 0], y2_valid[:, 0], label='Original')
#                 axs[0, 1].plot(t_valid[:, 0], y2_trend[:, 0], '--', label='Trend')
#                 axs[0, 1].plot(t_valid[:, 0], y2_detrended, label='Detrended')
#                 axs[0, 1].set_title(f'Cell {cell_id} - {feature2}')
#                 axs[0, 1].legend()

#                 axs[1, 0].plot(lags, xcor)
#                 axs[1, 0].axvline(x=0, color='k', linestyle='--', label='Zero Lag')
#                 axs[1, 0].axvline(x=xcor_lag, color='r', linestyle='--', label='Peak Lag')
#                 axs[1, 0].set_title('Cross-Correlation')
#                 axs[1, 0].legend()

#                 axs[1, 1].axis('off')

#                 plt.tight_layout()
#                 plt.show()

#         except Exception as e:
#             print(f"❌ Failed for Cell {cell_id}: {e}")
#             continue

#     df_result = pd.DataFrame.from_records(records)

#     output_path = os.path.join(WORKING_DIR, filename)
#     df_result.to_csv(output_path, index=False)
#     print(f"✔️ Cross-correlation results saved to: {output_path}")

#     return df_result



# def fit_exponential_to_acor(lags, acor_vals):
#     """
#     Fit an exponential curve to positive autocorrelation lags.
#     Assumes acor_vals and lags are 1D NumPy arrays.
    
#     Returns:
#         decay_const (float): Time constant tau of the exponential decay.
#         amplitude (float): Amplitude of the exponential curve.
#         fit_success (bool): Whether the fit was successful.
#     """
#     # Define exponential model
#     def exp_func(x, A, tau):
#         return A * np.exp(-x / tau)

#     try:
#         # Remove zero and negative autocorrelation values (and lags)
#         mask = (acor_vals > 0) & (lags > 0)
#         x = lags[mask]
#         y = acor_vals[mask]

#         # Normalize for stability
#         y_norm = y / np.max(y)

#         # Initial guess: A = 1, tau = half max lag
#         popt, _ = curve_fit(exp_func, x, y_norm, p0=(1.0, np.median(x)))

#         A_fit, tau_fit = popt
#         return tau_fit, A_fit, True

#     except Exception as e:
#         print(f"⚠️ Exponential fit failed: {e}")
#         return np.nan, np.nan, False

# import arviz as az




# import pymc as pm
# pm.set_backend("numpyro")  # ✅ forces PyMC to avoid Aesara entirely

# import pymc.sampling_jax as pmjax

# def bayesian_fit_model_A(lags, acor_vals):
#     with pm.Model() as model_exp:
#         A = pm.HalfNormal("A", sigma=1.0)
#         tau = pm.HalfNormal("tau", sigma=10.0)
#         sigma = pm.HalfNormal("sigma", sigma=0.1)

#         mu = A * pm.math.exp(-lags / tau)
#         Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=acor_vals)

#         trace = pmjax.sample_numpyro_nuts(
#             draws=1000, tune=1000, target_accept=0.9
#         )

#     return model_exp, trace





# def bayesian_fit_model_B(lags, acor_vals):
#     with pm.Model() as model_osc:
#         A = pm.HalfNormal("A", sigma=1.0)
#         tau = pm.HalfNormal("tau", sigma=10.0)
#         f = pm.HalfNormal("f", sigma=1.0)
#         phi = pm.Uniform("phi", lower=0, upper=2*np.pi)
#         sigma = pm.HalfNormal("sigma", sigma=0.1)

#         mu = A * pm.math.exp(-lags / tau) * pm.math.cos(2 * np.pi * f * lags + phi)

#         Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=acor_vals)

#         trace = pmjax.sample_numpyro_nuts(
#             draws=1000, tune=1000, target_accept=0.9
#         )

#     return model_osc, trace


# def quantify_all_cells_acor(df_all, cell_ids,
#                             feature1='pol1_int_corr', feature2='pol2_int_corr',
#                             delta_threshold=4, filename='acor_detrended_results.csv',
#                             visualize=False):
#     records = []

#     for cell_id in cell_ids:
#         df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
#         t = df_cell["time_point"].values.reshape(-1, 1)
#         y1 = df_cell[feature1].values.reshape(-1, 1)
#         y2 = df_cell[feature2].values.reshape(-1, 1)

#         valid = ~np.isnan(y1[:, 0]) & ~np.isnan(y2[:, 0])
#         t_valid, y1_valid, y2_valid = t[valid], y1[valid], y2[valid]

#         if len(t_valid) < 5:
#             continue

#         try:
#             # --- pol1 detrend and autocor ---
#             y1_trend, _, _ = model_selector_with_threshold(t_valid, y1_valid, N_max=3, delta_threshold=delta_threshold)
#             y1_detrended = (y1_valid - y1_trend)[:, 0]
#             acor1 = correlate(y1_detrended, y1_detrended, mode='full')
#             lags1 = np.arange(-len(y1_detrended) + 1, len(y1_detrended))
#             center_idx1 = len(acor1) // 2
#             acor1_zero = acor1[center_idx1]
#             acor1_temp = acor1.copy()
#             acor1_temp[center_idx1] = -np.inf

#             # Use positive lag only
#             pos_lags1 = lags1[center_idx1 + 1:]
#             acor1_pos = acor1[center_idx1 + 1:]

#             # Find peaks
#             # Estimate noise std from first differences
#             diffs1 = np.diff(acor1_pos)
#             std_noise1 = np.std(diffs1)
            
#             # Find peaks above noise threshold
#             peaks1, props1 = find_peaks(acor1_pos, height=1.96 * std_noise1)
            
#             # Define first dip index
#             first_dip_idx = np.argmax(np.diff(acor1_pos) < 0) + 1
#             valid_peaks1 = [p for p in peaks1 if p > first_dip_idx]
            
#             if valid_peaks1:
#                 acor1_max_idx = valid_peaks1[0] + center_idx1 + 1
#             else:
#                 acor1_max_idx = np.argmax(acor1_temp)

#             acor1_max = acor1[acor1_max_idx]
#             acor1_lag = lags1[acor1_max_idx]

#             # --- pol2 detrend and autocor ---
#             y2_trend, _, _ = model_selector_with_threshold(t_valid, y2_valid, N_max=3, delta_threshold=delta_threshold)
#             y2_detrended = (y2_valid - y2_trend)[:, 0]
#             acor2 = correlate(y2_detrended, y2_detrended, mode='full')
#             lags2 = np.arange(-len(y2_detrended) + 1, len(y2_detrended))
#             center_idx2 = len(acor2) // 2
#             acor2_zero = acor2[center_idx2]
#             acor2_temp = acor2.copy()
#             acor2_temp[center_idx2] = -np.inf
            
#             pos_lags2 = lags2[center_idx2 + 1:]
#             acor2_pos = acor2[center_idx2 + 1:]
            
#             diffs2 = np.diff(acor2_pos)
#             std_noise2 = np.std(diffs2)
            
#             peaks2, props2 = find_peaks(acor2_pos, height=1.96 * std_noise2)
            
#             first_dip_idx = np.argmax(np.diff(acor2_pos) < 0) + 1
#             valid_peaks2 = [p for p in peaks2 if p > first_dip_idx]
            
#             if valid_peaks2:
#                 acor2_max_idx = valid_peaks2[0] + center_idx2 + 1
#             else:
#                 acor2_max_idx = np.argmax(acor2_temp)


#             acor2_max = acor2[acor2_max_idx]
#             acor2_lag = lags2[acor2_max_idx]



#             tau1, amp1, fit1_success = fit_exponential_to_acor(pos_lags1, acor1_pos)
#             tau2, amp2, fit2_success = fit_exponential_to_acor(pos_lags2, acor2_pos)
            
#             model_A, trace_A = bayesian_fit_model_A(pos_lags1, acor1_pos)
#             model_B, trace_B = bayesian_fit_model_B(pos_lags1, acor1_pos)
            
#             cmp_df = az.compare({"Exponential": trace_A, "Oscillatory": trace_B}, method="WAIC")
#             print(cmp_df)
            
#             az.plot_posterior(trace_B, var_names=["A", "tau", "f", "phi"])



#             records.append({
#                 'cell_id': cell_id,
#                 'acor1_max': acor1_max,
#                 'acor1_lag': acor1_lag,
#                 'acor1_zero_lag': acor1_zero,
#                 'acor2_max': acor2_max,
#                 'acor2_lag': acor2_lag,
#                 'acor2_zero_lag': acor2_zero,
#                 'acor1_tau': tau1,
#                 'acor1_amp': amp1,
#                 'acor2_tau': tau2,
#                 'acor2_amp': amp2
#             })

#             # --- Optional visualization ---
#             if visualize:
#                 fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#                 axs[0, 0].plot(t_valid[:, 0], y1_valid[:, 0], label='Original')
#                 axs[0, 0].plot(t_valid[:, 0], y1_trend[:, 0], '--', label='Trend')
#                 axs[0, 0].plot(t_valid[:, 0], y1_detrended, label='Detrended')
#                 axs[0, 0].set_title(f'Cell {cell_id} - {feature1}')
#                 axs[0, 0].legend()

#                 axs[0, 1].plot(lags1, acor1)
#                 axs[0, 1].axvline(x=0, color='k', linestyle='--', label='Zero Lag')
#                 #axs[0, 1].axvline(x=acor1_lag, color='r', linestyle='--', label='Peak Lag')
#                 axs[0, 1].set_title('Autocorrelation (pol1)')
#                 axs[0, 1].legend()

#                 axs[1, 0].plot(t_valid[:, 0], y2_valid[:, 0], label='Original')
#                 axs[1, 0].plot(t_valid[:, 0], y2_trend[:, 0], '--', label='Trend')
#                 axs[1, 0].plot(t_valid[:, 0], y2_detrended, label='Detrended')
#                 axs[1, 0].set_title(f'Cell {cell_id} - {feature2}')
#                 axs[1, 0].legend()

#                 axs[1, 1].plot(lags2, acor2)
#                 axs[1, 1].axvline(x=0, color='k', linestyle='--', label='Zero Lag')
#                 #axs[1, 1].axvline(x=acor2_lag, color='r', linestyle='--', label='Peak Lag')
#                 axs[1, 1].set_title('Autocorrelation (pol2)')
#                 axs[1, 1].legend()
                
#                 if fit1_success:
#                     fit_vals1 = amp1 * np.exp(-pos_lags1 / tau1)
#                     axs[0, 1].plot(pos_lags1, fit_vals1, 'g--', label='Exp Fit')
#                 if fit2_success:
#                     fit_vals2 = amp2 * np.exp(-pos_lags2 / tau2)
#                     axs[1, 1].plot(pos_lags2, fit_vals2, 'g--', label='Exp Fit')

#                 plt.tight_layout()
#                 plt.show()

#         except Exception as e:
#             print(f"❌ Failed for Cell {cell_id}: {e}")
#             continue

#     df_result = pd.DataFrame.from_records(records)
#     output_path = os.path.join(WORKING_DIR, filename)
#     df_result.to_csv(output_path, index=False)
#     print(f"✔️ Autocorrelation results saved to: {output_path}")

#     return df_result






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


