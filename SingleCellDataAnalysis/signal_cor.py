#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 10:36:04 2025

@author: user
"""
import numpy as np
from scipy.signal import correlate
from scipy.stats import norm
from scipy.stats import uniform
#from scipy.signal import find_peaks
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from scipy.stats import wasserstein_distance
import pandas as pd
import os as os
from scipy.optimize import curve_fit
from SingleCellDataAnalysis.config import WORKING_DIR
from sklearn.linear_model import LinearRegression
from SingleCellDataAnalysis.signal_analysis import (
    model_selector_with_threshold,
    
    )


def quantify_all_cells_xcor(df_all, cell_ids, feature1='pol1_int_corr', feature2='pol2_int_corr',
                             delta_threshold=4, filename='xcor_detrended_results.csv',
                             visualize=False):
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
            # Detrend both signals using model_selector_with_threshold
            y1_trend, _, _ = model_selector_with_threshold(t_valid, y1_valid, N_max=3, delta_threshold=delta_threshold)
            y2_trend, _, _ = model_selector_with_threshold(t_valid, y2_valid, N_max=3, delta_threshold=delta_threshold)
            y1_detrended = (y1_valid - y1_trend)[:, 0]
            y2_detrended = (y2_valid - y2_trend)[:, 0]

            # Cross-correlation
            xcor = correlate(y1_detrended, y2_detrended, mode='full')
            lags = np.arange(-len(y1_detrended) + 1, len(y1_detrended))
            center_idx = len(xcor) // 2
            xcor_zero_lag = xcor[center_idx]

            # Improved peak detection (positive lags only)
            xcor_temp = xcor.copy()
            xcor_temp[center_idx] = -np.inf  # mask zero-lag

            pos_lags = lags[center_idx + 1:]
            xcor_pos = xcor[center_idx + 1:]

            diffs = np.diff(xcor_pos)
            std_noise = np.std(diffs)

            peaks, props = find_peaks(xcor_pos, height=1.96 * std_noise)

            first_dip_idx = np.argmax(diffs < 0) + 1
            valid_peaks = [p for p in peaks if p > first_dip_idx]

            if valid_peaks:
                xcor_max_idx = valid_peaks[0] + center_idx + 1
            else:
                xcor_max_idx = np.argmax(xcor_temp)

            xcor_max = xcor[xcor_max_idx]
            xcor_lag = lags[xcor_max_idx]

            records.append({
                'cell_id': cell_id,
                'xcor_max': xcor_max,
                'xcor_lag': xcor_lag,
                'xcor_zero_lag': xcor_zero_lag
            })

            # --- Optional Visualization ---
            if visualize:
                fig, axs = plt.subplots(2, 2, figsize=(12, 8))

                axs[0, 0].plot(t_valid[:, 0], y1_valid[:, 0], label='Original')
                axs[0, 0].plot(t_valid[:, 0], y1_trend[:, 0], '--', label='Trend')
                axs[0, 0].plot(t_valid[:, 0], y1_detrended, label='Detrended')
                axs[0, 0].set_title(f'Cell {cell_id} - {feature1}')
                axs[0, 0].legend()

                axs[0, 1].plot(t_valid[:, 0], y2_valid[:, 0], label='Original')
                axs[0, 1].plot(t_valid[:, 0], y2_trend[:, 0], '--', label='Trend')
                axs[0, 1].plot(t_valid[:, 0], y2_detrended, label='Detrended')
                axs[0, 1].set_title(f'Cell {cell_id} - {feature2}')
                axs[0, 1].legend()

                axs[1, 0].plot(lags, xcor)
                axs[1, 0].axvline(x=0, color='k', linestyle='--', label='Zero Lag')
                axs[1, 0].axvline(x=xcor_lag, color='r', linestyle='--', label='Peak Lag')
                axs[1, 0].set_title('Cross-Correlation')
                axs[1, 0].legend()

                axs[1, 1].axis('off')

                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"❌ Failed for Cell {cell_id}: {e}")
            continue

    df_result = pd.DataFrame.from_records(records)

    output_path = filename
    df_result.to_csv(output_path, index=False)
    print(f"✔️ Cross-correlation results saved to: {output_path}")

    return df_result


from sklearn.metrics import mutual_info_score

def compute_mi_empirical(y_true, y_pred, n_bins=10):
    # Bin both signals
    y_true_binned = np.digitize(y_true, bins=np.histogram_bin_edges(y_true, bins=n_bins))
    y_pred_binned = np.digitize(y_pred, bins=np.histogram_bin_edges(y_pred, bins=n_bins))

    return mutual_info_score(y_true_binned, y_pred_binned)
# from npeet.entropy_estimators import mi as mi_ksg
# import numpy as np
# 
# def compute_mi_ksg(y_true, y_pred):
#     # Ensure shape (n_samples, 1)
#     x = y_true.reshape(-1, 1)
#     y = y_pred.reshape(-1, 1)
#     return mi_ksg(x, y)
def compute_sse(y_true, y_pred):
    return 1/np.sum((y_true - y_pred) ** 2)

def biological_frequency_prior(f, mu=0.07, sigma=0.02):
    return np.exp(-0.5 * ((f - mu) / sigma)**2)
def double_exp_osc_model(lags, A1, tau1, tau2, f, phi, C, acf0):
    A2 = np.clip(acf0 - A1 - C, 0.0, 1.0)
    envelope = A1 * np.exp(-lags / tau1) + A2 * np.exp(-lags / tau2) + C
    return envelope * np.cos(2 * np.pi * f * lags + phi)

def run_model_B(lags, acor_vals, n_iter=2000, proposal_width=0.2):
    acf0 = acor_vals[0]

    # Function factory with env toggle
    def fit_model_factory(env=False):
        def fit_model(lags, A1, tau1, tau2, f, phi, C):
            A2 = np.clip(acf0 - A1 - C, 0.0, 1.0)
            envelope = A1 * np.exp(-lags / tau1) + A2 * np.exp(-lags / tau2) + C
            if env:
                return envelope
            else:
                return envelope * np.cos(2 * np.pi * f * lags + phi)
        return fit_model

    # Initial curve fitting
    try:
        fit_model = fit_model_factory(env=False)
        popt, _ = curve_fit(
            fit_model, lags, acor_vals,
            p0=[max(acf0 * 0.3, 0.01), 5, 15, 0.05, 0.01, acf0 * 0.1],
            bounds=([0.0, 1e-2, 1e-2, 1e-3, 0, -np.inf],
                    [1.0, 1e3, 1e3, 1.0, 2*np.pi, np.inf])
        )
        print(f"[CurveFit B] Success: A1={popt[0]:.4f}, tau1={popt[1]:.2f}, tau2={popt[2]:.2f}, "
              f"f={popt[3]:.4f}, φ={popt[4]:.2f}, C={popt[5]:.4f}")
    except Exception as e:
        print(f"⚠️ [CurveFit B] Failed: {e}")
        popt = [0.0, 10.0, 10.0, 0.05, 0.0, 0.0]
    mi_values = []
    freq_range = np.linspace(0.01, 0.2, 500)
    for f in freq_range:
        params = [popt[0], popt[1], popt[2], f, popt[4], popt[5], acf0]
        mu_f = double_exp_osc_model(lags, *params)
        mi_f = compute_sse(acor_vals, mu_f) * biological_frequency_prior(f)
        mi_values.append(mi_f)
    
    mi_values = np.array(mi_values)
    max_idx = np.argmax(mi_values)
    best_freq = freq_range[max_idx]
    best_params = [popt[0], popt[1], popt[2], best_freq, popt[4], popt[5], acf0]
    return freq_range, mi_values, best_params
    # # Envelope extraction
    # envelope_func = fit_model_factory(env=True)
    # envelope_at_popt_f = envelope_func(lags, *popt)

    # safe_envelope = np.clip(envelope_at_popt_f, 1e-6, None)
    # log_evidence = np.sum(norm.logpdf(acor_vals, loc=np.mean(acor_vals),
    #                                   scale=np.std(acor_vals)))# * safe_envelope))

    # # Frequency sweep
    # freq_range = np.linspace(0.01, 0.1, 500)
    # log_posteriors = []
    # prior_logpf = 0  # flat prior for now

    # for f in freq_range:
    #     params = [popt[0], popt[1], popt[2], f, popt[4], popt[5], acf0]
    #     mu = double_exp_osc_model(lags, *params)
    #     log_likelihood = np.sum(norm.logpdf(acor_vals, loc=mu, scale=np.std(acor_vals)))
    #     log_posteriors.append(log_likelihood + prior_logpf - log_evidence)

    # log_posteriors = np.array(log_posteriors)
    # posteriors = np.exp(log_posteriors - np.max(log_posteriors))  # numerical stability
    # #posteriors /= np.sum(posteriors)

    # max_idx = np.argmax(posteriors)
    # best_freq = freq_range[max_idx]
    # best_params = [popt[0], popt[1], popt[2], best_freq, popt[4], popt[5], acf0]

    return freq_range, mi_values, best_params



def single_gaussian_osc_model(lags, A1, sigma, f, phi, C):
    envelope = A1 * np.exp(-(lags**2) / (2 * sigma**2)) + C
    return envelope * np.cos(2 * np.pi * f * lags + phi)

def run_model_C_symmetric(lags, cor_vals, n_iter=2000, proposal_width=0.2):
    # Function factory
    def fit_model_factory(env=False):
        def fit_model(lags, A1, sigma, f, phi, C):
            envelope = A1 * np.exp(-(lags**2) / (2 * sigma**2)) + C
            return envelope if env else envelope * np.cos(2 * np.pi * f * lags + phi)
        return fit_model

    acf0 = cor_vals[np.argmin(np.abs(lags))]

    try:
        fit_model = fit_model_factory(env=False)
        popt, _ = curve_fit(
            fit_model, lags, cor_vals,
            p0=[max(acf0 * 0.3, 0.01), 10, 0.05, 0.01, acf0 * 0.1],
            #bounds=([-np.inf, 1e-2, 1e-3, 0, -np.inf],
            #        [np.inf, 1e3, 1.0, 2 * np.pi, np.inf]),
            bounds=([0.0, 1e-2, 1e-3, 0, -1],
                    [1, 1e3, 1.0, 2 * np.pi, 1]),
            maxfev=10000
        )
        print(f"[CurveFit Symmetric] Success: A1={popt[0]:.4f}, σ={popt[1]:.2f}, "
              f"f={popt[2]:.4f}, φ={popt[3]:.2f}, C={popt[4]:.4f}")
    except Exception as e:
        print(f"⚠️ [CurveFit Symmetric] Failed: {e}")
        popt = [acf0 * 0.3, 10, 0.05, 0.0, acf0 * 0.1]

    # Compute envelope for log_evidence
    envelope_func = fit_model_factory(env=True)
    envelope_at_popt = envelope_func(lags, *popt)

    safe_envelope = np.clip(envelope_at_popt, 1e-6, None)
    log_evidence = np.sum(norm.logpdf(cor_vals, loc=np.mean(cor_vals),
                                      scale=np.std(cor_vals)))# * safe_envelope))
    mi_values = []
    freq_range = np.linspace(0.01, 0.2, 500)
    for f in freq_range:
        params = [popt[0], popt[1], f, popt[3], popt[4]]
        mu_f = single_gaussian_osc_model(lags, *params)
        mi_f = compute_sse(cor_vals, mu_f) * biological_frequency_prior(f)
        mi_values.append(mi_f)
    
    mi_values = np.array(mi_values)
    max_idx = np.argmax(mi_values)
    best_freq = freq_range[max_idx]
    best_params = [popt[0], popt[1], best_freq, popt[3], popt[4]]
    return freq_range, mi_values, best_params

    # # Posterior over frequency
    # freq_range = np.linspace(0.01, 0.1, 500)
    # log_posteriors = []
    # prior_logpf = np.log(500)

    # for f in freq_range:
    #     params = [popt[0], popt[1], f, popt[3], popt[4]]
    #     mu = single_gaussian_osc_model(lags, *params)
    #     log_likelihood = np.sum(norm.logpdf(cor_vals, loc=mu, scale=np.std(cor_vals)))
    #     log_posteriors.append(log_likelihood + prior_logpf - log_evidence)

    # log_posteriors = np.array(log_posteriors)
    # posteriors = np.exp(log_posteriors - np.max(log_posteriors))
    # posteriors /= np.sum(posteriors)
    
    
    # max_idx = np.argmax(posteriors)
    # best_freq = freq_range[max_idx]
    # best_params = [popt[0], popt[1], best_freq, popt[3], popt[4]]

    return freq_range, mi_values, best_params





def quantify_all_cells_acor(df_all, cell_ids,
                            feature1='pol1_int_corr', feature2='pol2_int_corr',
                            delta_threshold=4, filename='acor_detrended_results.csv',
                            visualize=False):
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
            
            # --- pol1 detrend and autocor ---
            y1_trend, _, _ = model_selector_with_threshold(t_valid, y1_valid, N_max=2, delta_threshold=delta_threshold)
            y1_detrended = (y1_valid - y1_trend)[:, 0]
            acor1 = correlate(y1_detrended, y1_detrended, mode='full')
            lags1 = np.arange(-len(y1_detrended) + 1, len(y1_detrended))
            center_idx1 = len(acor1) // 2
            acor1_zero = acor1[center_idx1]
            
            # ✅ Normalize autocorrelation with division safety check
            if acor1_zero > 1e-8:
                acor1 = acor1 / acor1_zero
            else:
                acor1 = np.zeros_like(acor1)
            acor1_pos = acor1[center_idx1 + 1:]
            pos_lags1 = lags1[center_idx1 + 1:]
            
            # Model fitting
            f1, p1, trace_1 = run_model_B(pos_lags1, acor1_pos, n_iter=2000, proposal_width=0.2)
            max_idx = np.argmax(p1)
            best_f1 = f1[max_idx]
            
            
            # --- pol2 detrend and autocor ---
            y2_trend, _, _ = model_selector_with_threshold(t_valid, y2_valid, N_max=2, delta_threshold=delta_threshold)
            y2_detrended = (y2_valid - y2_trend)[:, 0]
            acor2 = correlate(y2_detrended, y2_detrended, mode='full')
            lags2 = np.arange(-len(y2_detrended) + 1, len(y2_detrended))
            center_idx2 = len(acor2) // 2
            acor2_zero = acor2[center_idx2]
            
            # ✅ Normalize autocorrelation with division safety check
            if acor2_zero > 1e-8:
                acor2 = acor2 / acor2_zero
            else:
                acor2 = np.zeros_like(acor2)
            acor2_pos = acor2[center_idx2 + 1:]
            pos_lags2 = lags2[center_idx2 + 1:]
            
            # Model fitting
            f2, p2, trace_2 = run_model_B(pos_lags2, acor2_pos, n_iter=2000, proposal_width=0.2)
            max_idx = np.argmax(p2)
            best_f2 = f2[max_idx]
            
            
            # --- Cross-correlation (normalized) ---
            xcor = correlate(y1_detrended, y2_detrended, mode='full')
            lags = np.arange(-len(y1_detrended) + 1, len(y1_detrended))
            #center_idx = len(xcor) // 2
            
            # ✅ Normalize cross-correlation with division safety check
            norm_factor = np.linalg.norm(y1_detrended) * np.linalg.norm(y2_detrended)
            if norm_factor > 1e-8:
                xcor = xcor / norm_factor
            else:
                xcor = np.zeros_like(xcor)
            
            #xcor_zero_lag = xcor[center_idx]

            #xcor_pos = xcor[center_idx1 + 1:]
            #pos_lags = lags[center_idx1 + 1:]
            
            #fx,px,trace_x=run_model_B(pos_lags, xcor_pos, n_iter=2000, proposal_width=0.2)
            fx,px,trace_x=run_model_C_symmetric(lags, xcor)
            max_idx = np.argmax(px)
            best_fx = fx[max_idx]
            
            
            # Compute pairwise distances
            d1 = abs(best_f1 - best_f2)
            d2 = abs(best_f1 - best_fx)
            d3 = abs(best_f2 - best_fx)
            
            # Total distance sum
            freq_distance_sum = d1 + d2 + d3

            # Weighted distance between f1 and f2
            #diff12 = np.abs(f1[:, None] - f2[None, :])  # shape (len(f1), len(f2))
            #wdist12 = np.sum(p1[:, None] * p2[None, :] * diff12)
            # Weighted distance between f1 and fx
            #diff1x = np.abs(f1[:, None] - fx[None, :])
            #wdist1x = np.sum(p1[:, None] * px[None, :] * diff1x)
            # Weighted distance between f2 and fx
            #diff2x = np.abs(f2[:, None] - fx[None, :])
            #wdist2x = np.sum(p2[:, None] * px[None, :] * diff2x)
            # Total sum
            #weighted_distance_sum = wdist12 + wdist1x + wdist2x

            

            
            
            # Calculate pairwise distances
            p1 = np.clip(p1, 0, None)
            p2 = np.clip(p2, 0, None)
            px = np.clip(px, 0, None)

            #d12 = wasserstein_distance(f1, f2, u_weights=p1, v_weights=p2)
            #d1x = wasserstein_distance(f1, fx, u_weights=p1, v_weights=px)
            #d2x = wasserstein_distance(f2, fx, u_weights=p2, v_weights=px)
            
            # Sum or average
            #distance_sum = d12 + d1x + d2x
            
            # Horizontal lines at precision peaks (raw fit goodness)
            y_f1_raw = p1[np.argmin(np.abs(f1 - best_f1))]
            y_f2_raw = p2[np.argmin(np.abs(f2 - best_f2))]
            y_fx_raw = px[np.argmin(np.abs(fx - best_fx))]
            
            # Extract tau1, tau2 from trace_1 and calculate decay penalty
            tau1_1, tau2_1 = trace_1[1], trace_1[2]
            tau_eff_1 = max(tau1_1, tau2_1)
            decay_factor_1 = 1 - np.exp(-tau_eff_1 / 10.0)
            
            # Amplitude penalty for Pol1 (A of longer tau)
            A1_1 = trace_1[0]
            C_1 = trace_1[5]
            acf0_1 = trace_1[6]
            A2_1 = acf0_1 - A1_1 - C_1
            A_long_1 = A1_1 if tau1_1 >= tau2_1 else A2_1
            amp_factor_1 = 1 - np.exp(-max(A_long_1, 0.0) / 0.2)

            # Extract tau1, tau2 from trace_2 and calculate decay penalty
            tau1_2, tau2_2 = trace_2[1], trace_2[2]
            tau_eff_2 = max(tau1_2, tau2_2)
            decay_factor_2 = 1 - np.exp(-tau_eff_2 / 10.0)
            
            # Amplitude penalty for Pol2 (A of longer tau)
            A1_2 = trace_2[0]
            C_2 = trace_2[5]
            acf0_2 = trace_2[6]
            A2_2 = acf0_2 - A1_2 - C_2
            A_long_2 = A1_2 if tau1_2 >= tau2_2 else A2_2
            amp_factor_2 = 1 - np.exp(-max(A_long_2, 0.0) / 0.2)

            # Extract sigma from trace_x and calculate decay penalty
            sigma_x = trace_x[1]
            decay_factor_x = 1 - np.exp(-sigma_x / 10.0)
            
            # Amplitude penalty for xcor (A of cross-correlation)
            A_x = trace_x[0]
            amp_factor_x = 1 - np.exp(-max(A_x, 0.0) / 0.2)

            # Standard deviation penalty to suppress low-amplitude flat noise
            std_ref = 0.8
            std_penalty_1 = min(np.std(y1_detrended) / std_ref, 1.0)
            std_penalty_2 = min(np.std(y2_detrended) / std_ref, 1.0)
            std_penalty_x = np.sqrt(std_penalty_1 * std_penalty_2)

            # Detrend fit goodness penalty (relative SSE / unexplained variance ratio)
            ratio_ref = 0.3
            sst1 = np.sum((y1_valid[:, 0] - np.mean(y1_valid))**2)
            ratio1 = np.sum(y1_detrended**2) / sst1 if sst1 > 1e-8 else 0.0
            detrend_penalty_1 = min(ratio1 / ratio_ref, 1.0)

            sst2 = np.sum((y2_valid[:, 0] - np.mean(y2_valid))**2)
            ratio2 = np.sum(y2_detrended**2) / sst2 if sst2 > 1e-8 else 0.0
            detrend_penalty_2 = min(ratio2 / ratio_ref, 1.0)

            # Apply biological frequency prior after frequency selection
            prior_1 = biological_frequency_prior(best_f1)
            prior_2 = biological_frequency_prior(best_f2)
            prior_x = biological_frequency_prior(best_fx)

            y_f1_penalized = y_f1_raw * prior_1
            y_f2_penalized = y_f2_raw * prior_2
            y_fx_penalized = y_fx_raw * prior_x

            precision_sum = y_f1_penalized + y_f2_penalized # + y_fx_penalized
            raw_precision_sum = y_f1_raw + y_f2_raw
            
            # SSE of ACF variance from its envelope (baseline of non-oscillatory decay)
            A2_1 = np.clip(acf0_1 - A1_1 - C_1, 0.0, 1.0)
            env1 = A1_1 * np.exp(-pos_lags1 / tau1_1) + A2_1 * np.exp(-pos_lags1 / tau2_1) + C_1
            sse_zero_1 = np.sum((acor1_pos - env1) ** 2)

            A2_2 = np.clip(acf0_2 - A1_2 - C_2, 0.0, 1.0)
            env2 = A1_2 * np.exp(-pos_lags2 / tau1_2) + A2_2 * np.exp(-pos_lags2 / tau2_2) + C_2
            sse_zero_2 = np.sum((acor2_pos - env2) ** 2)

            # Weighted inverse baseline envelope fit errors (to match the dimension of precision_sum)
            term_1 = (prior_1 / sse_zero_1) if sse_zero_1 > 1e-8 else (prior_1 / 1e-8)
            term_2 = (prior_2 / sse_zero_2) if sse_zero_2 > 1e-8 else (prior_2 / 1e-8)
            sse_zero = term_1 + term_2
            log_zero = np.log(sse_zero) if sse_zero > 1e-8 else -10.0

            A, sigma, f, phi, C = trace_x
            score = -(A+C)#(np.exp(-(A+C))-1)*(sigma/10+np.abs(C))#+precision_sum#np.log(y_fx))
            # print(-(A+C))
            # print(np.log(-(A+C)))
            # print(np.log(sigma))
            # print(np.log(y_fx))
            
            records.append({
                'cell_id': cell_id,
                'freq_distance_sum': log_zero,
                'precision_sum': np.log(precision_sum),
                'raw_freq_distance_sum': sse_zero,
                'raw_precision_sum': raw_precision_sum,
                'NC_score': score,
                
                # Pol1 fit params
                'pol1_A1': A1_1,
                'pol1_tau1': tau1_1,
                'pol1_tau2': tau2_1,
                'pol1_f': best_f1,
                'pol1_phi': trace_1[4],
                'pol1_C': C_1,
                'pol1_acf0': acf0_1,

                # Pol2 fit params
                'pol2_A1': A1_2,
                'pol2_tau1': tau1_2,
                'pol2_tau2': tau2_2,
                'pol2_f': best_f2,
                'pol2_phi': trace_2[4],
                'pol2_C': C_2,
                'pol2_acf0': acf0_2,

                # Penalties and precision components
                'pol1_y_f': y_f1_raw,
                'pol1_freq_prior': prior_1,
                'pol1_decay_factor': decay_factor_1,
                'pol1_std_penalty': std_penalty_1,
                'pol1_detrend_penalty': detrend_penalty_1,
                'pol1_penalized': y_f1_penalized,

                'pol2_y_f': y_f2_raw,
                'pol2_freq_prior': prior_2,
                'pol2_decay_factor': decay_factor_2,
                'pol2_std_penalty': std_penalty_2,
                'pol2_detrend_penalty': detrend_penalty_2,
                'pol2_penalized': y_f2_penalized,
            })



            if visualize:
                from matplotlib import gridspec

                fig = plt.figure(figsize=(14, 12))
                gs = gridspec.GridSpec(3, 3, width_ratios=[1.5, 3, 0.8])
                
                axs = np.empty((3, 3), dtype=object)
                for i in range(3):
                    axs[i, 0] = fig.add_subplot(gs[i, 0])  # original + trend + detrended
                    axs[i, 1] = fig.add_subplot(gs[i, 1])  # acor/xcor
                    axs[i, 2] = fig.add_subplot(gs[i, 2], sharey=axs[i, 1])  # violin plot

                # ---- Feature 1 ----
                axs[0, 0].plot(t_valid[:, 0], y1_valid[:, 0], label='Original')
                axs[0, 0].plot(t_valid[:, 0], y1_trend[:, 0], '--', label='Trend')
                axs[0, 0].plot(t_valid[:, 0], y1_detrended, label='Detrended')
                axs[0, 0].set_title(f'Cell {cell_id} - {feature1}')
                #axs[0, 0].legend()
                
                axs[0, 1].plot(lags1, acor1)
                axs[0, 1].axvline(x=0, color='w', linestyle='--', label='Zero Lag')
                axs[0, 1].set_title('Autocorrelation (pol1)')
                # --- Equation below title ---
                if trace_1 is not None:
                    A1, tau1, tau2, f_1, phi1, C1, _ = trace_1  # or however your output is structured
                    eq1 = (
                        r"$A_1 e^{-t/\tau_1} + A_2 e^{-t/\tau_2} + C$"
                        r"$\,\cdot\,\cos(2\pi f t + \phi)$" "\n"
                        rf"$f$={f_1:.3f} Hz, $\tau_1$={tau1:.2f}, $\tau_2$={tau2:.2f}, $\phi$={phi1:.2f}, $C$={C1:.2f}"
                    )
                    axs[0, 1].text(0.5, 1.02, eq1, transform=axs[0, 1].transAxes,
                                   ha='center', va='top', fontsize=9)
                #axs[0, 1].legend()
                
                # ---- Feature 2 ----
                axs[1, 0].plot(t_valid[:, 0], y2_valid[:, 0], label='Original')
                axs[1, 0].plot(t_valid[:, 0], y2_trend[:, 0], '--', label='Trend')
                axs[1, 0].plot(t_valid[:, 0], y2_detrended, label='Detrended')
                axs[1, 0].set_title(f'Cell {cell_id} - {feature2}')
                #axs[1, 0].legend()
                
                axs[1, 1].plot(lags2, acor2)
                axs[1, 1].axvline(x=0, color='w', linestyle='--', label='Zero Lag')
                axs[1, 1].set_title('Autocorrelation (pol2)')
                # --- Equation below title ---
                if trace_2 is not None:
                    A2, tau1_2, tau2_2, f_2, phi2, C2, _ = trace_2
                    eq2 = (
                        r"$A_1 e^{-t/\tau_1} + A_2 e^{-t/\tau_2} + C$"
                        r"$\,\cdot\,\cos(2\pi f t + \phi)$" "\n"
                        rf"$f$={f_2:.3f} Hz, $\tau_1$={tau1_2:.2f}, $\tau_2$={tau2_2:.2f}, $\phi$={phi2:.2f}, $C$={C2:.2f}"
                    )
                    axs[1, 1].text(0.5, 1.02, eq2, transform=axs[1, 1].transAxes,
                                   ha='center', va='top', fontsize=9)
                #axs[1, 1].legend()

                
                axs[2, 1].plot(lags, xcor)
                axs[2, 1].axvline(x=0, color='w', linestyle='--', label='Zero Lag')
                pred_b = single_gaussian_osc_model(lags, *trace_x)
                axs[2, 1].plot(lags, pred_b, 'r-', label='Fit of MAP f')
                
                # Equation string with fitted parameters
                
                equation_str = (
                    r"$A\,e^{-\frac{{\tau^2}}{{2\sigma^2}}}\cos(2\pi f\tau + \phi) + C$" "\n"
                    rf"$A$={A:.2f}, $\sigma$={sigma:.2f}, $f$={f:.3f} Hz, "
                    rf"$\phi$={phi:.2f} rad, $C$={C:.2f}"
                )
                # Display below title (adjust y and size as needed)
                axs[2, 1].text(0.5, 1.02, equation_str, transform=axs[2, 1].transAxes,
                               ha='center', va='top', fontsize=9)                
                
                axs[2, 1].set_title(f'Crosscorrelation| NC score:{score:.3f}')
                #axs[2, 1].legend()

                
             # Plot the posterior distributions
                               # Calculate area under each MI curve (approximate sum)
                #mi_sum1 = np.trapz(p1, f1)  # area under MI curve for Pol1
                #mi_sum2 = np.trapz(p2, f2)  # for Pol2
                #mi_sumx = np.trapz(px, fx)  # for Xcor
                #mi_total = mi_sum1 + mi_sum2 + mi_sumx
                #mi_total = np.argmax(p1) + np.argmax(p2) + np.argmax(px)
                
                # Plot MI curves
                axs[2, 0].plot(f1, p1, label='Pol1 MI$(f)$', color='cyan')
                axs[2, 0].plot(f2, p2, label='Pol2 MI$(f)$', color='yellow')
                axs[2, 0].plot(fx, px, label='Xcor MI$(f)$', color='red')
                
                # Vertical MAP f lines (same)
                axs[2, 0].axvline(best_f1, color='cyan', linestyle=':', label=f'MAP f = {best_f1:.4f} Hz')
                axs[2, 0].axvline(best_f2, color='yellow', linestyle=':', label=f'MAP f = {best_f2:.4f} Hz')
                axs[2, 0].axvline(best_fx, color='red', linestyle=':', label=f'MAP f = {best_fx:.4f} Hz')
                
               
                axs[2, 0].axhline(y_f1, color='cyan', linestyle=':', xmax=best_f1 / max(f1))
                axs[2, 0].axhline(y_f2, color='yellow', linestyle=':', xmax=best_f2 / max(f2))
                axs[2, 0].axhline(y_fx, color='red', linestyle=':', xmax=best_fx / max(fx))
                
                # Dummy entry showing MI sum
                axs[2, 0].plot([], [], ' ', label=f"Precision Sum = {precision_sum:.4f}")
                # Dummy legend entries for scalar metrics
                #axs[2, 0].plot([], [], ' ', label=f"Weighted Distance Sum = {weighted_distance_sum:.4f}")
                axs[2, 0].plot([], [], ' ', label=f"Distance Sum = {freq_distance_sum:.4f}")
                periodicity = np.log(precision_sum)-np.log(freq_distance_sum)
                axs[2, 0].plot([], [], ' ', label=f"Periodicity log(P/D) = {periodicity:.4f}")
                
               
                
                # Title and axis label
                axs[2, 0].set_title("Precision over Frequency")
                axs[2, 0].set_ylabel("1/SSR")
                axs[2, 0].set_xlabel("Frequency (Hz)")
                
                # Remove background grid
                # axs[2, 0].grid(True, linestyle="--", alpha=0.6)
                
                # Legend
                axs[2, 0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.15),
                                 ncol=3, fontsize='small', frameon=False)

                
                
                
                # Remove background grid
                # axs[2, 0].grid(True, linestyle="--", alpha=0.6)  ← comment this out
                
                # Consolidated legend
                axs[2, 0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.15),
                                 ncol=3, fontsize='small', frameon=False)

                def plot_fit(ax, lags, acor, trace_B, title, label, color):
                    ax.plot(lags, acor, label=label + ' Raw', color=color, alpha=0.3, lw=1.5)
                    ax.axvline(x=0, color='k', linestyle='--', lw=1)
                
                    # Exponential model (updated for double exponential)
                    acf0 = acor[0]
                    pred_b = double_exp_osc_model(lags, *trace_B)
                    ax.plot(lags, pred_b, color=color, linestyle='--', label=label + ' Fit', lw=1.5)
                    ax.set_title(title)
                    #ax.legend()
             
                plot_fit(axs[0, 1], pos_lags1, acor1_pos, trace_1, 'pol1 Best Fit', 'Autocorrelation', color='#ef4444')
                plot_fit(axs[1, 1], pos_lags2, acor2_pos, trace_2, 'pol2 Best Fit', 'Autocorrelation', color='#3b82f6')
                #plot_fit(axs[2, 1], pos_lags, xcor_pos, trace_x, 'Xcor Best Fit', 'Crosscorrelation')


                axs[0, 2].violinplot(acor1, vert=True)
                axs[0, 2].set_title("Acor1 Dist")
                
                axs[1, 2].violinplot(acor2, vert=True)
                axs[1, 2].set_title("Acor2 Dist")
                
                axs[2, 2].violinplot(xcor, vert=True)
                axs[2, 2].set_title("Xcor Dist")



                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"❌ Failed for Cell {cell_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    df_result = pd.DataFrame.from_records(records)
    output_path = filename
    df_result.to_csv(output_path, index=False)
    print(f"✔️ Autocorrelation results saved to: {output_path}")

    return df_result
def plot_perodicity_result(periodicity_result):
    
    
    # If you want to reverse the log transform, uncomment these lines:
    # periodicity_result['freq_distance_sum'] = np.exp(periodicity_result['freq_distance_sum'])
    # periodicity_result['precision_sum'] = np.exp(periodicity_result['precision_sum'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(periodicity_result['freq_distance_sum'], periodicity_result['precision_sum'], color='blue')
    
    # Add cell_id labels
    for _, row in periodicity_result.iterrows():
        ax.annotate(
            str(int(row['cell_id'])),
            (row['freq_distance_sum'], row['precision_sum']),
            textcoords="offset points",
            xytext=(3, 3),
            ha='left',
            fontsize=8
        )
    
    # Axis labels and title
    ax.set_xlabel("Log Frequency Distance Sum")
    ax.set_ylabel("Log Precision Sum")
    ax.set_title("Periodicity Consistency vs. Precision (per Cell ID)")
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def plot_perodicity_vs_NC(periodicity_result):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute x values once
    x = periodicity_result['precision_sum'] - periodicity_result['freq_distance_sum'] 
    y = periodicity_result['NC_score']
    
    # Scatter plot
    ax.scatter(x, y, color='blue')

    # Add cell_id labels
    for _, row in periodicity_result.iterrows():
        cell_x = row['precision_sum'] - row['freq_distance_sum'] 
        cell_y = row['NC_score']
        ax.annotate(
            str(int(row['cell_id'])),
            (cell_x, cell_y),
            textcoords="offset points",
            xytext=(3, 3),
            ha='left',
            fontsize=8
        )

    ax.set_xlabel("Periodicity Score")
    ax.set_ylabel("NC Score")
    ax.set_title("Periodicity vs. Negative Correlation (per Cell ID)")

    plt.grid(True)
    plt.tight_layout()
    plt.show()


# def double_gaussian_osc_model(lags, A1, sigma1, sigma2, f, phi, C, acf0):
#     A2 = acf0 - A1 - C
#     envelope = A1 * np.exp(-(lags**2) / (2 * sigma1**2)) + A2 * np.exp(-(lags**2) / (2 * sigma2**2)) + C
#     return envelope * np.cos(2 * np.pi * f * lags + phi)

# def run_model_C_symmetric(lags, cor_vals, n_iter=2000, proposal_width=0.2):
#     acf0 = cor_vals[np.argmin(np.abs(lags))]  # value at lag ~ 0

#     # Function factory
#     def fit_model_factory(env=False):
#         def fit_model(lags, A1, sigma1, sigma2, f, phi, C):
#             A2 = acf0 - A1 - C
#             envelope = A1 * np.exp(-(lags**2) / (2 * sigma1**2)) + A2 * np.exp(-(lags**2) / (2 * sigma2**2)) + C
#             if env:
#                 return envelope
#             else:
#                 return envelope * np.cos(2 * np.pi * f * lags + phi)
#         return fit_model

#     try:
#         # Use the env=False version for curve fitting
#         fit_model = fit_model_factory(env=False)
#         popt, _ = curve_fit(
#             fit_model, lags, cor_vals,
#             p0=[acf0 * 0.3, 5, 15, 0.05, 0.0, acf0 * 0.1],
#             bounds=([0, 1e-2, 1e-2, 1e-3, 0, 0],
#                     [acf0, 1e3, 1e3, 1.0, 2*np.pi, acf0])
#         )
#         print(f"[CurveFit Symmetric] Success: A1={popt[0]:.4f}, σ1={popt[1]:.2f}, σ2={popt[2]:.2f}, "
#               f"f={popt[3]:.4f}, φ={popt[4]:.2f}, C={popt[5]:.4f}")
#     except Exception as e:
#         print(f"⚠️ [CurveFit Symmetric] Failed: {e}")
#         popt = [acf0 * 0.3, 5, 15, 0.05, 0.0, acf0 * 0.1]

#     # Envelope for log_evidence
#     envelope_func = fit_model_factory(env=True)
#     envelope_at_popt_f = envelope_func(lags, *popt)

#     # Numerical stability for very small envelope values
#     safe_envelope = np.clip(envelope_at_popt_f, 1e-6, None)

#     # Evidence with scaled uncertainty
#     log_evidence = np.sum(norm.logpdf(cor_vals, loc=np.mean(cor_vals),
#                                       scale=np.std(cor_vals) * safe_envelope))

#     # Bayesian sweep over frequency
#     freq_range = np.linspace(0.01, 0.5, 500)
#     log_posteriors = []
#     prior_logpf = 0  # assuming flat prior for now

#     for f in freq_range:
#         params = [popt[0], popt[1], popt[2], f, popt[4], popt[5], acf0]
#         mu = double_gaussian_osc_model(lags, *params)
#         log_likelihood = np.sum(norm.logpdf(cor_vals, loc=mu, scale=np.std(cor_vals)))
#         log_posteriors.append(log_likelihood + prior_logpf - log_evidence)

#     log_posteriors = np.array(log_posteriors)
#     posteriors = np.exp(log_posteriors - np.max(log_posteriors))  # for stability
#     posteriors /= np.sum(posteriors)

#     max_idx = np.argmax(posteriors)
#     best_freq = freq_range[max_idx]
#     best_params = [popt[0], popt[1], popt[2], best_freq, popt[4], popt[5], acf0]

#     return freq_range, posteriors, best_params

