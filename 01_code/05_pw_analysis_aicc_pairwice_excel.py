#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Piecewise Linear Regression (PWLD) with Upper Envelope and Multivariate NNLS
============================================================================

This script performs a segmented power‑law analysis of CPU benchmark data.
It:
  1. Loads and cleans the data.
  2. Finds the optimal number of piecewise linear segments in log‑log space (AICc).
  3. Optionally bootstraps breakpoints and slopes (for 2‑3 segments).
  4. Derives univariate power formulas for each segment (ready for Excel).
  5. Evaluates the univariate PWLD model (R², adj.R², RMSE, MAPE).
  6. Builds segment‑wise multivariate non‑negative least squares (NNLS) models.
  7. Computes the upper envelope (maximum in a sliding window).
  8. Plots the results and saves everything to ../03_data/pw/ .

All terminal output is simultaneously written to a log file.
All tabular results are saved as CSV files; formulas are saved as TXT files.
"""

import sys
import os
import pandas as pd
import numpy as np
import pwlf
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# ===================== 0. SETUP & LOGGING =====================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(script_dir, '..', '03_data'))
output_dir = os.path.join(data_dir, 'pw')
os.makedirs(output_dir, exist_ok=True)

# Redirect stdout to both console and log file
log_file = os.path.join(output_dir, 'analysis_log.txt')

class Tee:
    """Duplicate output to multiple file objects."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

original_stdout = sys.stdout
log_handle = open(log_file, 'w', encoding='utf-8')
sys.stdout = Tee(original_stdout, log_handle)

# Input file
file_name = 'openbenchmarking - export.csv'
file_path = os.path.join(data_dir, file_name)

target = 'Natoms*ns/day'
x_param = 'GFLOPS'

candidate_features = ['GFLOPS', 'Mem Bandwidth. GB/s', 'Cache L2. KB', 'Cache L3. KB']

# Upper envelope parameters
use_max_for_envelope = True          # use max in window
neighborhood_ratio = 0.2
n_envelope_points = 200

# Multivariate regression parameters
force_nonnegative = True
min_points_for_multivariate = 3

# ===================== 1. LOAD AND CLEAN DATA =====================
df = pd.read_csv(file_path)

df[x_param] = pd.to_numeric(df[x_param], errors='coerce')
df[target] = pd.to_numeric(df[target], errors='coerce')
df = df.dropna(subset=[x_param, target])
df = df[(df[x_param] > 0) & (df[target] > 0)]

print(f"\nLoaded points: {len(df)}")
if len(df) < 3:
    print("Insufficient data.")
    sys.exit(1)

x = df[x_param].values
y = df[target].values

logx = np.log10(x)
logy = np.log10(y)

order = np.argsort(logx)
logx_sorted = logx[order]
logy_sorted = logy[order]

# ===================== 2. UNIVARIATE PWLD (AICc + BOOTSTRAP) =====================
def find_optimal_segments_aicc(logx, logy, max_segments=3, min_break_distance=0.15):
    """
    Find optimal number of piecewise linear segments using AICc.

    Parameters
    ----------
    logx, logy : array-like
        Data in log10 space.
    max_segments : int
        Maximum number of segments to test.
    min_break_distance : float
        Minimal allowed distance between breakpoints in log10 units.

    Returns
    -------
    tuple (n_segments, pwlf_model, breakpoints, AICc) or None
    """
    results = []
    n_data = len(logx)
    for n_seg in range(1, max_segments + 1):
        try:
            model = pwlf.PiecewiseLinFit(logx, logy)
            breaks = model.fit(n_seg)
            internal = breaks[1:-1]
            if len(internal) > 1:
                if np.min(np.diff(internal)) < min_break_distance:
                    print(f"Segments {n_seg}: breakpoints too close, skipping")
                    continue
            y_pred = model.predict(logx)
            rss = np.sum((logy - y_pred) ** 2)
            n_params = 2 * n_seg
            aic = n_data * np.log(rss / n_data) + 2 * n_params
            if n_data - n_params - 1 > 0:
                aicc = aic + (2 * n_params * (n_params + 1)) / (n_data - n_params - 1)
            else:
                aicc = np.inf
            results.append((n_seg, model, breaks, aicc))
            print(f"Segments: {n_seg}, AICc = {aicc:.1f}")
        except Exception as e:
            print(f"Error for {n_seg} segments: {e}")
    if not results:
        return None
    return min(results, key=lambda x: x[3])

best = find_optimal_segments_aicc(logx_sorted, logy_sorted, max_segments=5)
if best is None:
    print("Failed to build PWLD model.")
    sys.exit(1)

best_n_seg, best_model, best_breaks, best_score = best
print(f"\nOptimal number of segments: {best_n_seg}")
if best_n_seg > 1:
    breaks_gflops = 10 ** best_breaks[1:-1]
    slopes = best_model.calc_slopes()
    print(f"Breakpoints (GFLOPS): {breaks_gflops.round(0).astype(int)}")
    print(f"Slopes (log-log): {np.round(slopes, 3)}")
else:
    breaks_gflops = []
    slopes = best_model.calc_slopes()
    print(f"Single‑segment model: slope = {slopes[0]:.3f}")

# Save breakpoints and slopes to CSV
breakpoints_df = pd.DataFrame({
    'segment_index': list(range(best_n_seg)),
    'slope_loglog': slopes,
    'breakpoint_GFLOPS': [breaks_gflops[i] if i < len(breaks_gflops) else None for i in range(best_n_seg-1)] + [None],
    'n_segments': best_n_seg,
    'AICc': best_score
})
breakpoints_csv = os.path.join(output_dir, 'pwld_breakpoints.csv')
breakpoints_df.to_csv(breakpoints_csv, index=False)
print(f"Breakpoints and slopes saved to: {os.path.basename(breakpoints_csv)}")

# Bootstrap for optimal model (only for 2 or 3 segments)
if best_n_seg > 1 and best_n_seg <= 3:
    print(f"\nBootstrap (200 iterations) for {best_n_seg} segments:")
    def bootstrap_optimal_model(logx, logy, n_seg, n_bootstrap=100):
        n = len(logx)
        all_breaks = []
        all_slopes = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            x_bs = logx[idx]
            y_bs = logy[idx]
            try:
                model = pwlf.PiecewiseLinFit(x_bs, y_bs)
                breaks = model.fit(n_seg)
                slopes = model.calc_slopes()
                all_breaks.append(breaks)
                all_slopes.append(slopes)
            except:
                continue
        return np.array(all_breaks), np.array(all_slopes)

    all_breaks, all_slopes = bootstrap_optimal_model(logx_sorted, logy_sorted, best_n_seg, 200)
    bootstrap_data = []
    if len(all_breaks) > 0:
        for j in range(best_n_seg - 1):
            bp_log = all_breaks[:, j+1]
            bp_median = 10**np.median(bp_log)
            bp_ci_low = 10**np.percentile(bp_log, 2.5)
            bp_ci_high = 10**np.percentile(bp_log, 97.5)
            slope_left = all_slopes[:, j]
            slope_right = all_slopes[:, j+1]
            bootstrap_data.append({
                'breakpoint_index': j+1,
                'breakpoint_GFLOPS_median': bp_median,
                'breakpoint_CI_low': bp_ci_low,
                'breakpoint_CI_high': bp_ci_high,
                'slope_left_mean': np.mean(slope_left),
                'slope_left_std': np.std(slope_left),
                'slope_right_mean': np.mean(slope_right),
                'slope_right_std': np.std(slope_right)
            })
            print(f"  Break {j+1}: {bp_median:.0f} GFLOPS [95% CI: {bp_ci_low:.0f}–{bp_ci_high:.0f}], "
                  f"slope: {np.mean(slope_left):.2f}±{np.std(slope_left):.2f} → {np.mean(slope_right):.2f}±{np.std(slope_right):.2f}")
        bootstrap_df = pd.DataFrame(bootstrap_data)
        bootstrap_csv = os.path.join(output_dir, 'pwld_bootstrap.csv')
        bootstrap_df.to_csv(bootstrap_csv, index=False)
        print(f"Bootstrap results saved to: {os.path.basename(bootstrap_csv)}")
    else:
        print("  Bootstrap failed.")

# ===================== 3. UNIVARIATE POWER FORMULAS (Excel) & METRICS =====================
def get_intercept_for_segment(segment_idx, slopes, breaks_gflops, logx_all, logy_all):
    """
    Calculate intercept for a given segment of the piecewise linear model.

    Parameters
    ----------
    segment_idx : int
        Index of the segment (0‑based).
    slopes : list or array
        Slopes of all segments.
    breaks_gflops : list
        Breakpoints in original GFLOPS units.
    logx_all, logy_all : array
        Full log10 transformed data.

    Returns
    -------
    float : intercept in log10 space.
    """
    if len(breaks_gflops) == 0:
        return np.mean(logy_all - slopes[0] * logx_all)
    log_breaks = np.log10(breaks_gflops)
    seg_indices = np.digitize(logx_all, log_breaks)
    mask = (seg_indices == segment_idx)
    if np.any(mask):
        return np.mean(logy_all[mask] - slopes[segment_idx] * logx_all[mask])
    else:
        if segment_idx == 0:
            return np.mean(logy_all - slopes[0] * logx_all)
        else:
            prev_int = get_intercept_for_segment(segment_idx-1, slopes, breaks_gflops, logx_all, logy_all)
            prev_slope = slopes[segment_idx-1]
            bp_log = np.log10(breaks_gflops[segment_idx-1])
            return prev_int + (prev_slope - slopes[segment_idx]) * bp_log

# Compute predicted values for the entire dataset
y_pred_log = best_model.predict(logx_sorted)
y_pred = 10 ** y_pred_log
y_actual = 10 ** logy_sorted

# Univariate model quality metrics
r2 = r2_score(logy_sorted, y_pred_log)
n = len(logy_sorted)
p = best_n_seg + max(0, best_n_seg - 1)  # slopes + breakpoints
if n - p - 1 > 0:
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
else:
    r2_adj = np.nan
rmse_log = np.sqrt(mean_squared_error(logy_sorted, y_pred_log))
rmse_orig = np.sqrt(mean_squared_error(y_actual, y_pred))
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

print("\nUnivariate PWLD model quality metrics:")
print(f"  R² (log) = {r2:.4f}")
print(f"  Adjusted R² = {r2_adj:.4f}")
print(f"  RMSE (log) = {rmse_log:.4f}")
print(f"  RMSE (original) = {rmse_orig:.2e}")
print(f"  MAPE = {mape:.2f}%")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'metric': ['R2_log', 'R2_adj', 'RMSE_log', 'RMSE_original', 'MAPE_percent'],
    'value': [r2, r2_adj, rmse_log, rmse_orig, mape]
})
metrics_csv = os.path.join(output_dir, 'univariate_metrics.csv')
metrics_df.to_csv(metrics_csv, index=False)
print(f"Univariate metrics saved to: {os.path.basename(metrics_csv)}")

# Univariate formulas (text)
print("\nUnivariate regression formulas (for Excel):")
formulas = []
if best_n_seg == 1:
    intercept = get_intercept_for_segment(0, slopes, [], logx_sorted, logy_sorted)
    line1 = f"LOG10({target}) = {slopes[0]:.4f} * LOG10(GFLOPS) + {intercept:.4f}"
    line2 = f"{target} = 10^{intercept:.4f} * (GFLOPS)^{slopes[0]:.4f}"
    print(f"  {line1}")
    print(f"  {line2}")
    formulas.extend([line1, line2])
else:
    intervals = []
    intervals.append((f"GFLOPS ≤ {breaks_gflops[0]:.0f}", 0))
    for i in range(len(breaks_gflops)-1):
        intervals.append((f"{breaks_gflops[i]:.0f} < GFLOPS ≤ {breaks_gflops[i+1]:.0f}", i+1))
    intervals.append((f"GFLOPS > {breaks_gflops[-1]:.0f}", best_n_seg-1))
    for desc, idx in intervals:
        intercept = get_intercept_for_segment(idx, slopes, breaks_gflops, logx_sorted, logy_sorted)
        formulas.append(desc)
        formulas.append(f"  LOG10(y) = {slopes[idx]:.4f} * LOG10(GFLOPS) + {intercept:.4f}")
        formulas.append(f"  y = 10^{intercept:.4f} * (GFLOPS)^{slopes[idx]:.4f}")
        print(f"  {desc}:")
        print(f"    LOG10(y) = {slopes[idx]:.4f} * LOG10(GFLOPS) + {intercept:.4f}")
        print(f"    y = 10^{intercept:.4f} * (GFLOPS)^{slopes[idx]:.4f}")

formulas_txt = os.path.join(output_dir, 'pwld_formulas.txt')
with open(formulas_txt, 'w', encoding='utf-8') as f:
    f.write("\n".join(formulas))
print(f"Univariate formulas saved to: {os.path.basename(formulas_txt)}")

# ===================== 4. SEGMENTED MULTIVARIATE NNLS REGRESSION =====================
multi_features = [f for f in candidate_features if f in df.columns]
print("\nFeatures found in data:", multi_features)

multivariate_results = []  # to collect data for CSV and TXT

if len(multi_features) < 2:
    print("\nNeed at least 2 features for multivariate regression. Skipping this section.")
else:
    print("\n" + "="*70)
    print("SEGMENTED MULTIVARIATE POWER LAW REGRESSION (log‑log)")
    if force_nonnegative:
        print("(coefficients forced non‑negative via NNLS)")
    print("="*70)

    def build_multivariate_formula(df_sub, features, target_col, min_points=3):
        """
        Build a multivariate power‑law model using non‑negative least squares.

        Parameters
        ----------
        df_sub : DataFrame
            Subset of data (e.g., one segment).
        features : list
            Names of predictor features.
        target_col : str
            Name of the target column.
        min_points : int
            Minimum number of valid points required.

        Returns
        -------
        model : object or None
            Object with attributes .params (intercept + coefficients),
            .rsquared, .rsquared_adj.
        used_features : list or None
            Feature names actually used.
        df_clean : DataFrame or None
            Cleaned data used for fitting.
        """
        df_clean = df_sub[features + [target_col]].dropna()
        for f in features:
            df_clean = df_clean[df_clean[f] > 0]
        df_clean = df_clean[df_clean[target_col] > 0]
        if len(df_clean) < min_points:
            return None, None, None
        X_log = np.log10(df_clean[features].values)
        y_log = np.log10(df_clean[target_col].values)
        X_with_const = np.column_stack([np.ones(len(X_log)), X_log])
        coeff_full, _ = nnls(X_with_const, y_log)
        intercept = coeff_full[0]
        coeff = np.maximum(coeff_full[1:], 0)
        y_pred_log = intercept + X_log @ coeff
        ss_res = np.sum((y_log - y_pred_log) ** 2)
        ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2 = max(0.0, min(1.0, r2))
        n = len(y_log)
        p = len(coeff)
        if n - p - 1 > 0:
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            r2_adj = max(0.0, min(1.0, r2_adj))
        else:
            r2_adj = np.nan
        class Result:
            pass
        model_sel = Result()
        model_sel.params = np.concatenate([[intercept], coeff])
        model_sel.rsquared = r2
        model_sel.rsquared_adj = r2_adj
        return model_sel, features, df_clean

    if best_n_seg == 1 or len(breaks_gflops) == 0:
        print("\nSingle segment (all data):")
        model, used_feat, _ = build_multivariate_formula(df, multi_features, target, min_points=min_points_for_multivariate)
        if model:
            coeff = model.params
            const = 10 ** coeff[0]
            formula = f"{target} = {const:.4e}"
            for i, f in enumerate(used_feat):
                formula += f" * ({f})^{coeff[i+1]:.4f}"
            print(f"  Formula: {formula}")
            print(f"  R² = {model.rsquared:.4f}, adj.R² = {model.rsquared_adj:.4f}")
            multivariate_results.append({
                'segment': 'all_data',
                'formula': formula,
                'R2': model.rsquared,
                'R2_adj': model.rsquared_adj,
                'n_points': len(df),
                'coefficients': str(coeff.tolist()),
                'features': str(used_feat)
            })
        else:
            print("  Failed to build model.")
    else:
        df_segmented = df.copy()
        seg_labels = np.digitize(df_segmented[x_param].values, breaks_gflops, right=False)
        df_segmented['segment'] = seg_labels
        segment_descs = []
        segment_descs.append((0, f"GFLOPS ≤ {breaks_gflops[0]:.0f}"))
        for i in range(len(breaks_gflops)-1):
            segment_descs.append((i+1, f"{breaks_gflops[i]:.0f} < GFLOPS ≤ {breaks_gflops[i+1]:.0f}"))
        segment_descs.append((best_n_seg-1, f"GFLOPS > {breaks_gflops[-1]:.0f}"))
        for seg_id, desc in segment_descs:
            df_seg = df_segmented[df_segmented['segment'] == seg_id]
            if len(df_seg) < min_points_for_multivariate:
                print(f"\nSegment {desc}: insufficient points ({len(df_seg)}), skipping.")
                continue
            print(f"\n--- Segment: {desc} (points: {len(df_seg)}) ---")
            model, used_feat, _ = build_multivariate_formula(df_seg, multi_features, target, min_points=min_points_for_multivariate)
            if model:
                coeff = model.params
                const = 10 ** coeff[0]
                formula = f"{target} = {const:.4e}"
                for i, f in enumerate(used_feat):
                    formula += f" * ({f})^{coeff[i+1]:.4f}"
                print(f"  Formula: {formula}")
                print(f"  R² = {model.rsquared:.4f}, adj.R² = {model.rsquared_adj:.4f}")
                multivariate_results.append({
                    'segment': desc,
                    'formula': formula,
                    'R2': model.rsquared,
                    'R2_adj': model.rsquared_adj,
                    'n_points': len(df_seg),
                    'coefficients': str(coeff.tolist()),
                    'features': str(used_feat)
                })
            else:
                print("  Failed to build model.")

    # Save multivariate results to CSV
    if multivariate_results:
        mv_df = pd.DataFrame(multivariate_results)
        mv_csv = os.path.join(output_dir, 'multivariate_segments.csv')
        mv_df.to_csv(mv_csv, index=False)
        print(f"Multivariate segment results (CSV) saved to: {os.path.basename(mv_csv)}")

        # Save multivariate formulas to a text file (analogous to pwld_formulas.txt)
        mv_txt = os.path.join(output_dir, 'multivariate_formulas.txt')
        with open(mv_txt, 'w', encoding='utf-8') as f:
            for res in multivariate_results:
                f.write(f"Segment: {res['segment']}\n")
                f.write(f"Formula: {res['formula']}\n")
                f.write(f"R² = {res['R2']:.4f}, adj.R² = {res['R2_adj']:.4f}\n")
                f.write(f"Points: {res['n_points']}\n")
                f.write(f"Coefficients (log10): {res['coefficients']}\n")
                f.write(f"Features: {res['features']}\n")
                f.write("-" * 60 + "\n")
        print(f"Multivariate formulas (TXT) saved to: {os.path.basename(mv_txt)}")

# ===================== 5. UPPER ENVELOPE (MAX IN SLIDING WINDOW) =====================
def upper_envelope(x, y, n_points=200, window_ratio=0.2):
    """
    Compute upper envelope by taking maximum (or percentile) in a sliding window.

    Parameters
    ----------
    x, y : array-like
        Original data (positive).
    n_points : int
        Number of points for the envelope curve.
    window_ratio : float
        Window width relative to the current x: [x/(1+ratio), x*(1+ratio)].

    Returns
    -------
    x_grid, y_envelope : ndarray
    """
    x_sorted = np.sort(x)
    log_x = np.log10(x_sorted)
    log_x_grid = np.linspace(log_x.min(), log_x.max(), n_points)
    x_grid = 10 ** log_x_grid
    y_env = np.zeros_like(x_grid)
    for i, xc in enumerate(x_grid):
        lower = xc / (1 + window_ratio)
        upper = xc * (1 + window_ratio)
        mask = (x >= lower) & (x <= upper)
        if np.any(mask):
            if use_max_for_envelope:
                y_env[i] = np.max(y[mask])
            else:
                y_env[i] = np.percentile(y[mask], 95)
        else:
            idx = np.argmin(np.abs(x - xc))
            y_env[i] = y[idx]
    return x_grid, y_env

print("\nBuilding upper envelope (maximum in window)...")
x_env, y_env = upper_envelope(x, y, n_points=n_envelope_points, window_ratio=neighborhood_ratio)

# Save envelope points to CSV
envelope_df = pd.DataFrame({'GFLOPS': x_env, target: y_env})
envelope_csv = os.path.join(output_dir, 'upper_envelope.csv')
envelope_df.to_csv(envelope_csv, index=False)
print(f"Upper envelope points saved to: {os.path.basename(envelope_csv)}")

# ===================== 6. PLOT WITH REGRESSION AND ENVELOPE =====================
def format_power(value, pos):
    """Format tick labels for log axes."""
    if value <= 0:
        return ''
    logv = np.log10(value)
    if logv < 4:
        return f'{int(value)}'
    else:
        return f'$10^{int(logv)}$'

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(x, y, alpha=0.6, s=40, label='Data')

x_pred_log = np.linspace(logx.min(), logx.max(), 500)
y_pred_log = best_model.predict(x_pred_log)
x_pred = 10 ** x_pred_log
y_pred = 10 ** y_pred_log
ax.plot(x_pred, y_pred, 'r-', linewidth=2, label=f'PWLD regression ({best_n_seg} seg.)')

ax.plot(x_env, y_env, 'b-', linewidth=2, label='Upper envelope (max)')

ax.set_xscale('log')
ax.set_yscale('log')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
log_xlim = np.log10(xlim)

for bp in breaks_gflops:
    ax.axvline(x=bp, linestyle='--', color='gray', alpha=0.8, linewidth=1.5)
    bp_log = np.log10(bp)
    offset = 0.03
    x_text_log = bp_log + offset
    if x_text_log > log_xlim[1]:
        x_text_log = bp_log - offset
    x_text = 10 ** x_text_log
    y_text = ylim[1] * 0.95
    ax.text(x_text, y_text, f'{bp:.0f}', ha='left' if x_text > bp else 'right', va='top',
            fontsize=9, color='gray', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

ax.xaxis.set_major_formatter(FuncFormatter(format_power))
ax.yaxis.set_major_formatter(FuncFormatter(format_power))
ax.set_xlabel(f'{x_param} (log scale)', fontsize=12)
ax.set_ylabel(f'{target} (log scale)', fontsize=12)
ax.set_title('Performance vs GFLOPS\nUpper envelope (max) and PWLD regression', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()

# Save plot
plot_file = os.path.join(output_dir, 'regression_plot.png')
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to: {os.path.basename(plot_file)}")

print(f"\nAnalysis finished. All results are in folder: pw/")

# Restore stdout and close log
sys.stdout = original_stdout
log_handle.close()
print(f"Full console log saved to: {os.path.basename(log_file)}")