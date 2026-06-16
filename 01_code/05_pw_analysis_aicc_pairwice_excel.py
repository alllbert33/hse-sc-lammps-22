"""
Piecewise Linear Regression with Positive Lasso, Positive ElasticNet and NNLS
===========================================================================
- Finds optimal breakpoints in log‑log space (AICc)
- For each segment builds multivariate power‑law models with NON‑NEGATIVE coefficients
- Outliers (|residual| > 3σ in log space) are removed before fitting
- Generates a simple prediction plot (all points same colour) for each model
"""

import sys
import os
import pandas as pd
import numpy as np
import pwlf
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import Lasso, ElasticNet, LassoCV, ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ===================== 0. SETUP & LOGGING =====================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(script_dir, '..', '03_data'))
output_dir = os.path.join(data_dir, 'pw')
os.makedirs(output_dir, exist_ok=True)

# Logging
log_file = os.path.join(output_dir, 'analysis_log.txt')
class Tee:
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

# Input data
file_name = 'openbenchmarking - export.csv'
file_path = os.path.join(data_dir, file_name)
target = 'Natoms*ns/day'
x_param = 'GFLOPS'
candidate_features = ['GFLOPS', 'Mem Bandwidth. GB/s', 'Cache L2. KB', 'Cache L3. KB']
min_points_for_multivariate = 5

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

# ===================== 2. UNIVARIATE PWLD (AICc) =====================
def find_optimal_segments_aicc(logx, logy, max_segments=5, min_break_distance=0.15):
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

# Save breakpoints
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

# ===================== 3. UNIVARIATE POWER FORMULAS & METRICS =====================
def get_intercept_for_segment(segment_idx, slopes, breaks_gflops, logx_all, logy_all):
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

y_pred_log = best_model.predict(logx_sorted)
y_pred = 10 ** y_pred_log
y_actual = 10 ** logy_sorted

r2 = r2_score(logy_sorted, y_pred_log)
n = len(logy_sorted)
p = best_n_seg + max(0, best_n_seg - 1)
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

metrics_df = pd.DataFrame({
    'metric': ['R2_log', 'R2_adj', 'RMSE_log', 'RMSE_original', 'MAPE_percent'],
    'value': [r2, r2_adj, rmse_log, rmse_orig, mape]
})
metrics_csv = os.path.join(output_dir, 'univariate_metrics.csv')
metrics_df.to_csv(metrics_csv, index=False)

# Univariate formulas for Excel
formulas = []
if best_n_seg == 1:
    intercept = get_intercept_for_segment(0, slopes, [], logx_sorted, logy_sorted)
    line1 = f"LOG10({target}) = {slopes[0]:.4f} * LOG10(GFLOPS) + {intercept:.4f}"
    line2 = f"{target} = 10^{intercept:.4f} * (GFLOPS)^{slopes[0]:.4f}"
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

formulas_txt = os.path.join(output_dir, 'pwld_formulas.txt')
with open(formulas_txt, 'w', encoding='utf-8') as f:
    f.write("\n".join(formulas))
print(f"Univariate formulas saved to: {os.path.basename(formulas_txt)}")

# ===================== 4. MULTIVARIATE REGRESSION (NNLS, POSITIVE LASSO, POSITIVE ELASTICNET) =====================
multi_features = [f for f in candidate_features if f in df.columns]
print("\nFeatures found in data:", multi_features)

# ---------- Outlier removal helper ----------
def filter_outliers(df, features, target_col, sigma=3):
    """Remove rows where absolute residual in log space > sigma * std."""
    X_log = np.log10(df[features].values)
    y_log = np.log10(df[target_col].values)
    # Use standard linear regression to detect outliers (no positivity constraints)
    lr = LinearRegression()
    lr.fit(X_log, y_log)
    residuals = y_log - lr.predict(X_log)
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    mask = np.abs(residuals - mean_res) <= sigma * std_res
    return df[mask].copy()

# ---------- Helper: NNLS (non‑negative) ----------
def build_nnls(df_sub, features, target_col):
    df_clean = df_sub[features + [target_col]].dropna()
    for f in features:
        df_clean = df_clean[df_clean[f] > 0]
    df_clean = df_clean[df_clean[target_col] > 0]
    if len(df_clean) < min_points_for_multivariate:
        return None
    # Remove outliers
    df_clean = filter_outliers(df_clean, features, target_col)
    if len(df_clean) < min_points_for_multivariate:
        print(f"    After outlier removal only {len(df_clean)} points left, skipping.")
        return None
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
    n_pts = len(y_log)
    p = len(coeff)
    if n_pts - p - 1 > 0:
        r2_adj = 1 - (1 - r2) * (n_pts - 1) / (n_pts - p - 1)
    else:
        r2_adj = np.nan
    const = 10 ** intercept
    formula = f"{target} = {const:.4e}"
    for i, f in enumerate(features):
        formula += f" * ({f})^{coeff[i]:.4f}"
    return {
        'model_type': 'NNLS',
        'formula': formula,
        'R2': r2,
        'R2_adj': r2_adj,
        'RMSE_log': np.sqrt(mean_squared_error(y_log, y_pred_log)),
        'MAPE': np.mean(np.abs((10**y_log - 10**y_pred_log) / (10**y_log))) * 100,
        'n_points': len(df_clean),
        'coefficients': coeff.tolist(),
        'intercept': intercept,
        'features': features,
        'X_log': X_log,
        'y_log': y_log,
        'model': None
    }

# ---------- Helper: Lasso with positive=True ----------
def build_lasso_positive(df_sub, features, target_col):
    df_clean = df_sub[features + [target_col]].dropna()
    for f in features:
        df_clean = df_clean[df_clean[f] > 0]
    df_clean = df_clean[df_clean[target_col] > 0]
    if len(df_clean) < min_points_for_multivariate:
        return None
    # Remove outliers
    df_clean = filter_outliers(df_clean, features, target_col)
    if len(df_clean) < min_points_for_multivariate:
        print(f"    After outlier removal only {len(df_clean)} points left, skipping.")
        return None
    X_log = np.log10(df_clean[features].values)
    y_log = np.log10(df_clean[target_col].values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    # Find best alpha with CV (without positivity, because LassoCV does not support positive)
    lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 30), cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X_scaled, y_log)
    best_alpha = lasso_cv.alpha_
    # Refit with positive=True
    lasso = Lasso(alpha=best_alpha, positive=True, max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y_log)
    # Transform coefficients back to original log scale
    intercept_orig = lasso.intercept_ - np.sum(lasso.coef_ * scaler.mean_ / scaler.scale_)
    coeff_orig = lasso.coef_ / scaler.scale_
    y_pred_log = lasso.predict(X_scaled)
    r2 = r2_score(y_log, y_pred_log)
    n_pts = len(y_log)
    p = np.sum(np.abs(coeff_orig) > 1e-6)
    if n_pts - p - 1 > 0:
        r2_adj = 1 - (1 - r2) * (n_pts - 1) / (n_pts - p - 1)
    else:
        r2_adj = np.nan
    rmse_log = np.sqrt(mean_squared_error(y_log, y_pred_log))
    y_true_orig = 10 ** y_log
    y_pred_orig = 10 ** y_pred_log
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
    const = 10 ** intercept_orig
    formula = f"{target} = {const:.4e}"
    for i, f in enumerate(features):
        if abs(coeff_orig[i]) > 1e-6:
            formula += f" * ({f})^{coeff_orig[i]:.4f}"
    return {
        'model_type': 'Lasso (positive)',
        'formula': formula,
        'R2': r2,
        'R2_adj': r2_adj,
        'RMSE_log': rmse_log,
        'MAPE': mape,
        'n_points': len(df_clean),
        'coefficients': coeff_orig.tolist(),
        'intercept': intercept_orig,
        'features': features,
        'alpha': best_alpha,
        'model': lasso,
        'scaler': scaler,
        'X_log': X_log,
        'y_log': y_log
    }

# ---------- Helper: ElasticNet with positive=True ----------
def build_elasticnet_positive(df_sub, features, target_col):
    df_clean = df_sub[features + [target_col]].dropna()
    for f in features:
        df_clean = df_clean[df_clean[f] > 0]
    df_clean = df_clean[df_clean[target_col] > 0]
    if len(df_clean) < min_points_for_multivariate:
        return None
    # Remove outliers
    df_clean = filter_outliers(df_clean, features, target_col)
    if len(df_clean) < min_points_for_multivariate:
        print(f"    After outlier removal only {len(df_clean)} points left, skipping.")
        return None
    X_log = np.log10(df_clean[features].values)
    y_log = np.log10(df_clean[target_col].values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    enet_cv = ElasticNetCV(alphas=np.logspace(-4, 1, 20),
                           l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                           cv=5, random_state=42, max_iter=10000)
    enet_cv.fit(X_scaled, y_log)
    best_alpha = enet_cv.alpha_
    best_l1 = enet_cv.l1_ratio_
    enet = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, positive=True, max_iter=10000, random_state=42)
    enet.fit(X_scaled, y_log)
    intercept_orig = enet.intercept_ - np.sum(enet.coef_ * scaler.mean_ / scaler.scale_)
    coeff_orig = enet.coef_ / scaler.scale_
    y_pred_log = enet.predict(X_scaled)
    r2 = r2_score(y_log, y_pred_log)
    n_pts = len(y_log)
    p = np.sum(np.abs(coeff_orig) > 1e-6)
    if n_pts - p - 1 > 0:
        r2_adj = 1 - (1 - r2) * (n_pts - 1) / (n_pts - p - 1)
    else:
        r2_adj = np.nan
    rmse_log = np.sqrt(mean_squared_error(y_log, y_pred_log))
    y_true_orig = 10 ** y_log
    y_pred_orig = 10 ** y_pred_log
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
    const = 10 ** intercept_orig
    formula = f"{target} = {const:.4e}"
    for i, f in enumerate(features):
        if abs(coeff_orig[i]) > 1e-6:
            formula += f" * ({f})^{coeff_orig[i]:.4f}"
    return {
        'model_type': 'ElasticNet (positive)',
        'formula': formula,
        'R2': r2,
        'R2_adj': r2_adj,
        'RMSE_log': rmse_log,
        'MAPE': mape,
        'n_points': len(df_clean),
        'coefficients': coeff_orig.tolist(),
        'intercept': intercept_orig,
        'features': features,
        'alpha': best_alpha,
        'l1_ratio': best_l1,
        'model': enet,
        'scaler': scaler,
        'X_log': X_log,
        'y_log': y_log
    }

# ---------- Plotting: simple scatter (all points same colour) ----------
def plot_pred_vs_true(model_info, df_seg, segment_name, method_name, save_dir):
    """Создаёт один простой график предсказанных vs реальных значений (все точки одного цвета)."""
    # Get predictions
    if model_info['model_type'] == 'NNLS':
        X_log = model_info['X_log']
        intercept = model_info['intercept']
        coeff = np.array(model_info['coefficients'])
        y_pred_log = intercept + X_log @ coeff
        y_true_log = model_info['y_log']
    else:
        X_log = model_info['X_log']
        scaler = model_info['scaler']
        X_scaled = scaler.transform(X_log)
        y_pred_log = model_info['model'].predict(X_scaled)
        y_true_log = model_info['y_log']
    y_pred = 10 ** y_pred_log
    y_true = 10 ** y_true_log

    # Metrics
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # Create plot
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.scatter(y_pred, y_true, alpha=0.7, edgecolors='w', s=40, color='steelblue')
    # Diagonal y=x
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, '--', color='grey', linewidth=2, alpha=0.8, label='y = x')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Предсказанная производительность', fontsize=14)
    ax.set_ylabel(f'Реальная производительность ({target})', fontsize=14)
    ax.set_title(f'{segment_name}\n{method_name}\nR² = {r2:.3f}, MAPE = {mape:.1f}%', fontsize=12)
    ax.grid(True, which='both', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'pred_vs_true.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"    Plot saved: {plot_path}")

# ---------- Apply to segments ----------
all_results = []
methods = [('NNLS', build_nnls), ('Lasso (positive)', build_lasso_positive), ('ElasticNet (positive)', build_elasticnet_positive)]

if best_n_seg == 1 or len(breaks_gflops) == 0:
    segments = [('all_data', df)]
else:
    df_segmented = df.copy()
    seg_labels = np.digitize(df_segmented[x_param].values, breaks_gflops, right=False)
    df_segmented['segment'] = seg_labels
    segments = []
    segments.append((f"GFLOPS ≤ {breaks_gflops[0]:.0f}", df_segmented[df_segmented['segment'] == 0]))
    for i in range(len(breaks_gflops)-1):
        segments.append((f"{breaks_gflops[i]:.0f} < GFLOPS ≤ {breaks_gflops[i+1]:.0f}", df_segmented[df_segmented['segment'] == i+1]))
    segments.append((f"GFLOPS > {breaks_gflops[-1]:.0f}", df_segmented[df_segmented['segment'] == best_n_seg-1]))

print("\n" + "="*80)
print("MULTIVARIATE REGRESSION COMPARISON (NNLS vs Positive Lasso vs Positive ElasticNet)")
print("Outliers (|residual| > 3σ in log space) are removed before fitting.")
print("="*80)

for seg_name, seg_df in segments:
    if len(seg_df) < min_points_for_multivariate:
        print(f"\nSegment '{seg_name}': insufficient points ({len(seg_df)}), skipping.")
        continue
    print(f"\n--- Segment: {seg_name} (points: {len(seg_df)}) ---")
    for method_name, builder in methods:
        result = builder(seg_df, multi_features, target)
        if result is None:
            print(f"  {method_name}: failed to build model.")
            continue
        result['segment'] = seg_name
        all_results.append(result)
        print(f"  {method_name}: R² = {result['R2']:.4f}, adj.R² = {result['R2_adj']:.4f}, RMSE(log) = {result['RMSE_log']:.4f}, MAPE = {result['MAPE']:.2f}% (after outlier removal, n={result['n_points']})")
        if 'alpha' in result:
            print(f"      Hyperparameters: alpha={result['alpha']:.5f}" + (f", l1_ratio={result['l1_ratio']:.3f}" if 'l1_ratio' in result else ''))
        print(f"      Formula: {result['formula'][:100]}...")
        # Create subfolder for this method inside segment folder
        safe_seg = seg_name.replace(' ', '_').replace('≤', 'le').replace('>', 'gt').replace('<', 'lt').replace('=', 'eq')
        safe_method = method_name.replace(' ', '_').replace('(', '').replace(')', '')
        method_dir = os.path.join(output_dir, safe_seg, safe_method)
        os.makedirs(method_dir, exist_ok=True)
        # Save the plot
        plot_pred_vs_true(result, seg_df, seg_name, method_name, method_dir)
        # Also save the formula and coefficients as text
        info_txt = os.path.join(method_dir, 'model_info.txt')
        with open(info_txt, 'w', encoding='utf-8') as f:
            f.write(f"Segment: {seg_name}\n")
            f.write(f"Method: {method_name}\n")
            f.write(f"Formula: {result['formula']}\n")
            f.write(f"R² = {result['R2']:.4f}\n")
            f.write(f"Adj. R² = {result['R2_adj']:.4f}\n")
            f.write(f"RMSE (log) = {result['RMSE_log']:.4f}\n")
            f.write(f"MAPE = {result['MAPE']:.2f}%\n")
            f.write(f"Number of points (after outlier removal): {result['n_points']}\n")
            if 'alpha' in result:
                f.write(f"alpha = {result['alpha']:.5f}\n")
            if 'l1_ratio' in result:
                f.write(f"l1_ratio = {result['l1_ratio']:.3f}\n")
            f.write("\nCoefficients (original log scale):\n")
            for i, feat in enumerate(result['features']):
                f.write(f"  {feat}: {result['coefficients'][i]:.6f}\n")
            f.write(f"\nIntercept (log10): {result['intercept']:.6f}\n")

# Save all results to CSV
results_df = pd.DataFrame(all_results)
results_csv = os.path.join(output_dir, 'multivariate_comparison.csv')
results_df.to_csv(results_csv, index=False)
print(f"\nAll multivariate results saved to: {os.path.basename(results_csv)}")

# ===================== 5. UPPER ENVELOPE AND MAIN PLOT =====================
def upper_envelope(x, y, n_points=200, window_ratio=0.2):
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
            y_env[i] = np.max(y[mask])
        else:
            idx = np.argmin(np.abs(x - xc))
            y_env[i] = y[idx]
    return x_grid, y_env

print("\nBuilding upper envelope...")
x_env, y_env = upper_envelope(x, y)
envelope_df = pd.DataFrame({'GFLOPS': x_env, target: y_env})
envelope_csv = os.path.join(output_dir, 'upper_envelope.csv')
envelope_df.to_csv(envelope_csv, index=False)
print(f"Upper envelope saved to: {os.path.basename(envelope_csv)}")

def format_power(value, pos):
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
ax.set_title('Performance vs GFLOPS\nUpper envelope and PWLD regression', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plot_file = os.path.join(output_dir, 'regression_plot.png')
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to: {os.path.basename(plot_file)}")

print(f"\nAnalysis finished. All results are in folder: pw/")
sys.stdout = original_stdout
log_handle.close()
print(f"Full console log saved to: {os.path.basename(log_file)}")