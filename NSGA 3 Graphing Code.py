import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import os

# ======================================================
# NSGA-III GRAPHING SCRIPT — VERSION 1.0
# Mirrors the structure of MOPSO_Graphing_Code.py
# Run ONLY after NSGA3_main.py has already produced
# the four CSV files below:
#   • nsga3_pareto_results.csv
#   • nsga3_validation_data.csv
#   • nsga3_convergence_data.csv
#   • nsga3_sensitivity_data.csv
# ======================================================

# ── THEME ──────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size':       12,
    'axes.labelsize':  14,
    'axes.titlesize':  16,
    'figure.dpi':      300,
    'savefig.dpi':     300,
    'font.family':     'serif'
})

save_folder = 'publication_plots_nsga3'
os.makedirs(save_folder, exist_ok=True)

obj_cols = ['BSFC', 'BTE', 'NOx', 'CO', 'POWER', 'TORQUE', 'CO2']

# ── LOAD DATA ──────────────────────────────────────────
try:
    pareto_df = pd.read_csv('nsga3_pareto_results.csv')
    valid_df  = pd.read_csv('nsga3_validation_data.csv')
    conv_df   = pd.read_csv('nsga3_convergence_data.csv')
    sens_df   = pd.read_csv('nsga3_sensitivity_data.csv')
    print("All CSV files loaded successfully!\n")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Run the NSGA-III optimisation script first to generate the CSV files.")
    raise SystemExit


# ══════════════════════════════════════════════════════
# 1. PARITY PLOTS (Experimental vs GPR Predicted)
#    Shows how accurate the surrogate model is.
# ══════════════════════════════════════════════════════
def plot_parity_grid(df):
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes = axes.flatten()

    for i, col in enumerate(obj_cols):
        exp  = df[f'{col}_Exp']
        pred = df[f'{col}_Pred']

        # Perfect-fit reference line
        lo = min(exp.min(), pred.min())
        hi = max(exp.max(), pred.max())
        axes[i].plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect fit')

        sns.regplot(x=exp, y=pred, ax=axes[i],
                    scatter_kws={'alpha': 0.6, 'color': 'steelblue', 's': 40},
                    line_kws={'color': 'darkorange', 'lw': 1.5})

        # R² annotation
        ss_res = np.sum((exp - pred) ** 2)
        ss_tot = np.sum((exp - exp.mean()) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        axes[i].annotate(f'R² = {r2:.4f}', xy=(0.05, 0.92),
                         xycoords='axes fraction', fontsize=11,
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

        axes[i].set_title(f'Parity: {col}', fontsize=13)
        axes[i].set_xlabel('Experimental')
        axes[i].set_ylabel('GPR Predicted')

    # Hide the spare (8th) subplot
    axes[-1].set_visible(False)
    fig.suptitle('GPR Model Accuracy — Experimental vs. Predicted (NSGA-III)', fontsize=16, y=1.01)
    plt.tight_layout()
    path = f'{save_folder}/parity_grid.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════
# 2. RESIDUAL ERROR DISTRIBUTIONS
#    One histogram per objective — proves no model bias.
# ══════════════════════════════════════════════════════
def plot_residuals(df):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, col in enumerate(obj_cols):
        residuals = df[f'{col}_Exp'] - df[f'{col}_Pred']
        sns.histplot(residuals, kde=True, ax=axes[i],
                     color='mediumpurple', edgecolor='white', bins=12)
        axes[i].axvline(0, color='red', lw=1.5, ls='--')
        axes[i].set_title(f'Residuals: {col}')
        axes[i].set_xlabel('Prediction Error')
        axes[i].set_ylabel('Count')

    axes[-1].set_visible(False)
    fig.suptitle('Residual Error Distribution — NSGA-III GPR Models', fontsize=16, y=1.01)
    plt.tight_layout()
    path = f'{save_folder}/residuals_distribution.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════
# 3. CONVERGENCE + HYPERVOLUME (two-panel)
#    Tracks optimisation health over generations.
# ══════════════════════════════════════════════════════
def plot_convergence(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(df['Generation'], df['Mean_Desirability'],
             color='darkorange', lw=2.5, marker='o', markersize=3)
    ax1.set_title('NSGA-III Convergence — Mean Desirability')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mean Desirability of Pareto Front')
    ax1.grid(alpha=0.4)

    ax2.plot(df['Generation'], df['Pareto_Size'],
             color='steelblue', lw=2.5, marker='s', markersize=3)
    ax2.set_title('Pareto Front Growth Over Generations')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Pareto Front Size')
    ax2.grid(alpha=0.4)

    plt.tight_layout()
    path = f'{save_folder}/convergence_and_pareto_size.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════
# 4. SENSITIVITY ANALYSIS REPLOT
#    Reads pre-computed data; no model needed at runtime.
# ══════════════════════════════════════════════════════
def plot_sensitivity(df):
    plt.figure(figsize=(10, 6))
    colors  = {'BSFC': 'red', 'NOx': 'blue', 'BTE': 'green'}
    targets = df['Target'].unique()

    for target in targets:
        sub = df[df['Target'] == target].sort_values('Perturbation_%')
        plt.plot(sub['Perturbation_%'], sub['Change_%'],
                 label=f'Sensitivity of {target}',
                 color=colors.get(target, 'gray'),
                 marker='o', markersize=4, lw=2)

    plt.axhline(0, color='black', lw=1, ls='--')
    plt.axvline(0, color='black', lw=1, ls='--')
    plt.xlabel('% Deviation from Optimal RPM')
    plt.ylabel('% Change in Objective Value')
    plt.title('NSGA-III: System Sensitivity Analysis at Optimal Point')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = f'{save_folder}/sensitivity_analysis.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════
# 5. RADAR CHART (Best vs Pareto Mean)
#    Normalized performance fingerprint of best solution.
# ══════════════════════════════════════════════════════
def plot_radar_comparison(df):
    labels   = ['BSFC', 'BTE', 'POWER', 'TORQUE', 'NOx', 'CO2']
    df_norm  = (df[labels] - df[labels].min()) / (df[labels].max() - df[labels].min())
    df_norm  = df_norm.fillna(0.5)

    best_idx  = df['Desirability'].idxmax()
    best_vals = df_norm.loc[best_idx].values
    mean_vals = df_norm.mean().values

    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Close the polygon
    best_vals = np.concatenate((best_vals, [best_vals[0]]))
    mean_vals = np.concatenate((mean_vals, [mean_vals[0]]))
    angles    = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.fill(angles, best_vals, color='blue',   alpha=0.20)
    ax.plot(angles, best_vals, color='blue',   lw=2,  label='Best Solution')
    ax.fill(angles, mean_vals, color='orange', alpha=0.15)
    ax.plot(angles, mean_vals, color='orange', lw=2,  label='Pareto Mean', ls='--')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels([])
    ax.set_title('Performance Fingerprint (Normalized)\nNSGA-III Best vs Pareto Mean',
                 pad=20, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    path = f'{save_folder}/radar_comparison.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════
# 6. PARETO SCATTER MATRIX (BSFC / BTE / NOx / CO2)
#    Four 2-D trade-off slices, coloured by Desirability.
# ══════════════════════════════════════════════════════
def plot_pareto_scatter_matrix(df):
    pairs = [
        ('BSFC', 'BTE',   'BSFC vs BTE'),
        ('NOx',  'CO2',   'NOx vs CO2'),
        ('BSFC', 'NOx',   'BSFC vs NOx'),
        ('BTE',  'POWER', 'BTE vs POWER'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (xcol, ycol, title) in zip(axes, pairs):
        sc = ax.scatter(df[xcol], df[ycol],
                        c=df['Desirability'], cmap='plasma',
                        edgecolors='k', lw=0.3, s=50, alpha=0.85)
        plt.colorbar(sc, ax=ax, label='Desirability')
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(title)

    fig.suptitle('NSGA-III Pareto Front — Trade-off Slices', fontsize=16)
    plt.tight_layout()
    path = f'{save_folder}/pareto_scatter_matrix.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════
# 7. CORRELATION HEATMAP
#    Objective + input correlations across Pareto front.
# ══════════════════════════════════════════════════════
def plot_correlation(df):
    cols = ['RPM', 'BLEND'] + obj_cols + ['Desirability']
    plt.figure(figsize=(11, 9))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f',
                linewidths=0.5, annot_kws={'size': 9})
    plt.title('Correlation Matrix — NSGA-III Pareto Front\n(Inputs + Objectives)',
              fontsize=14)
    plt.tight_layout()
    path = f'{save_folder}/correlation_heatmap.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════
# 8. PARALLEL COORDINATES PLOT
#    Shows every Pareto solution across all objectives
#    simultaneously — great for spotting trade-offs.
# ══════════════════════════════════════════════════════
def plot_parallel_coordinates(df):
    # Normalize for uniform axis scaling
    df_norm = df[obj_cols + ['Desirability']].copy()
    for col in df_norm.columns:
        lo, hi = df_norm[col].min(), df_norm[col].max()
        df_norm[col] = (df_norm[col] - lo) / (hi - lo + 1e-12)

    # Bin desirability for colour legend
    df_norm['D_rank'] = pd.cut(df['Desirability'], bins=5,
                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    fig, ax = plt.subplots(figsize=(16, 7))
    palette = {'Very Low': '#d73027', 'Low': '#fc8d59',
               'Medium':   '#fee090', 'High': '#91bfdb', 'Very High': '#4575b4'}

    cols_to_plot = obj_cols + ['Desirability']
    x_positions  = list(range(len(cols_to_plot)))

    for _, row in df_norm.iterrows():
        color = palette.get(str(row['D_rank']), 'grey')
        ax.plot(x_positions, [row[c] for c in cols_to_plot],
                color=color, alpha=0.4, lw=1)

    patches = [mpatches.Patch(color=v, label=k) for k, v in palette.items()]
    ax.legend(handles=patches, title='Desirability', loc='upper right', fontsize=10)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(cols_to_plot, fontsize=11, rotation=15)
    ax.set_ylabel('Normalized Value')
    ax.set_title('Parallel Coordinates — NSGA-III Pareto Front\n(coloured by Desirability tier)',
                 fontsize=14)
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    path = f'{save_folder}/parallel_coordinates.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════
# 9. INTERACTIVE 3-D PARETO FRONT (Plotly)
#    Saved as standalone HTML — rotate in any browser.
# ══════════════════════════════════════════════════════
def plot_3d_pareto(df):
    fig = px.scatter_3d(
        df,
        x='BSFC', y='NOx', z='BTE',
        color='Desirability',
        color_continuous_scale='Plasma',
        size='POWER',
        hover_data=['RPM', 'BLEND', 'CO2', 'TORQUE'],
        title='NSGA-III — Interactive 3D Pareto Front (BSFC / NOx / BTE)'
    )
    fig.update_layout(scene=dict(
        xaxis_title='BSFC',
        yaxis_title='NOx',
        zaxis_title='BTE'
    ))
    path = f'{save_folder}/pareto_3d_interactive.html'
    fig.write_html(path)
    print(f"Saved: {path}  (open in any browser to rotate)")


# ══════════════════════════════════════════════════════
# 10. BEST SOLUTION SUMMARY TABLE
#     Prints and saves a clean summary of the top-5
#     Pareto solutions ranked by desirability.
# ══════════════════════════════════════════════════════
def print_best_solutions(df, top_n=5):
    top = df.nlargest(top_n, 'Desirability')[['RPM', 'BLEND'] + obj_cols + ['Desirability']]
    print("\n" + "=" * 70)
    print(f"TOP {top_n} NSGA-III SOLUTIONS (ranked by Desirability)")
    print("=" * 70)
    print(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 70)

    top.to_csv(f'{save_folder}/top_{top_n}_solutions.csv', index=False)
    print(f"Saved: {save_folder}/top_{top_n}_solutions.csv\n")


# ══════════════════════════════════════════════════════
# EXECUTE ALL PLOTS
# ══════════════════════════════════════════════════════
print_best_solutions(pareto_df)
plot_parity_grid(valid_df)
plot_residuals(valid_df)
plot_convergence(conv_df)
plot_sensitivity(sens_df)
plot_radar_comparison(pareto_df)
plot_pareto_scatter_matrix(pareto_df)
plot_correlation(pareto_df)
plot_parallel_coordinates(pareto_df)
plot_3d_pareto(pareto_df)

print(f"\nAll plots saved in '{save_folder}/'")
plt.show()