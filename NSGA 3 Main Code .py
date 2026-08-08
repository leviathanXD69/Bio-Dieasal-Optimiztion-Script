import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler

# ======================================================
# GOD TIER ENGINE NSGA-III OPTIMIZER - VERSION 1.0
# (Das-Dennis Reference Points + Niching Selection)
# Diesel + Biodiesel Blend Optimization
# ======================================================

np.random.seed(42)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 140

# ---------------- USER SETTINGS ----------------
filepath = r"C:\Users\User\Desktop\The Engineering Knowledge\semester 5\semester design project usman don\organized data.xlsx"

N_OBJ       = 7        # Number of objectives (must match obj_cols)
N_DIVISIONS = 3        # Das-Dennis divisions: C(N_OBJ+p-1, p) reference points
               #   p=3, M=7 → C(9,3) = 84 reference points → set POP_SIZE ≥ 84
POP_SIZE    = 84       # Recommended ≈ number of reference points
GENERATIONS = 60
SBX_ETA     = 15
PM_ETA      = 20
CROSSOVER_P = 0.9
MUTATION_P  = 0.1
save_folder = 'results_figures_nsga3'
os.makedirs(save_folder, exist_ok=True)
# -----------------------------------------------

# =============== LOAD & CLEAN DATA =============
df = pd.read_excel(filepath)
df = df.dropna().reset_index(drop=True)
df = df.rename(columns={'SPEED (RPM)': 'RPM', 'BLEND (%)': 'BLEND'})

obj_cols = ['BSFC', 'BTE', 'NOx', 'CO', 'POWER', 'TORQUE', 'CO2']

# NSGA-III minimizes all objectives internally;
# objectives that should be maximised are negated in normalize().
# Desirability is computed separately in original units.
TO_MAXIMIZE = {1, 4, 5}   # indices in obj_cols: BTE, POWER, TORQUE

points = df[['RPM', 'BLEND']].values

scaler_x = StandardScaler()
points_scaled = scaler_x.fit_transform(points)

def bounds(col):
    return df[col].min(), df[col].max()

rpm_bounds   = bounds('RPM')
blend_bounds = bounds('BLEND')
obj_bounds   = {c: bounds(c) for c in obj_cols}

# ======================================================
# TRAIN STABILIZED GPR MODELS
# ======================================================
print("\nTraining Stabilized GPR Models... Please wait.")
models, scalers_y = {}, {}

kernel = (C(1.0, (1e-5, 1e5))
          * RBF([1.0, 1.0], (1e-3, 1e3))
          + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-7, 1e1)))

for c in obj_cols:
    y = df[c].values.reshape(-1, 1)
    sy = StandardScaler()
    y_scaled = sy.fit_transform(y)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=1e-5)
    gpr.fit(points_scaled, y_scaled)
    models[c] = gpr
    scalers_y[c] = sy

# ======================================================
# GPR ACCURACY REPORT
# ======================================================
print("\n" + "="*60)
print("STABILIZED GPR ACCURACY REPORT")
print("="*60)
accuracy_results = []
for c in obj_cols:
    y_true = df[c].values
    y_pred_scaled = models[c].predict(points_scaled).reshape(-1, 1)
    y_pred = scalers_y[c].inverse_transform(y_pred_scaled).flatten()
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy_results.append({'Objective': c, 'R2': r2, 'MAE': mae, 'MAPE%': mape})
print(pd.DataFrame(accuracy_results).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ===============================================
#  GPR EVALUATION
# ===============================================
def evaluate_raw(pos):
    """Return raw (physical-unit) objective values for a candidate [RPM, BLEND]."""
    pos_s = scaler_x.transform(np.array(pos).reshape(1, -1))
    vals = [
        scalers_y[c].inverse_transform(
            models[c].predict(pos_s).reshape(-1, 1)
        ).item()
        for c in obj_cols
    ]
    return np.array(vals, dtype=float)




    # ... Line 110: print(pd.DataFrame(accuracy_results).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ======================================================
# INSERT NEW CODE HERE (Starting at Line 114)
# ======================================================
print("\nGenerating Validation Data for separate Visualizer...")
valid_data = pd.DataFrame()

for c in obj_cols:
    # Get actual values from original dataframe
    y_true = df[c].values
    
    # Generate predictions using the trained model
    y_pred_scaled = models[c].predict(points_scaled).reshape(-1, 1)
    y_pred = scalers_y[c].inverse_transform(y_pred_scaled).flatten()
    
    # Store in dataframe with specific naming for the visualizer script
    valid_data[f'{c}_Exp'] = y_true
    valid_data[f'{c}_Pred'] = y_pred

# Save to CSV so the second script can access it
valid_data.to_csv('nsga3_validation_data.csv', index=False)
print("Validation CSV saved successfully for parity plots.")

# ======================================================
# Line 132: # GPR EVALUATION (Continue with existing code)
# ======================================================









def normalize_for_nsga3(vals):
    """
    Convert raw objectives to a minimisation vector.
    Maximised objectives are negated so ALL objectives are minimised.
    """
    v = vals.copy()
    for i in TO_MAXIMIZE:
        v[i] = -v[i]
    return v

# ===============================================
#  DESIRABILITY (unchanged from NSGA-II version)
# ===============================================
def desirability_min(v, lo, hi):
    if v <= lo: return 1.0
    if v >= hi: return 0.0
    return (hi - v) / (hi - lo)

def desirability_max(v, lo, hi):
    if v >= hi: return 1.0
    if v <= lo: return 0.0
    return (v - lo) / (hi - lo)

weights = np.array([3, 3, 1, 1, 3, 2, 1])

def total_desirability(row):
    epsilon = 0.005
    ds = [
        max(epsilon, desirability_min(row['BSFC'],  *obj_bounds['BSFC'])),
        max(epsilon, desirability_max(row['BTE'],   *obj_bounds['BTE'])),
        max(epsilon, desirability_min(row['NOx'],   *obj_bounds['NOx'])),
        max(epsilon, desirability_min(row['CO'],    *obj_bounds['CO'])),
        max(epsilon, desirability_max(row['POWER'], *obj_bounds['POWER'])),
        max(epsilon, desirability_max(row['TORQUE'],*obj_bounds['TORQUE'])),
        max(epsilon, desirability_min(row['CO2'],   *obj_bounds['CO2']))
    ]
    prod = np.prod(np.power(ds, weights))
    return prod ** (1.0 / weights.sum())

# ======================================================
#  NSGA-III CORE — Reference Points (Das-Dennis method)
# ======================================================
def _recursive_ref(n_obj, n_div, current, remaining, ref_pts):
    """Recursively enumerate Das-Dennis reference point coordinates."""
    if n_obj == 1:
        ref_pts.append(current + [remaining / n_div])
        return
    for i in range(remaining + 1):
        _recursive_ref(n_obj - 1, n_div, current + [i / n_div],
                       remaining - i, ref_pts)

def generate_reference_points(n_obj, n_div):
    """
    Generate Das-Dennis structured reference points on the unit simplex.
    Number of points = C(n_obj + n_div - 1, n_div).
    """
    ref_pts = []
    _recursive_ref(n_obj, n_div, [], n_div, ref_pts)
    return np.array(ref_pts)

# Pre-compute reference points once
REF_POINTS = generate_reference_points(N_OBJ, N_DIVISIONS)
print(f"\nNSGA-III Reference Points generated: {len(REF_POINTS)}")
print(f"Population size: {POP_SIZE}")

# ======================================================
#  NSGA-III CORE — Non-dominated Sort (same as NSGA-II)
# ======================================================
def fast_non_dominated_sort(F):
    n = len(F)
    S, n_dom, fronts = [[] for _ in range(n)], np.zeros(n, dtype=int), [[]]
    for i in range(n):
        for j in range(i + 1, n):
            Fi, Fj = F[i], F[j]
            if np.all(Fi <= Fj) and np.any(Fi < Fj):
                S[i].append(j); n_dom[j] += 1
            elif np.all(Fj <= Fi) and np.any(Fj < Fi):
                S[j].append(i); n_dom[i] += 1
        if n_dom[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    nxt.append(j)
        k += 1; fronts.append(nxt)
    return [f for f in fronts if f]

# ======================================================
#  NSGA-III CORE — Normalisation & Association
# ======================================================
def get_ideal_nadir(F):
    """Compute ideal (min per objective) and nadir (max per objective)."""
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
    return ideal, nadir

def normalize_population(F, ideal, nadir):
    """Translate & scale so each objective lies in [0, ∞), with front on unit simplex."""
    span = nadir - ideal
    span[span < 1e-12] = 1e-12          # avoid division by zero
    return (F - ideal) / span

def associate_to_reference(F_norm, ref_pts):
    """
    For each solution find the closest reference line (from origin).
    Returns:
        assoc  : (n_solutions,) index into ref_pts
        dist   : (n_solutions,) perpendicular distance to assigned ref line
    """
    # Normalise reference vectors to unit length
    ref_norm = ref_pts / (np.linalg.norm(ref_pts, axis=1, keepdims=True) + 1e-12)

    assoc = np.empty(len(F_norm), dtype=int)
    dist  = np.empty(len(F_norm), dtype=float)

    for i, f in enumerate(F_norm):
        # Scalar projection onto each reference direction
        scalar_proj = ref_norm @ f                        # (n_ref,)
        scalar_proj = np.maximum(scalar_proj, 0.0)        # keep positive side
        # Closest point on each reference line
        proj_pts = scalar_proj[:, None] * ref_norm         # (n_ref, M)
        # Perpendicular distances
        dists = np.linalg.norm(f - proj_pts, axis=1)       # (n_ref,)
        best = np.argmin(dists)
        assoc[i] = best
        dist[i]  = dists[best]

    return assoc, dist

# ======================================================
#  NSGA-III CORE — Niching Selection
# ======================================================
def niching_selection(candidates, needed, niche_count, assoc, dist):
    """
    Select `needed` solutions from `candidates` using NSGA-III niche preservation.
    candidates : list of global indices in the last critical front
    niche_count: dict {ref_idx: count already chosen from prev fronts}
    assoc      : global assoc array (n_combined,)
    dist       : global dist  array (n_combined,)
    """
    selected = []
    niche_count = niche_count.copy()    # local copy

    while len(selected) < needed:
        # Reference points with minimum niche count
        j_min = min(niche_count.values())
        # All ref lines with that minimum count that still have candidates
        ref_with_min = [r for r, c in niche_count.items()
                        if c == j_min
                        and any(assoc[s] == r for s in candidates)]
        if not ref_with_min:
            # Fall back: pick the ref line with fewest choices globally
            ref_with_min = list(niche_count.keys())

        # Pick a random reference line from the minimally-occupied set
        chosen_ref = ref_with_min[np.random.randint(len(ref_with_min))]

        # Candidates assigned to this reference line
        cands_in_ref = [s for s in candidates if assoc[s] == chosen_ref]
        if not cands_in_ref:
            niche_count[chosen_ref] += 1   # skip exhausted niches
            continue

        if niche_count[chosen_ref] == 0:
            # Pick the one with the smallest perpendicular distance
            pick = min(cands_in_ref, key=lambda s: dist[s])
        else:
            # Pick randomly among the candidates in this niche
            pick = cands_in_ref[np.random.randint(len(cands_in_ref))]

        selected.append(pick)
        candidates.remove(pick)
        niche_count[chosen_ref] = niche_count.get(chosen_ref, 0) + 1

    return selected

# ======================================================
#  NSGA-III CORE — Genetic Operators (same as NSGA-II)
# ======================================================
def tournament_select_nsga3(pop_size, ranks, assoc, dist, niche_count):
    """Binary tournament: rank first, then niche count, then distance."""
    c = np.random.choice(pop_size, 2, replace=False)
    a, b = c[0], c[1]
    if ranks[a] < ranks[b]:
        return a
    if ranks[b] < ranks[a]:
        return b
    # Same rank — prefer less-crowded niche
    na, nb = niche_count.get(assoc[a], 0), niche_count.get(assoc[b], 0)
    if na < nb: return a
    if nb < na: return b
    # Same niche count — prefer smaller perpendicular distance
    return a if dist[a] <= dist[b] else b

def sbx_crossover(p1, p2, lb, ub):
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if np.random.rand() < 0.5 and abs(p1[i] - p2[i]) > 1e-9:
            y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
            u = np.random.rand()
            if (y1 - lb[i]) < (ub[i] - y2):
                beta = 1.0 + (2.0 * (y1 - lb[i]) / (y2 - y1))
            else:
                beta = 1.0 + (2.0 * (ub[i] - y2) / (y2 - y1))
            alpha = 2.0 - beta ** (-(SBX_ETA + 1))
            bq = ((u * alpha) ** (1 / (SBX_ETA + 1)) if u <= 1 / alpha
                  else (1 / (2.0 - u * alpha)) ** (1 / (SBX_ETA + 1)))
            c1[i] = 0.5 * ((y1 + y2) - bq * (y2 - y1))
            c2[i] = 0.5 * ((y1 + y2) + bq * (y2 - y1))
    return np.clip(c1, lb, ub), np.clip(c2, lb, ub)

def polynomial_mutation(x, lb, ub):
    child = x.copy()
    for i in range(len(x)):
        if np.random.rand() < MUTATION_P:
            u = np.random.rand()
            d1 = (x[i] - lb[i]) / (ub[i] - lb[i] + 1e-12)
            d2 = (ub[i] - x[i]) / (ub[i] - lb[i] + 1e-12)
            dq = ((2*u + (1 - 2*u)*(1 - d1)**(PM_ETA+1))**(1/(PM_ETA+1)) - 1
                  if u < 0.5 else
                  1 - (2*(1-u) + 2*(u-0.5)*(1-d2)**(PM_ETA+1))**(1/(PM_ETA+1)))
            child[i] = np.clip(x[i] + dq * (ub[i] - lb[i]), lb[i], ub[i])
    return child

# ======================================================
#  NSGA-III MAIN LOOP
# ======================================================
lb = np.array([rpm_bounds[0],   blend_bounds[0]])
ub = np.array([rpm_bounds[1],   blend_bounds[1]])

# Initialise population
pop   = np.column_stack([np.random.uniform(lb[0], ub[0], POP_SIZE),
                          np.random.uniform(lb[1], ub[1], POP_SIZE)])

# F_pop stores the MINIMISATION objectives (maximised ones are negated)
F_pop = np.array([normalize_for_nsga3(evaluate_raw(p)) for p in pop])

conv, arch_hist = [], []
print("\nStarting NSGA-III GPR Optimisation...")

for gen in range(GENERATIONS):
    # ---- Non-dominated sort ----
    fronts = fast_non_dominated_sort(F_pop)
    ranks  = np.zeros(POP_SIZE, dtype=int)
    for r, f in enumerate(fronts):
        for idx in f:
            ranks[idx] = r

    # ---- Normalise & associate (current population) ----
    ideal, nadir = get_ideal_nadir(F_pop)
    F_norm       = normalize_population(F_pop, ideal, nadir)
    assoc, dist  = associate_to_reference(F_norm, REF_POINTS)

    # Build niche count over the already-selected fronts (all but potentially the last)
    niche_count = {i: 0 for i in range(len(REF_POINTS))}
    for idx in range(POP_SIZE):
        niche_count[assoc[idx]] = niche_count.get(assoc[idx], 0) + 1

    # ---- Tournament selection & offspring generation ----
    off, F_off = np.empty_like(pop), np.empty_like(F_pop)
    for k in range(0, POP_SIZE, 2):
        p1 = tournament_select_nsga3(POP_SIZE, ranks, assoc, dist, niche_count)
        p2 = tournament_select_nsga3(POP_SIZE, ranks, assoc, dist, niche_count)
        if np.random.rand() < CROSSOVER_P:
            c1, c2 = sbx_crossover(pop[p1], pop[p2], lb, ub)
        else:
            c1, c2 = pop[p1].copy(), pop[p2].copy()
        off[k],   F_off[k]   = polynomial_mutation(c1, lb, ub), None
        off[k+1], F_off[k+1] = polynomial_mutation(c2, lb, ub), None

    # Evaluate offspring
    for k in range(POP_SIZE):
        F_off[k] = normalize_for_nsga3(evaluate_raw(off[k]))

    # ---- Merge parent + offspring ----
    comb   = np.vstack([pop, off])
    F_comb = np.vstack([F_pop, F_off])
    n_comb = len(comb)

    # ---- Non-dominated sort on combined pool ----
    f_c    = fast_non_dominated_sort(F_comb)

    # ---- Reference-point–based selection (NSGA-III Procedure) ----
    ideal_c, nadir_c = get_ideal_nadir(F_comb)
    F_norm_c         = normalize_population(F_comb, ideal_c, nadir_c)
    assoc_c, dist_c  = associate_to_reference(F_norm_c, REF_POINTS)

    new_idx     = []
    niche_sel   = {i: 0 for i in range(len(REF_POINTS))}

    for front in f_c:
        if len(new_idx) + len(front) <= POP_SIZE:
            # Entire front fits — add all, update niche counts
            new_idx.extend(front)
            for idx in front:
                niche_sel[assoc_c[idx]] = niche_sel.get(assoc_c[idx], 0) + 1
        else:
            # Last front: fill with niching selection
            needed    = POP_SIZE - len(new_idx)
            remaining = list(front)
            chosen    = niching_selection(remaining, needed,
                                          niche_sel, assoc_c, dist_c)
            new_idx.extend(chosen)
            break

    pop   = comb[new_idx]
    F_pop = F_comb[new_idx]

    # ---- Convergence tracking ----
    pf    = fast_non_dominated_sort(F_pop)[0]
    arch_hist.append(len(pf))
    p_des = [
        total_desirability({c: evaluate_raw(pop[idx])[i]
                            for i, c in enumerate(obj_cols)})
        for idx in pf
    ]
    conv.append(np.mean(p_des))
    if (gen + 1) % 10 == 0:
        print(f"  Gen {gen+1:>2} | Pareto: {len(pf):>3} | Mean Des: {conv[-1]:.4f}")



conv_df = pd.DataFrame({
    'Generation':        range(1, GENERATIONS + 1),
    'Mean_Desirability': conv,
    'Pareto_Size':       arch_hist
})
conv_df.to_csv('nsga3_convergence_data.csv', index=False)
print("Convergence CSV saved.")





# ======================================================
#  EXPORT RESULTS
# ======================================================
final_pf = fast_non_dominated_sort(F_pop)[0]
final_results = []
for idx in final_pf:
    raw = evaluate_raw(pop[idx])
    row = {'RPM': pop[idx][0], 'BLEND': pop[idx][1],
           **{c: raw[i] for i, c in enumerate(obj_cols)}}
    row['Desirability'] = total_desirability(row)
    final_results.append(row)

res  = pd.DataFrame(final_results).sort_values('Desirability', ascending=False)
best = res.iloc[0]
res.to_excel('NSGA3_GPR_Full_Results.xlsx', index=False)


# ======================================================
# INSERT SENSITIVITY ANALYSIS HERE 
# ======================================================
print("\nRunning Perturbation Analysis for Journal Visuals...")
perturbations = np.linspace(-0.2, 0.2, 25) # -20% to +20%
sens_targets = ['BSFC', 'NOx', 'BTE']
colors = ['red', 'blue', 'green']

plt.figure(figsize=(10, 6))

for i, target in enumerate(sens_targets):
    changes = []
    base_val = best[target]
    # Get the index of this target in our obj_cols list
    t_idx = obj_cols.index(target)
    
    for p in perturbations:
        # Vary RPM while keeping the optimal Blend constant
        new_rpm = best['RPM'] * (1 + p)
        # Use your existing evaluate_raw function
        new_vals = evaluate_raw([new_rpm, best['BLEND']])
        new_val = new_vals[t_idx]
        
        # Calculate % change from the optimal baseline
        changes.append(((new_val - base_val) / base_val) * 100)
    
    plt.plot(perturbations * 100, changes, label=f'Sensitivity of {target}', 
             color=colors[i], marker='o', markersize=4, lw=2)

plt.axhline(0, color='black', lw=1, ls='--')
plt.axvline(0, color='black', lw=1, ls='--')
plt.xlabel('% Deviation from Optimal RPM', fontsize=12)
plt.ylabel('% Change in Objective Value', fontsize=12)
plt.title('System Sensitivity Analysis at Optimal Point', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

# ADD THESE NEW LINES RIGHT AFTER 524, BEFORE THE plt.savefig:
sens_rows = []
for target in sens_targets:
    base_val = best[target]
    t_idx = obj_cols.index(target)
    for p in perturbations:
        new_vals = evaluate_raw([best['RPM'] * (1 + p), best['BLEND']])
        pct_change = ((new_vals[t_idx] - base_val) / base_val) * 100
        sens_rows.append({'Target': target, 'Perturbation_%': p * 100, 'Change_%': pct_change})
pd.DataFrame(sens_rows).to_csv('nsga3_sensitivity_data.csv', index=False)
print("Sensitivity CSV saved.")

plt.savefig(f'{save_folder}/sensitivity_analysis_fixed.png', dpi=300)
# ======================================================



print("\n" + "="*60)
print("BEST SOLUTION (Highest Desirability)")
print("="*60)
print(best.to_string())

# ======================================================
#  PLOTS
# ======================================================

# 1. Convergence
plt.figure()
plt.plot(conv, color='green', lw=2)
plt.title('NSGA-III GPR Convergence (Mean Desirability)')
plt.xlabel('Generation'); plt.ylabel('Mean Desirability')
plt.grid(True)
plt.savefig(f'{save_folder}/convergence.png', bbox_inches='tight')

# 2. Pareto Front Size History
plt.figure()
plt.plot(arch_hist, color='steelblue', lw=2)
plt.title('Pareto Front Size — NSGA-III')
plt.xlabel('Generation'); plt.ylabel('Pareto Front Size')
plt.grid(True)
plt.savefig(f'{save_folder}/pareto_size.png', bbox_inches='tight')

# 3. Desirability Surface
rpm_l   = np.linspace(*rpm_bounds,   40)
blend_l = np.linspace(*blend_bounds, 40)
R, B    = np.meshgrid(rpm_l, blend_l)
Z       = np.zeros_like(R)
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        raw    = evaluate_raw([R[i, j], B[i, j]])
        Z[i,j] = total_desirability({c: raw[k] for k, c in enumerate(obj_cols)})
plt.figure()
plt.contourf(R, B, Z, 20, cmap='viridis')
plt.colorbar(label='Desirability')
plt.scatter(res['RPM'], res['BLEND'], c='white', s=20, label='Pareto Front')
plt.scatter(best['RPM'], best['BLEND'], c='red', s=100,
            edgecolors='white', zorder=5, label='Best')
plt.xlabel('RPM'); plt.ylabel('Blend %')
plt.title('Desirability Surface — NSGA-III')
plt.legend()
plt.savefig(f'{save_folder}/surface.png', bbox_inches='tight')

# 4. Radar Chart
labels  = ['BSFC', 'BTE', 'POWER', 'TORQUE', 'NOx', 'CO2']
angles  = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]

def get_scaled(row):
    vals = [(row[l] - obj_bounds[l][0]) / (obj_bounds[l][1] - obj_bounds[l][0])
            for l in labels]
    return vals + [vals[0]]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles, get_scaled(best), 'b-', lw=2, label='Best Solution')
ax.fill(angles, get_scaled(best), 'b', alpha=0.1)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=12)
plt.title('Best Solution — Objective Radar', pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.savefig(f'{save_folder}/radar.png', bbox_inches='tight')

# 5. Correlation Heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(res[obj_cols + ['Desirability']].corr(),
            annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Objective Correlation Heatmap — NSGA-III Pareto Front')
plt.savefig(f'{save_folder}/heatmap.png', bbox_inches='tight')

# 6. Pareto Scatter (BSFC vs BTE)
plt.figure(); plt.scatter(res['BSFC'], res['BTE'], c=res['Desirability'], cmap='viridis')
plt.xlabel('BSFC'); plt.ylabel('BTE'); plt.colorbar(label='Desirability'); plt.savefig(f'{save_folder}/scatter.png')

# 7. Reference Point Association Plot (NEW — unique to NSGA-III)
ideal_f, nadir_f = get_ideal_nadir(F_pop)
F_norm_f         = normalize_population(F_pop, ideal_f, nadir_f)
assoc_f, dist_f  = associate_to_reference(F_norm_f, REF_POINTS)

plt.figure()
niche_usage = np.bincount(assoc_f, minlength=len(REF_POINTS))
plt.bar(range(len(REF_POINTS)), niche_usage, color='steelblue', alpha=0.8)
plt.xlabel('Reference Point Index'); plt.ylabel('Solutions Assigned')
plt.title('Reference Point Niche Utilisation — Final Generation')
plt.grid(axis='y', alpha=0.4)
plt.savefig(f'{save_folder}/niche_utilisation.png', bbox_inches='tight')

print(f"\nAll charts saved in '{save_folder}/'")
plt.show()

# Save the Pareto front for the Visualizer Script
res.to_csv('nsga3_pareto_results.csv', index=False)
print("Data saved for Visualization script.")


