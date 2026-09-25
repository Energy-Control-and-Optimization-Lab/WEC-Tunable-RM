"""
Mass_St5_A_OptimizationSetup.py - Constrained Minimization of Total Mass via SLSQP
with Monthly Battery (ESS) Feasibility Constraints

Minimizes analytically-computed displaced mass (D1..D5) subject to a monthly
energy-storage feasibility framework: generation-battery-load balance,
discharge limit based on previous battery state, and state-of-charge (SOC)
bounds. D6 (B_PTO) and D7 (K_PTO) do not enter the mass function directly,
but are still free to vary and are implicitly constrained through
AP(m; D_i) in the monthly balance constraint.

Author: Pablo Antonio Matamala Carvajal
Date: 2025-06-18
Updated: 2025-09-11 - Added D7 (PTO stiffness K_PTO). Parameter ranges for
                      D1..D7 now match St1_A_DoeValues.py exactly (same
                      domain the metamodels were trained on).
Updated: 2026-09-22 - Switched from a single-year (2024) monthly profile
                      repeated N_YEARS times, to the REAL 15-year MCP wave
                      climatology (see St3_A_Spectra.py / St4_B_MetaModel.py):
                      N_YEARS now selects the first N_YEARS real
                      climatological years (y01=2010 .. y15=2024, in
                      chronological order, no cycling) instead of
                      repeating one fixed 12-month pattern. N_YEARS > 15
                      is rejected (no real data beyond 2024). Loads
                      N_YEARS x 12 distinct monthly metamodels instead of
                      12. Battery sizing policy also changed: C_B is no
                      longer sized from "year 1" only -- it is now sized
                      independently for EACH of the N_YEARS years (same
                      self-resetting E_net calculation, run once per
                      year) and the design uses the WORST (maximum) of
                      those N_YEARS candidate capacities, since the years
                      are now genuinely different instead of identical
                      repeats. H_MONTH still ignores leap years (same
                      fixed 12-value hours-per-month pattern tiled
                      N_YEARS times) -- deliberate simplification, not an
                      oversight.

Description:
- Loads N_YEARS x 12 monthly P_sea_yNN_Mon metamodels from St4_B_MetaModel
  output (used to predict AP(m; D_i) for the battery feasibility framework).
- Mass is computed analytically (compute_mass), NOT from a metamodel.
- Builds the monthly battery model: E_diff -> E_charge -> E_net (reset per
  year) -> C_B (= worst of the N_YEARS per-year sizings) -> E_B -> SOC ->
  E_B_out, identical per-year formulation to the original PMR script,
  generalized across multiple distinct years.
- Minimizes M_total [kg] using SLSQP (gradient-based, supports constraints)
- Constraints (per month m = 1..N_MONTHS) -- ONLY the original four:
    1) Generation-battery-load balance:  AP(m)*h(m) + E_B_out(m) >= PT*h(m)
    2) Discharge limit:                  E_B_out(m) <= eta_discharge * E_B(m-1)
    3) SOC bounds:                       SOC_min <= SOC(m) <= SOC_max
  NOTE: the minimum-deficit constraint (forces C_B>0) and the monthly
  overshoot cap (AP(m)<=2*PT) used in the PMR script are intentionally
  NOT included here, per explicit request -- the hypothesis being tested
  is that minimizing mass (which does not depend on D6) will naturally
  keep the design away from the "no battery needed" corner, since D6
  remains free to satisfy the (tautological) balance constraint without
  needing extreme geometry. This has NOT been proven analytically; verify
  empirically against the C_B values in the results (see end of script).
- Multi-start: N_STARTS random starting points for global coverage
- Ranks all converged, feasible candidates by MASS (ascending)
- Saves results as PKL, MAT, CSV
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import savemat

#%%============================================================================
# CONFIGURATION
#==============================================================================

# --- Metamodel folder (only monthly AP surrogates are needed here) ---
METAMODEL_FOLDER = "EcoData/St4_MetaModel/Analysis/individual_enhanced_metamodels"

# Month abbreviations, in calendar order, matching the P_sea_yNN_Mon
# metamodel names from St3_A_Spectra.py / St4_B_MetaModel.py.
MONTH_ABBR = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
N_MONTHS_YEAR = 12   # months in a single calendar year -- do not change

# Number of real climatological years available (y01=2010 .. y15=2024).
# Hard limit -- there is no real MCP wave data beyond this.
N_YEARS_AVAILABLE = 15

# --- Simulation horizon length, in years ------------------------------------
# Selects the first N_YEARS REAL climatological years, in chronological
# order (y01_Jan..y01_Dec, y02_Jan..y02_Dec, ...) -- NOT a repeated
# 12-month cycle. Each year is genuinely different (real interannual
# variability from the 15-yr MCP climatology), so N_YEARS=15 simulates
# the WEC's actual design life against 15 distinct real years rather than
# one year's profile repeated. Must be <= N_YEARS_AVAILABLE.
N_YEARS = 15   # <-- our case: full 15-year climatology

if N_YEARS > N_YEARS_AVAILABLE:
    raise ValueError(
        f"N_YEARS={N_YEARS} exceeds N_YEARS_AVAILABLE={N_YEARS_AVAILABLE}: "
        f"there is no real climatological data beyond year y{N_YEARS_AVAILABLE:02d} "
        f"(2024). Reduce N_YEARS, or extend the 15-yr MCP climatology "
        f"(St2_A_...MCP pipeline) with more WIS offshore data before raising this."
    )

N_MONTHS = N_MONTHS_YEAR * N_YEARS   # total months simulated

# =============================================================================
# GEOMETRIC CONSTANTS (fixed, same values as St2_A_Hydro.py)
# =============================================================================

F1 = 0.8    # Inner radius reference [m]  (float/spar connection radius)
F2 = 0.16   # Spar bottom flange height [m]
F3 = 0.6    # Float freeboard [m]  (not used in mass, kept for reference)
F4 = 1.6    # Spar freeboard [m]   (not used in mass, kept for reference)

# --- Water density ---
RHO = 1025.0   # [kg/m^3]  seawater density -- CONFIRM this matches your convention

# =============================================================================
# BATTERY (ESS) MODEL PARAMETERS — all editable here, nowhere else
# =============================================================================

PT             = 15.0   # [W]   Power target (AquaFort sensor suite demand x SF)
ETA_CHARGE     = 0.92   # [-]   Battery charging efficiency
ETA_DISCHARGE  = 0.92   # [-]   Battery discharging efficiency
SOC_MIN        = 0.10   # [-]   Lower usable state-of-charge bound
SOC_MAX        = 0.90   # [-]   Upper usable state-of-charge bound

# h(m): fixed generic hours-per-month pattern, deliberately NOT adjusted
# for leap years (Feb fixed at 28 days / 672 h every year) -- a real 15-yr
# span (2010-2024) mixes leap and non-leap Februaries, but tracking that
# per-year would only complicate the script/paper for a negligible effect
# on the monthly energy balance. Same fixed [12] pattern tiled N_YEARS
# times, same as before.
H_MONTH_1YR = np.array([744, 672, 744, 720, 744, 720,
                         744, 744, 720, 744, 720, 744], dtype=float)  # [12] hours
H_MONTH = np.tile(H_MONTH_1YR, N_YEARS)   # [N_MONTHS] hours, repeats per year

# Sea-state labels for the REAL N_YEARS climatological years, in
# chronological order (y01=2010 .. yNN), NOT a repeated 12-month cycle:
# e.g. for N_YEARS=3 -> y01_Jan..y01_Dec, y02_Jan..y02_Dec, y03_Jan..y03_Dec.
# Each label maps to its OWN distinct metamodel (loaded below).
SEA_LABELS_FULL = [f"y{yi:02d}_{mon}" for yi in range(1, N_YEARS + 1) for mon in MONTH_ABBR]
assert len(SEA_LABELS_FULL) == N_MONTHS

# Small numerical guard: if a design has no monthly deficit at all,
# C_B would be 0 and SOC = E_B / C_B would be a 0/0 division.
C_B_EPS = 1e-9

# NOTE (explicit, per user request): the minimum-deficit constraint and the
# monthly overshoot cap used in the PMR script are NOT applied here.
# See module docstring for the rationale and the required empirical check.

# --- Multi-start ---
N_STARTS = 100   # Number of random starting points
SEED     = 1000    # Reproducibility

# --- Parameter space (matches St1_A_DoeValues.py exactly — same domain
#     the metamodels were trained on, no extrapolation) ---
PARAMETER_RANGES = {
    'D1': [2.1,  2.8],      # Float diameter [m]
    'D2': [0.3,  0.5],      # Float external draft [m]
    'D3': [0.0,  0.20],     # Float internal draft offset [m]
    'D4': [3.9,  5.2],      # Spar draft [m]
    'D5': [3.0,  4.0],      # Spar plate diameter [m]
    'D6': [50000, 100000],  # PTO damping B_PTO [kg/s]  (not in mass fn)
    'D7': [0,    50000],    # PTO stiffness K_PTO [N/m]  (not in mass fn)
}
PARAMETER_NAMES = list(PARAMETER_RANGES.keys())

# --- Output ---
OUTPUT_FOLDER = "EcoData/St5_Optimization_Iterative"

# --- Duplicate filtering tolerance ---
UNIQUE_TOL = 1e-3   # Two candidates considered identical if all params differ < this

#%%============================================================================
# MASS FUNCTION (analytical, no metamodel)
#==============================================================================

def compute_mass(x):
    """
    Compute total displaced mass analytically from physical dimensions.
    Parameters (x):
        D1: Float diameter [m]
        D2: Float external draft [m]
        D3: Float internal draft offset [m]  (frustum cone height)
        D4: Spar draft [m]
        D5: Spar plate diameter [m]
        D6: PTO damping [kg/s]     (not used in mass calculation)
        D7: PTO stiffness [N/m]    (not used in mass calculation)
    """
    D1, D2, D3, D4, D5, D6, D7 = x
    # Frustum cone (buoy inner bottom)
    Vol_con = (D3 * np.pi / 3) * ((D1/2)**2 + (F1/2)**2 + (D1/2)*(F1/2))
    # Outer cylinder (buoy body)
    Vol_cil = (D1/2)**2 * np.pi * D2
    # Inner cylindrical hole (spar passes through buoy)
    Vol_hole = (F1/2)**2 * np.pi * (D2 + D3)
    # Spar tube
    Vol_spar = (F1/2)**2 * np.pi * (D4 - F2)
    # Spar base plate
    Vol_plate = (D5/2)**2 * np.pi * F2
    # Total displaced volume and mass
    Vol_total = Vol_con + Vol_cil - Vol_hole + Vol_spar + Vol_plate
    M_total   = Vol_total * RHO
    return M_total

#%%============================================================================
# SETUP
#==============================================================================

print("=" * 70)
print("CONSTRAINED OPTIMIZATION — MASS via SLSQP + Monthly Battery Feasibility")
print("=" * 70)
print(f"\n🎯 Objective:   MINIMIZE M_total [kg]  (analytical, D1..D5; D6,D7 not in mass fn)")
print(f"🔋 Battery model: PT={PT} W, eta_charge={ETA_CHARGE}, "
      f"eta_discharge={ETA_DISCHARGE}, SOC=[{SOC_MIN}, {SOC_MAX}]")
print(f"⚖️  Constraints: monthly balance + discharge limit + SOC bounds "
      f"({N_MONTHS} months x 4) = {4*N_MONTHS} inequality constraints")
print(f"   ⚠️  NO minimum-deficit constraint, NO overshoot cap (removed by request)")
print(f"🔧 Method:      SLSQP  ({N_STARTS} random starts)")
print(f"📂 Metamodels:  {METAMODEL_FOLDER}/  (monthly AP surrogates only)")
print(f"📁 Output:      {OUTPUT_FOLDER}/")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "final_results"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "raw_results"), exist_ok=True)

np.random.seed(SEED)

#%%============================================================================
# LOAD METAMODELS  (N_YEARS x 12 monthly AP(m) surrogates -- no PMR, no mass model)
#==============================================================================

def load_metamodel(name):
    path = os.path.join(METAMODEL_FOLDER, f"{name}_enhanced_metamodel.pkl")
    if not os.path.exists(path):
        print(f"❌ Metamodel not found: {path}")
        available = [f.replace('_enhanced_metamodel.pkl', '')
                     for f in os.listdir(METAMODEL_FOLDER)
                     if f.endswith('_enhanced_metamodel.pkl')] if os.path.exists(METAMODEL_FOLDER) else []
        print(f"   Available: {available}")
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ Loaded metamodel: {name:16s}  (R²={data['statistics']['r2']:.4f}, "
          f"MAPE={data['statistics']['mape']:.2f}%)")
    return data


print(f"\n{'='*70}")
print(f"LOADING METAMODELS  ({N_MONTHS} monthly surrogates, y01..y{N_YEARS:02d})")
print(f"{'='*70}")

mm_monthly = {}
for lbl in SEA_LABELS_FULL:   # already unique (no repeated years), N_MONTHS distinct labels
    mm_monthly[lbl] = load_metamodel(f"P_sea_{lbl}")

mm_pmr = load_metamodel("PMR")
mm_aap = load_metamodel("P_avg_annual")

if mm_pmr is None or mm_aap is None or any(v is None for v in mm_monthly.values()):
    print("❌ Cannot proceed — one or more required metamodels are missing.")
    sys.exit(1)

#%%============================================================================
# PREDICTION FUNCTIONS
#==============================================================================

def predict(metamodel_data, x):
    """Predict response for parameter vector x (physical units)."""
    model  = metamodel_data['model']
    poly   = metamodel_data['polynomial_features']
    x_2d   = np.array(x).reshape(1, -1)
    x_poly = poly.transform(x_2d)
    return float(model.predict(x_poly)[0])


def predict_pmr(x):
    return predict(mm_pmr, x)


def predict_aap(x):
    return predict(mm_aap, x)


def predict_AP_all_months(x):
    """Returns AP(m; x) for m = 1..N_MONTHS, cycling through the 12 monthly
    metamodels once per repeated year. Shape [N_MONTHS]."""
    return np.array([predict(mm_monthly[lbl], x) for lbl in SEA_LABELS_FULL])

#%%============================================================================
# BATTERY (ESS) MODEL  —  identical formulation to the validated PMR script
#==============================================================================
#
#   E_diff(m)    = AP(m)*h(m) - PT*h(m)                                (Eq. Ediff)
#   E_charge(m)  = eta_charge * E_diff(m)          if E_diff(m) > 0    (Eq. Estored)
#                = E_diff(m) / eta_discharge       if E_diff(m) < 0
#   E_net(m)     = min(0, E_net(m-1) + E_charge(m)), E_net(0)=0        (Eq. EstoredCum)
#   C_B          = |min_m E_net(m)| / (SOC_max - SOC_min)              (Eq. size_battery)
#   E_B(m)       = SOC_max * C_B + E_net(m)                            (Eq. Ees)
#   SOC(m)       = E_B(m) / C_B                                        (Eq. soc_definition)
#   E_B_out(m)   = 0                                if E_diff(m) >= 0  (Eq. Ebattout)
#                = -E_diff(m) / eta_discharge       if E_diff(m) < 0
#
# Index convention: E_net and E_B have length 13 (index 0 = "month 0"
# initial condition; index k = month k, k=1..12). AP, E_diff, E_charge,
# E_B_out have length 12 (index m_idx = calendar month m_idx+1).
#==============================================================================

# Simple cache keyed on x, since objective + all constraints for the same
# x share the identical battery-state computation (avoids redundant calls
# to the 12 monthly metamodels on every constraint evaluation).
_state_cache = {'x': None, 'state': None}


def compute_battery_state(x):
    """
    Evaluates the full monthly battery model for a design vector x.
    Cached on x to avoid recomputation across objective/constraint calls.

    Battery sizing policy (updated 2026-09-22): C_B is no longer sized
    from "year 1" only. Since each of the N_YEARS years now comes from a
    genuinely different real climatological year (y01=2010 .. y15=2024,
    not a repeated 12-month cycle), the battery must be sized for the
    WORST of those years, not an arbitrary first one. For EACH year y
    (1..N_YEARS) independently, the self-resetting E_net trajectory
    (Eq. EstoredCum / size_battery) is computed over that year's own 12
    months (E_net reset to 0 at the start of every year), giving a
    candidate C_B_y; the design capacity is C_B = max_y(C_B_y). The
    physical battery state E_B for the full horizon is then obtained with
    a two-sided clipped recursion (saturating at both SOC_min*C_B and
    SOC_max*C_B) using this fixed worst-case C_B, which additionally lets
    the SOC bound constraints (3 and 4) correctly catch a battery that
    becomes insufficient in a year other than the one that sized it.
    """
    x_key = tuple(np.round(np.asarray(x, dtype=float), 10))
    if _state_cache['x'] == x_key:
        return _state_cache['state']

    AP = predict_AP_all_months(x)                                   # [N_MONTHS]

    E_diff = AP * H_MONTH - PT * H_MONTH                            # [N_MONTHS]  Eq. Ediff

    E_charge = np.where(
        E_diff > 0,
        ETA_CHARGE * E_diff,
        E_diff / ETA_DISCHARGE
    )                                                                # [N_MONTHS]  Eq. Estored

    # --- Sizing pass: EACH of the N_YEARS years independently, take worst ---
    C_B_per_year = np.zeros(N_YEARS)
    for yi in range(N_YEARS):
        m0 = yi * N_MONTHS_YEAR         # first month index (0-based) of year yi+1
        E_charge_year = E_charge[m0:m0 + N_MONTHS_YEAR]
        E_net_year = np.zeros(N_MONTHS_YEAR + 1)                     # [13], E_net[0] = 0 (reset every year)
        for m in range(1, N_MONTHS_YEAR + 1):
            E_net_year[m] = min(0.0, E_net_year[m - 1] + E_charge_year[m - 1])   # Eq. EstoredCum
        worst_deficit_year = np.min(E_net_year)                      # <= 0
        C_B_per_year[yi] = abs(worst_deficit_year) / (SOC_MAX - SOC_MIN)   # Eq. size_battery

    worst_case_year = int(np.argmax(C_B_per_year)) + 1   # 1-based (y01=year 1, ..., yNN)
    C_B = float(C_B_per_year[worst_case_year - 1])        # design capacity = worst of the N_YEARS years
    # NOTE: C_B is fixed here (worst-case among all N_YEARS years) and
    # reused for the entire N_MONTHS horizon below.

    # --- Simulation pass: full horizon, fixed C_B, two-sided clip ---------
    E_B  = np.zeros(N_MONTHS + 1)
    SOC  = np.zeros(N_MONTHS + 1)
    if C_B > C_B_EPS:
        E_B[0] = SOC_MAX * C_B                                       # starts full, same as E_net(0)=0 -> Ees(0)
        for m in range(1, N_MONTHS + 1):
            E_B[m] = np.clip(E_B[m - 1] + E_charge[m - 1],
                              SOC_MIN * C_B, SOC_MAX * C_B)
        SOC = E_B / C_B
    else:
        # No real deficit in year 1 -> battery not needed (degenerate case).
        # Guard against 0/0: treat the device as permanently "full".
        E_B[:] = 0.0
        SOC[:] = SOC_MAX

    E_B_out = np.where(E_diff < 0, -E_diff / ETA_DISCHARGE, 0.0)     # [N_MONTHS]  Eq. Ebattout

    state = {
        'AP': AP, 'E_diff': E_diff, 'E_charge': E_charge,
        'C_B_per_year': C_B_per_year, 'worst_case_year': worst_case_year,
        'C_B': C_B, 'E_B': E_B, 'SOC': SOC,
        'E_B_out': E_B_out,
    }
    _state_cache['x'] = x_key
    _state_cache['state'] = state
    return state

#%%============================================================================
# OBJECTIVE AND CONSTRAINTS  (battery feasibility framework, 3 types)
#
# CORRECTION (2026-09-22): the monthly "generation-battery-load balance" and
# "discharge limited by previous battery state" constraints have been merged
# into a single constraint. The original two-constraint formulation applied
# ETA_DISCHARGE twice for the same physical discharge event (once inside
# E_charge(m) = E_diff(m)/ETA_DISCHARGE, and again as a separate availability
# cap E_B_out(m) <= ETA_DISCHARGE*E_B(m-1)), which double-penalizes the same
# loss mechanism. Standard BESS modeling applies the discharge efficiency
# once, when converting stored energy to energy delivered to the load; the
# amount that can be drawn is limited by the state of charge itself (already
# enforced by the SOC bounds constraints below), not by a second efficiency
# factor. The combined constraint below is mathematically equivalent to
# requiring AP(m)*h(m) + ETA_DISCHARGE*E_B(m-1) >= PT*h(m) for all m, i.e. a
# single application of ETA_DISCHARGE, and E_B_out is no longer used as an
# optimization constraint (it is still computed in compute_battery_state()
# for reporting/plotting purposes only).
#==============================================================================

# SLSQP minimizes -- mass is already a "minimize" objective, no sign flip needed
def objective(x):
    return compute_mass(x)


# --- Constraint 1 (per month): generation + battery (single-eta discharge) -
#     AP(m)*h(m) + eta_discharge*E_B(m-1) - PT*h(m) >= 0
def make_constraint_balance(m_idx):
    def _c(x):
        st = compute_battery_state(x)
        return st['AP'][m_idx] * H_MONTH[m_idx] + ETA_DISCHARGE * st['E_B'][m_idx] \
               - PT * H_MONTH[m_idx]
    return _c


# --- Constraint 2 (per month): SOC lower bound ------------------------------
#     SOC(m) - SOC_min >= 0
def make_constraint_soc_lower(m_idx):
    def _c(x):
        st = compute_battery_state(x)
        return st['SOC'][m_idx + 1] - SOC_MIN
    return _c


# --- Constraint 3 (per month): SOC upper bound ------------------------------
#     SOC_max - SOC(m) >= 0
def make_constraint_soc_upper(m_idx):
    def _c(x):
        st = compute_battery_state(x)
        return SOC_MAX - st['SOC'][m_idx + 1]
    return _c


constraints = []
for m_idx in range(N_MONTHS):
    constraints.append({'type': 'ineq', 'fun': make_constraint_balance(m_idx)})
    constraints.append({'type': 'ineq', 'fun': make_constraint_soc_lower(m_idx)})
    constraints.append({'type': 'ineq', 'fun': make_constraint_soc_upper(m_idx)})

constraint_label = (f"Monthly battery feasibility: balance (single-eta discharge) "
                     f"+ SOC in [{SOC_MIN}, {SOC_MAX}]  ({len(constraints)} constraints)")

bounds = [PARAMETER_RANGES[p] for p in PARAMETER_NAMES]

#%%============================================================================
# FEASIBILITY CHECK HELPER
#==============================================================================

def check_feasibility(x, tol=1e-6):
    """
    Re-evaluates all monthly constraints for a converged design and returns
    (is_feasible, worst_violation, battery_state_dict).
    """
    st = compute_battery_state(x)
    violations = []
    for m_idx in range(N_MONTHS):
        violations.append(make_constraint_balance(m_idx)(x))
        violations.append(make_constraint_soc_lower(m_idx)(x))
        violations.append(make_constraint_soc_upper(m_idx)(x))
    worst = min(violations)
    return (worst >= -tol), worst, st

#%%============================================================================
# LIVE RESULTS SAVING  (filter/dedup/rank + atomic write-then-rename)
#
# Called once per start (see multi-start loop below) so that
# Mass_optimal_designs.pkl/.csv/.mat get updated progressively as the
# optimization runs, instead of only once at the very end. This lets you
# inspect results, plot them, or copy the files to another folder while
# the script is still running.
#
# Each file is written to a "<name>.tmp" path first and then moved onto
# the final name with os.replace(), which is an atomic rename on both
# POSIX and Windows. That guarantees a reader (MATLAB, a copy command,
# you opening the CSV) never sees a half-written file: at any instant the
# final-named file is either the previous complete version or the new
# complete version, never something in between.
#==============================================================================

def _atomic_write(write_func, final_path):
    """write_func(tmp_path) writes the full file to tmp_path; only then is
    it moved (atomically) onto final_path."""
    tmp_path = final_path + '.tmp'
    write_func(tmp_path)
    os.replace(tmp_path, final_path)


def _write_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_results(raw_results_so_far, verbose=True):
    """
    Filters raw_results_so_far to the feasible ones, deduplicates, ranks by
    mass (ascending), and (over)writes Mass_optimal_designs.pkl/.csv/.mat +
    all_starts_raw.pkl, all via atomic write-then-rename.

    Returns (unique, n_feasible). If no feasible candidate exists yet,
    returns (None, 0) and writes NOTHING -- any previously written files
    from an earlier, more advanced start are left untouched (never blanked
    out or overwritten with an "empty" result).
    """
    feasible = [r for r in raw_results_so_far if r['constraint_satisfied']]
    n_feasible = len(feasible)
    if not feasible:
        return None, 0

    unique = []
    for r in sorted(feasible, key=lambda x: x['M_total']):   # ascending: smaller mass first
        is_dup = False
        for u in unique:
            if np.max(np.abs(r['x'] - u['x']) / np.array([hi - lo for lo, hi in bounds])) < UNIQUE_TOL:
                is_dup = True
                break
        if not is_dup:
            unique.append(r)
    unique.sort(key=lambda x: x['M_total'])   # ascending

    rows = []
    for rank, r in enumerate(unique, 1):
        row = {'rank': rank, 'M_total_kg': r['M_total'], 'PMR_predicted': r['PMR'],
               'AAP_predicted': r['AAP'], 'C_B_Wh': r['C_B'],
               'worst_case_year': r['worst_case_year']}
        for j, p in enumerate(PARAMETER_NAMES):
            row[p] = r['x'][j]
        for yi in range(1, N_YEARS + 1):
            row[f'C_B_y{yi:02d}_Wh'] = r['C_B_per_year'][yi - 1]
        for m_idx in range(N_MONTHS):
            lbl = SEA_LABELS_FULL[m_idx]   # already unique per year-month, e.g. 'y01_Jan'
            row[f'AP_{lbl}'] = r['AP_monthly'][m_idx]
            row[f'SOC_{lbl}'] = r['SOC_monthly'][m_idx]
        rows.append(row)
    df = pd.DataFrame(rows)

    results_pkg = {
        'unique_candidates':   unique,
        'design_matrix':       np.array([r['x'] for r in unique]),
        'M_total_values':      np.array([r['M_total'] for r in unique]),
        'PMR_values':          np.array([r['PMR'] for r in unique]),
        'AAP_values':          np.array([r['AAP'] for r in unique]),
        'C_B_values':          np.array([r['C_B'] for r in unique]),
        'C_B_per_year_matrix': np.array([r['C_B_per_year'] for r in unique]),   # [n_unique x N_YEARS]
        'worst_case_year_values': np.array([r['worst_case_year'] for r in unique]),
        'AP_monthly_matrix':   np.array([r['AP_monthly'] for r in unique]),
        'SOC_monthly_matrix':  np.array([r['SOC_monthly'] for r in unique]),
        'sea_labels_full':     SEA_LABELS_FULL,   # N_YEARS x 12 real labels, y01_Jan..yNN_Dec
        'n_years':             N_YEARS,
        'n_years_available':   N_YEARS_AVAILABLE,
        'parameter_names':     PARAMETER_NAMES,
        'parameter_ranges':    PARAMETER_RANGES,
        'battery_params': {
            'PT': PT, 'ETA_CHARGE': ETA_CHARGE, 'ETA_DISCHARGE': ETA_DISCHARGE,
            'SOC_MIN': SOC_MIN, 'SOC_MAX': SOC_MAX, 'H_MONTH': H_MONTH,
        },
        'n_starts_so_far': len(raw_results_so_far),   # NEW: how many starts went into this snapshot
        'n_starts':   N_STARTS,
        'n_feasible': n_feasible,
        'n_unique':   len(unique),
    }

    pkl_path = os.path.join(OUTPUT_FOLDER, "final_results", "Mass_optimal_designs.pkl")
    try:
        _atomic_write(lambda p: _write_pkl(p, results_pkg), pkl_path)
        if verbose:
            print(f"✅ PKL saved: Mass_optimal_designs.pkl")
    except Exception as e:
        print(f"⚠️  PKL error: {e}")

    csv_path = os.path.join(OUTPUT_FOLDER, "final_results", "Mass_optimal_designs.csv")
    try:
        _atomic_write(lambda p: df.to_csv(p, index=False, float_format='%.6f'), csv_path)
        if verbose:
            print(f"✅ CSV saved: Mass_optimal_designs.csv")
    except Exception as e:
        print(f"⚠️  CSV error (file may be open elsewhere): {e}")

    try:
        mat_data = {
            'design_matrix':      results_pkg['design_matrix'],
            'M_total_values':     results_pkg['M_total_values'],
            'PMR_values':         results_pkg['PMR_values'],
            'AAP_values':         results_pkg['AAP_values'],
            'C_B_values':         results_pkg['C_B_values'],
            'C_B_per_year_matrix': results_pkg['C_B_per_year_matrix'],
            'worst_case_year_values': results_pkg['worst_case_year_values'],
            'AP_monthly_matrix':  results_pkg['AP_monthly_matrix'],
            'SOC_monthly_matrix': results_pkg['SOC_monthly_matrix'],
            'parameter_names':    np.array(PARAMETER_NAMES, dtype=object),
            'PT':                 PT,
            'n_years':             N_YEARS,
            'n_unique':            len(unique),
        }
        mat_path = os.path.join(OUTPUT_FOLDER, "final_results", "Mass_optimal_designs.mat")
        _atomic_write(lambda p: savemat(p, mat_data, do_compression=True), mat_path)
        if verbose:
            print(f"✅ MAT saved: Mass_optimal_designs.mat")
    except Exception as e:
        print(f"⚠️  MAT error: {e}")

    try:
        raw_path = os.path.join(OUTPUT_FOLDER, "raw_results", "all_starts_raw.pkl")
        _atomic_write(lambda p: _write_pkl(p, raw_results_so_far), raw_path)
        if verbose:
            print(f"✅ Raw results saved: all_starts_raw.pkl")
    except Exception as e:
        print(f"⚠️  Raw results save error: {e}")

    return unique, n_feasible

#%%============================================================================
# MULTI-START SLSQP OPTIMIZATION
#==============================================================================

print(f"\n{'='*70}")
print(f"MULTI-START SLSQP  ({N_STARTS} starts)")
print(f"{'='*70}")
print(f"Constraint: {constraint_label}")

starts = np.column_stack([
    np.random.uniform(lo, hi, N_STARTS)
    for lo, hi in bounds
])

raw_results = []
_t_opt_start = time.time()

for i in range(N_STARTS):
    x0 = starts[i]
    _t_start_i = time.time()

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 200, 'disp': False}
        )

        if res.success or res.status in (0, 1, 2):
            mass_val = res.fun
            pmr_val = predict_pmr(res.x)
            aap_val = predict_aap(res.x)
            is_feasible, worst_violation, st = check_feasibility(res.x)

            raw_results.append({
                'run_id':               i + 1,
                'x':                    res.x.copy(),
                'M_total':              mass_val,
                'PMR':                  pmr_val,
                'AAP':                  aap_val,
                'AP_monthly':           st['AP'].copy(),
                'C_B':                  st['C_B'],
                'C_B_per_year':         st['C_B_per_year'].copy(),
                'worst_case_year':      st['worst_case_year'],
                'SOC_monthly':          st['SOC'][1:].copy(),
                'E_B_out_monthly':      st['E_B_out'].copy(),
                'constraint_satisfied': is_feasible,
                'worst_violation':      worst_violation,
                'converged':            res.success,
                'message':              res.message,
                'n_iter':               res.nit,
            })

    except Exception as e:
        pass

    # Live update: re-filter/dedup/rank everything accumulated so far and
    # (over)write Mass_optimal_designs.pkl/.csv/.mat via atomic
    # write-then-rename (see save_results() above). Safe to copy these
    # files to another folder at any point while the script keeps running.
    _unique_so_far, _ = save_results(raw_results, verbose=False)
    n_unique_so_far = len(_unique_so_far) if _unique_so_far is not None else 0

    _dt_i = time.time() - _t_start_i
    _elapsed = time.time() - _t_opt_start
    _avg = _elapsed / (i + 1)
    _eta_min = _avg * (N_STARTS - (i + 1)) / 60.0
    n_ok = sum(1 for r in raw_results if r['constraint_satisfied'])
    print(f"   Start {i+1:3d}/{N_STARTS}  |  {_dt_i:6.1f}s  "
          f"|  converged: {len(raw_results)}  |  constraint satisfied: {n_ok}  "
          f"|  unique saved: {n_unique_so_far}  "
          f"|  avg: {_avg:5.1f}s/start  |  ETA: {_eta_min:6.1f} min")

print(f"\n✅ Optimization completed:")
print(f"   Total starts:           {N_STARTS}")
print(f"   Converged:              {len(raw_results)}")
n_feasible = sum(1 for r in raw_results if r['constraint_satisfied'])
print(f"   Constraint satisfied:   {n_feasible}")

#%%============================================================================
# FINAL FILTER, DEDUPLICATE, RANK, SAVE
#
# Reuses the exact same save_results() function called after every start
# during the live updates above -- so the final files are guaranteed to
# be consistent with whatever a live viewer would have seen after the
# last start (same dedup/ranking logic, same atomic write-then-rename).
#==============================================================================

feasible = [r for r in raw_results if r['constraint_satisfied']]

print(f"\n{'='*70}")
print("SAVING FINAL RESULTS")
print(f"{'='*70}")

unique, n_feasible = save_results(raw_results, verbose=True)

if unique is None:
    print("\n⚠️  No feasible solutions found.")
    print(f"   Consider checking metamodel quality for the monthly P_sea surrogates,")
    print(f"   or revisiting PT / eta_charge / eta_discharge / SOC bounds.")
    sys.exit(1)

print(f"\n📊 Unique candidates after deduplication: {len(unique)}")
N_SHOW = min(50, len(unique))
print(f"\n🏆 Top {N_SHOW} candidates (MASS ranking, ascending):")
print(f"   {'Rank':>4}  {'M_total [kg]':>12}  {'C_B [Wh]':>10}  "
      + "  ".join([f"{p:>8}" for p in PARAMETER_NAMES]))
print("   " + "-" * (4 + 12 + 10 + 10 * len(PARAMETER_NAMES) + 10))

for rank, r in enumerate(unique[:N_SHOW], 1):
    params_str = "  ".join([f"{v:8.3f}" for v in r['x']])
    print(f"   {rank:4d}  {r['M_total']:12.2f}  {r['C_B']:10.1f}  {params_str}")

# ------------------------------------------------------------------
# EMPIRICAL CHECK: how many top candidates ended up with C_B == 0?
# (Verifies the hypothesis that mass minimization alone keeps designs
#  away from the "no battery needed" corner -- NOT guaranteed a priori.)
# ------------------------------------------------------------------
n_zero_cb = sum(1 for r in unique[:N_SHOW] if r['C_B'] <= C_B_EPS)
print(f"\n🔍 Empirical check — candidates with C_B ≈ 0 among top {N_SHOW}: {n_zero_cb}")
if n_zero_cb > 0:
    print(f"   ⚠️  WARNING: {n_zero_cb} of the top {N_SHOW} candidates need NO battery.")
    print(f"      Mass minimization alone did NOT avoid the degenerate corner.")
    print(f"      Consider re-adding the minimum-deficit constraint if this is unwanted.")
else:
    print(f"   ✅ All top {N_SHOW} candidates require a nonzero battery (C_B > 0).")

#%%============================================================================
# PLOTS
#==============================================================================

try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Constrained Optimization — Minimize Mass  |  '
                 f'Monthly Battery Feasibility (PT={PT} W)',
                 fontsize=13, fontweight='bold')

    mass_all = [r['M_total'] for r in feasible]
    axes[0].hist(mass_all, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(unique[0]['M_total'], color='red', linewidth=2,
                    label=f'Best: {unique[0]["M_total"]:.1f} kg')
    axes[0].set_xlabel('M_total [kg]', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Mass Distribution (feasible solutions)', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    cb_all = [r['C_B'] for r in feasible]
    axes[1].scatter(cb_all, mass_all, color='steelblue', alpha=0.6, s=30, label='Feasible')
    axes[1].scatter([unique[0]['C_B']], [unique[0]['M_total']],
                    color='red', s=100, zorder=5, label='Best candidate')
    axes[1].set_xlabel('Required battery capacity $C_B$ [Wh]', fontsize=11)
    axes[1].set_ylabel('M_total [kg]', fontsize=11)
    axes[1].set_title('Mass vs Battery Capacity Trade-off', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_FOLDER, "Mass_optimization_results.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Plot saved: Mass_optimization_results.png")
except Exception as e:
    print(f"⚠️  Plot 1 error: {e}")

try:
    best = unique[0]
    # months_x: continuous month index 1..N_MONTHS across all N_YEARS real years.
    # With N_YEARS up to 15 (180 months), a per-month x-tick would be illegible
    # (labels used to show SEA_LABELS_FULL[i].split('_')[0], which after the
    # yNN_Mon relabeling would show 'y01' instead of the month — fixed here):
    # only the first month of each calendar year gets a tick/label (its real
    # year, y01=2010..yNN), and the year that sized the battery (worst_case_year)
    # is marked with a vertical line.
    fig2, ax2 = plt.subplots(figsize=(max(10, N_YEARS * 1.1), 5))
    months_x = np.arange(1, N_MONTHS + 1)
    ax2.plot(months_x, best['SOC_monthly'], '-', color='steelblue', linewidth=1.3,
             marker='o', markersize=2.5)
    ax2.axhline(SOC_MIN, color='red', linestyle='--', linewidth=1.2, label=f'SOC_min={SOC_MIN}')
    ax2.axhline(SOC_MAX, color='green', linestyle='--', linewidth=1.2, label=f'SOC_max={SOC_MAX}')

    year_tick_pos = [yi * N_MONTHS_YEAR + 1 for yi in range(N_YEARS)]   # Jan of each year
    year_tick_lbl = [str(2009 + yi) for yi in range(1, N_YEARS + 1)]    # y01->2010, ..., yNN
    ax2.set_xticks(year_tick_pos)
    ax2.set_xticklabels(year_tick_lbl, rotation=45 if N_YEARS > 6 else 0)

    wcy = best['worst_case_year']
    wcy_month0 = (wcy - 1) * N_MONTHS_YEAR + 1
    ax2.axvspan(wcy_month0, wcy_month0 + N_MONTHS_YEAR - 1, color='orange', alpha=0.15,
                label=f'Worst-case year (y{wcy:02d} = {2009+wcy}, sized C_B)')

    ax2.set_xlabel('Year (tick = January)', fontsize=11)
    ax2.set_ylabel('SOC(m)', fontsize=11)
    ax2.set_title(f'Best Candidate — Monthly SOC Trajectory over {N_YEARS} real climatological year(s) '
                  f'(C_B={best["C_B"]:.1f} Wh, sized by y{wcy:02d}={2009+wcy})',
                  fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.set_xlim([1, N_MONTHS])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    soc_plot_path = os.path.join(OUTPUT_FOLDER, "Best_Candidate_SOC_trajectory.png")
    fig2.savefig(soc_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"✅ Plot saved: Best_Candidate_SOC_trajectory.png")
except Exception as e:
    print(f"⚠️  Plot 2 error: {e}")

#%%============================================================================
# FINAL SUMMARY
#==============================================================================

print(f"\n{'='*70}")
print("🎉 CONSTRAINED OPTIMIZATION COMPLETED")
print(f"{'='*70}")
print(f"   Objective:    MINIMIZE M_total [kg]")
print(f"   Constraint:   {constraint_label}")
print(f"   Starts:       {N_STARTS}  |  Feasible: {n_feasible}  |  Unique: {len(unique)}")
print(f"\n🏆 Best candidate:")
best = unique[0]
for j, p in enumerate(PARAMETER_NAMES):
    lo, hi = PARAMETER_RANGES[p]
    print(f"   {p}: {best['x'][j]:.4f}  (range [{lo}, {hi}])")
print(f"   M_total:      {best['M_total']:.2f} kg")
print(f"   PMR:          {best['PMR']:.4f} W/t")
print(f"   AAP:          {best['AAP']:.4f} W")
print(f"   C_B (battery): {best['C_B']:.1f} Wh  "
      f"{'⚠️  ZERO — no battery needed for this design!' if best['C_B'] <= C_B_EPS else ''}")
print(f"   Worst-case year: y{best['worst_case_year']:02d} ({2009+best['worst_case_year']})  "
      f"— sized the battery among the {N_YEARS} real climatological years")
print(f"   SOC range:    [{best['SOC_monthly'].min():.3f}, {best['SOC_monthly'].max():.3f}]")
print(f"\n📁 Results in: {OUTPUT_FOLDER}/final_results/")
print(f"   Mass_optimal_designs.pkl / .csv / .mat")
print("=" * 70)
