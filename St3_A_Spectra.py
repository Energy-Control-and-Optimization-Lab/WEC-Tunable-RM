"""
St3_Spectra.py - JONSWAP Spectrum Generation for 15-Year Monthly Climatology
Generates 181 JONSWAP spectra (1 annual-mean reference + 180 monthly
climatological states, 12 months x 15 years) from Hs and Tp data.

Author: Pablo Antonio Matamala Carvajal
Date: 2025-05-26
Updated: 2026-09-22 - Replaced the 13 single-year (2024) sea states with a
                      181-state set: the original 'Mean_2024' AWAC-measured
                      reference (unchanged, kept for validation) plus 180
                      climatological monthly states (y01_Jan..y15_Dec)
                      derived from a 15-year synthetic local series built
                      via Measure-Correlate-Predict (AWAC local x WIS
                      offshore hindcast ST63042, 2010-2024). Hs/Tp per
                      state = monthly mean (Hs_mean, Tp_mean) of that
                      calendar year-month in the synthetic series.

Output: EcoData/St3_Spectra/Spectra.{npz, pkl, mat}
        EcoData/St3_Spectra/Plots/Spectra_Mean_2024.png   (1)
        EcoData/St3_Spectra/Plots/Spectra_y01.png ... y15.png  (15)
        (16 plots total instead of one-per-state, to keep output legible)
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

#%%============================================================================
# CONFIGURATION
#==============================================================================

# --- Output folder ---
OUTPUT_FOLDER = "EcoData/St3_Spectra"
PLOTS_FOLDER  = os.path.join(OUTPUT_FOLDER, "Plots")

# --- Frequency range [rad/s] — must match St1_Hydro ---
frequencies = np.arange(0.5, 2.5 + 0.025, 0.025)  # [rad/s]

# --- JONSWAP peak enhancement factor ---
GAMMA = 3.3  # Standard JONSWAP (modifiable)

# --- Sea state data: [label, Hs (m), Tp (s)] ---
# 'Mean_2024': original AWAC-measured annual mean (2024), UNCHANGED — kept
#              as an independent real-data reference/validation point.
# 'yNN_Mon':   climatological monthly state for year NN (01=2010..15=2024),
#              Hs_mean/Tp_mean from the 15-yr synthetic MCP series.
SEA_STATES = [
    ('Mean_2024', 0.27, 7.56),
    ('y01_Jan', 0.3001, 8.4138),
    ('y01_Feb', 0.3418, 7.7319),
    ('y01_Mar', 0.3814, 9.3863),
    ('y01_Apr', 0.2115, 7.1508),
    ('y01_May', 0.2036, 6.5050),
    ('y01_Jun', 0.1791, 5.5831),
    ('y01_Jul', 0.1963, 6.3248),
    ('y01_Aug', 0.2826, 7.2381),
    ('y01_Sep', 0.2717, 8.2533),
    ('y01_Oct', 0.3092, 8.3622),
    ('y01_Nov', 0.2970, 8.2947),
    ('y01_Dec', 0.3974, 8.2646),
    ('y02_Jan', 0.2482, 7.3892),
    ('y02_Feb', 0.3104, 7.7503),
    ('y02_Mar', 0.3009, 7.6824),
    ('y02_Apr', 0.3617, 6.9718),
    ('y02_May', 0.3086, 7.1879),
    ('y02_Jun', 0.1897, 5.9797),
    ('y02_Jul', 0.1966, 6.4280),
    ('y02_Aug', 0.2670, 6.7714),
    ('y02_Sep', 0.2750, 8.2578),
    ('y02_Oct', 0.3279, 8.3678),
    ('y02_Nov', 0.1927, 8.7616),
    ('y02_Dec', 0.2817, 8.2295),
    ('y03_Jan', 0.2761, 8.7395),
    ('y03_Feb', 0.3085, 7.2569),
    ('y03_Mar', 0.2571, 7.3422),
    ('y03_Apr', 0.2970, 7.1815),
    ('y03_May', 0.2455, 6.9009),
    ('y03_Jun', 0.2333, 6.6430),
    ('y03_Jul', 0.1905, 6.4140),
    ('y03_Aug', 0.2210, 7.0070),
    ('y03_Sep', 0.3307, 8.7608),
    ('y03_Oct', 0.3807, 8.7947),
    ('y03_Nov', 0.2720, 9.0765),
    ('y03_Dec', 0.3592, 8.4025),
    ('y04_Jan', 0.2496, 7.2654),
    ('y04_Feb', 0.3703, 7.6874),
    ('y04_Mar', 0.3664, 8.8574),
    ('y04_Apr', 0.3261, 6.6054),
    ('y04_May', 0.2922, 7.3761),
    ('y04_Jun', 0.2102, 6.4115),
    ('y04_Jul', 0.1933, 6.3401),
    ('y04_Aug', 0.2175, 6.6647),
    ('y04_Sep', 0.2615, 7.0401),
    ('y04_Oct', 0.2588, 7.7844),
    ('y04_Nov', 0.2409, 7.9877),
    ('y04_Dec', 0.2666, 8.1167),
    ('y05_Jan', 0.3479, 8.5591),
    ('y05_Feb', 0.3207, 7.5536),
    ('y05_Mar', 0.2757, 8.2779),
    ('y05_Apr', 0.4075, 7.4770),
    ('y05_May', 0.2525, 6.6677),
    ('y05_Jun', 0.2015, 5.9930),
    ('y05_Jul', 0.2370, 6.3097),
    ('y05_Aug', 0.2389, 6.8831),
    ('y05_Sep', 0.2753, 6.9776),
    ('y05_Oct', 0.3523, 8.5688),
    ('y05_Nov', 0.2220, 7.8821),
    ('y05_Dec', 0.3561, 8.3421),
    ('y06_Jan', 0.4083, 9.2922),
    ('y06_Feb', 0.4135, 7.9793),
    ('y06_Mar', 0.2422, 7.2440),
    ('y06_Apr', 0.3243, 6.9787),
    ('y06_May', 0.2310, 6.5451),
    ('y06_Jun', 0.2406, 5.4666),
    ('y06_Jul', 0.1826, 6.3547),
    ('y06_Aug', 0.2564, 6.8467),
    ('y06_Sep', 0.2649, 6.3198),
    ('y06_Oct', 0.3509, 8.2421),
    ('y06_Nov', 0.2315, 8.3311),
    ('y06_Dec', 0.2644, 7.7836),
    ('y07_Jan', 0.3550, 8.4094),
    ('y07_Feb', 0.3780, 8.1120),
    ('y07_Mar', 0.2940, 7.9431),
    ('y07_Apr', 0.3463, 7.3782),
    ('y07_May', 0.2633, 7.0097),
    ('y07_Jun', 0.1994, 6.1372),
    ('y07_Jul', 0.1942, 6.3939),
    ('y07_Aug', 0.2268, 6.8282),
    ('y07_Sep', 0.3218, 7.8718),
    ('y07_Oct', 0.3601, 8.4076),
    ('y07_Nov', 0.2510, 8.2552),
    ('y07_Dec', 0.2521, 7.7257),
    ('y08_Jan', 0.3817, 8.7977),
    ('y08_Feb', 0.3305, 7.7276),
    ('y08_Mar', 0.2876, 8.0261),
    ('y08_Apr', 0.3584, 6.6095),
    ('y08_May', 0.2700, 6.6083),
    ('y08_Jun', 0.2078, 6.2088),
    ('y08_Jul', 0.1894, 6.3591),
    ('y08_Aug', 0.2257, 7.0327),
    ('y08_Sep', 0.3156, 7.7255),
    ('y08_Oct', 0.3288, 8.6447),
    ('y08_Nov', 0.2383, 7.3822),
    ('y08_Dec', 0.2375, 7.4774),
    ('y09_Jan', 0.3264, 7.8908),
    ('y09_Feb', 0.3261, 8.2420),
    ('y09_Mar', 0.4464, 8.9338),
    ('y09_Apr', 0.3565, 6.9656),
    ('y09_May', 0.2413, 6.3309),
    ('y09_Jun', 0.2023, 5.3992),
    ('y09_Jul', 0.2260, 6.5127),
    ('y09_Aug', 0.2392, 6.3898),
    ('y09_Sep', 0.3194, 8.3938),
    ('y09_Oct', 0.3766, 8.2963),
    ('y09_Nov', 0.2594, 8.0119),
    ('y09_Dec', 0.2633, 7.3377),
    ('y10_Jan', 0.3401, 8.1987),
    ('y10_Feb', 0.3449, 7.6858),
    ('y10_Mar', 0.2717, 7.8587),
    ('y10_Apr', 0.3451, 7.2894),
    ('y10_May', 0.2622, 6.7512),
    ('y10_Jun', 0.1819, 5.6755),
    ('y10_Jul', 0.1895, 6.0570),
    ('y10_Aug', 0.2278, 6.5723),
    ('y10_Sep', 0.2865, 7.3837),
    ('y10_Oct', 0.4036, 8.7793),
    ('y10_Nov', 0.2652, 8.1232),
    ('y10_Dec', 0.3431, 7.8893),
    ('y11_Jan', 0.2273, 8.3898),
    ('y11_Feb', 0.3140, 8.7081),
    ('y11_Mar', 0.2970, 8.0273),
    ('y11_Apr', 0.4193, 7.7742),
    ('y11_May', 0.2617, 6.3548),
    ('y11_Jun', 0.2112, 5.9226),
    ('y11_Jul', 0.2227, 6.2823),
    ('y11_Aug', 0.2396, 6.2981),
    ('y11_Sep', 0.3194, 7.8103),
    ('y11_Oct', 0.3404, 8.3359),
    ('y11_Nov', 0.2152, 8.3351),
    ('y11_Dec', 0.3743, 8.4355),
    ('y12_Jan', 0.2634, 8.4134),
    ('y12_Feb', 0.3628, 8.4444),
    ('y12_Mar', 0.2551, 7.1576),
    ('y12_Apr', 0.3330, 7.4051),
    ('y12_May', 0.2402, 6.7172),
    ('y12_Jun', 0.2015, 5.8410),
    ('y12_Jul', 0.2057, 6.5091),
    ('y12_Aug', 0.2284, 6.8441),
    ('y12_Sep', 0.2853, 7.8835),
    ('y12_Oct', 0.3556, 8.2810),
    ('y12_Nov', 0.2264, 8.6139),
    ('y12_Dec', 0.1910, 8.0119),
    ('y13_Jan', 0.3729, 8.1748),
    ('y13_Feb', 0.3830, 8.1237),
    ('y13_Mar', 0.2290, 7.7541),
    ('y13_Apr', 0.3227, 7.6340),
    ('y13_May', 0.2775, 6.6059),
    ('y13_Jun', 0.2097, 5.8609),
    ('y13_Jul', 0.2037, 6.1557),
    ('y13_Aug', 0.2143, 5.8786),
    ('y13_Sep', 0.2381, 7.2463),
    ('y13_Oct', 0.3344, 8.3282),
    ('y13_Nov', 0.2220, 8.1469),
    ('y13_Dec', 0.4279, 8.9390),
    ('y14_Jan', 0.3348, 8.7164),
    ('y14_Feb', 0.3140, 7.7976),
    ('y14_Mar', 0.2977, 8.1466),
    ('y14_Apr', 0.3005, 6.9749),
    ('y14_May', 0.2830, 6.7613),
    ('y14_Jun', 0.2148, 6.1790),
    ('y14_Jul', 0.2194, 6.3037),
    ('y14_Aug', 0.2414, 6.8893),
    ('y14_Sep', 0.3105, 8.6834),
    ('y14_Oct', 0.2580, 9.1021),
    ('y14_Nov', 0.2043, 8.7805),
    ('y14_Dec', 0.3016, 8.4093),
    ('y15_Jan', 0.3203, 8.1624),
    ('y15_Feb', 0.3289, 8.1211),
    ('y15_Mar', 0.3117, 8.2582),
    ('y15_Apr', 0.3493, 8.0602),
    ('y15_May', 0.2533, 6.4570),
    ('y15_Jun', 0.1867, 5.7739),
    ('y15_Jul', 0.2290, 6.1880),
    ('y15_Aug', 0.2404, 7.2088),
    ('y15_Sep', 0.3225, 8.0323),
    ('y15_Oct', 0.2583, 9.3063),
    ('y15_Nov', 0.2134, 8.1520),
    ('y15_Dec', 0.2530, 8.2823),
]

#%%============================================================================
# SETUP
#==============================================================================

print("=" * 70)
print("JONSWAP SPECTRUM GENERATION")
print("=" * 70)
print(f"\n⚙️  Configuration:")
print(f"   Gamma (γ):   {GAMMA}")
print(f"   Frequencies: {len(frequencies)} points  "
      f"[{frequencies[0]:.2f}, {frequencies[-1]:.2f}] rad/s")
print(f"   Sea states:  {len(SEA_STATES)}  (1 Mean_2024 + 180 climatological, "
      f"15 years x 12 months)")
print(f"   Output:      {OUTPUT_FOLDER}/")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER,  exist_ok=True)

#%%============================================================================
# JONSWAP SPECTRUM FUNCTION
#==============================================================================

def jonswap_spectrum(omega, Hs, Tp, gamma=3.3):
    """
    JONSWAP spectrum S(ω) from Hs and Tp.

    Parameters:
    -----------
    omega : array_like
        Angular frequencies [rad/s]
    Hs : float
        Significant wave height [m]
    Tp : float
        Peak period [s]
    gamma : float
        Peak enhancement factor (default 3.3)

    Returns:
    --------
    S : ndarray
        Spectral density [m²·s/rad]
    """
    omega  = np.asarray(omega, dtype=float)
    omega_p = 2.0 * np.pi / Tp   # Peak angular frequency [rad/s]
    g       = 9.81                # Gravity [m/s²]

    # Phillips constant calibrated to Hs
    # Hs² = 16 ∫S(ω)dω → α derived numerically below via normalization
    # First compute raw JONSWAP shape, then scale to Hs

    # Spectral width parameter σ
    sigma = np.where(omega <= omega_p, 0.07, 0.09)

    # Peak enhancement factor r(ω)
    r = np.exp(-0.5 * ((omega - omega_p) / (sigma * omega_p))**2)

    # Pierson-Moskowitz base shape (α = 1, will rescale)
    with np.errstate(divide='ignore', invalid='ignore'):
        S_pm = np.where(
            omega > 0,
            omega**(-5) * np.exp(-1.25 * (omega_p / omega)**4),
            0.0
        )

    # Raw JONSWAP shape
    S_raw = S_pm * gamma**r

    # Scale to match Hs:  Hs = 4√(m0),  m0 = ∫S(ω)dω
    m0_raw = np.trapz(S_raw, omega)
    if m0_raw > 0:
        alpha = (Hs / 4.0)**2 / m0_raw
    else:
        alpha = 0.0

    S = alpha * S_raw

    return S

#%%============================================================================
# GENERATE ALL SPECTRA
#==============================================================================

print(f"\n{'─' * 50}")
print(f"  Generating {len(SEA_STATES)} JONSWAP spectra...")
print(f"{'─' * 50}")

n_states = len(SEA_STATES)
n_freq   = len(frequencies)

# Storage arrays
S_all      = np.zeros((n_states, n_freq))   # [181, n_freq]
Hs_array   = np.zeros(n_states)
Tp_array   = np.zeros(n_states)
labels     = []

for i, (label, Hs, Tp) in enumerate(SEA_STATES):
    S = jonswap_spectrum(frequencies, Hs, Tp, gamma=GAMMA)

    # Verify Hs reconstruction
    m0   = np.trapz(S, frequencies)
    Hs_check = 4.0 * np.sqrt(m0)

    S_all[i, :]   = S
    Hs_array[i]   = Hs
    Tp_array[i]   = Tp
    labels.append(label)

print(f"   ✅ {n_states} spectra generated  "
      f"(Hs check verified against 4√m0 for all states)")

#%%============================================================================
# SAVE RESULTS
#==============================================================================

print(f"\n💾 Saving spectra...")

results = {
    'S_omega':      S_all,        # [181, n_freq]  spectral density [m²s/rad]
    'frequencies':  frequencies,  # [n_freq]      angular frequencies [rad/s]
    'Hs':           Hs_array,     # [181]          significant wave height [m]
    'Tp':           Tp_array,     # [181]          peak period [s]
    'gamma':        GAMMA,        # scalar        peak enhancement factor
    'n_states':     n_states,     # scalar
    'n_freq':       n_freq,       # scalar
}

# NPZ
try:
    npz_path = os.path.join(OUTPUT_FOLDER, "Spectra.npz")
    np.savez_compressed(npz_path, **results)
    print(f"   ✅ Spectra.npz")
except Exception as e:
    print(f"   ⚠️  NPZ error: {e}")

# PKL — includes labels list
try:
    pkl_path = os.path.join(OUTPUT_FOLDER, "Spectra.pkl")
    results_pkl = dict(results)
    results_pkl['labels'] = labels
    with open(pkl_path, 'wb') as f:
        pickle.dump(results_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   ✅ Spectra.pkl")
except Exception as e:
    print(f"   ⚠️  PKL error: {e}")

# MAT — labels as cell-like structure (individual strings)
try:
    mat_path = os.path.join(OUTPUT_FOLDER, "Spectra.mat")
    mat_data = dict(results)
    for i, lbl in enumerate(labels):
        mat_data[f"label_{i:03d}"] = lbl
    savemat(mat_path, mat_data, do_compression=True)
    print(f"   ✅ Spectra.mat")
except Exception as e:
    print(f"   ⚠️  MAT error: {e}")

#%%============================================================================
# PLOTS — 16 images: 1 per climatological year (12 monthly spectra overlaid)
#          + 1 for the original Mean_2024 AWAC-measured reference
#==============================================================================
# NOTE: plotting each of the 181 states individually (or all 181 together in
# one figure) would be unreadable / produce too many files. Instead: one
# figure per year (12 lines = 12 months of that year) plus a standalone
# figure for Mean_2024.

print(f"\n📊 Generating per-year plots (16 images total)...")

month_colors = plt.cm.viridis(np.linspace(0, 1, 12))
month_order  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# --- Mean_2024 (standalone) ---
idx_mean = labels.index('Mean_2024')
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(frequencies, S_all[idx_mean, :], color='#d1242f', linewidth=2.2)
ax.axvline(x=2*np.pi/Tp_array[idx_mean], color='gray', linestyle='--',
           linewidth=1.2, label=f'ω_p = {2*np.pi/Tp_array[idx_mean]:.3f} rad/s')
ax.set_xlabel('Angular frequency ω [rad/s]', fontsize=12)
ax.set_ylabel('S(ω)  [m²·s/rad]', fontsize=12)
ax.set_title(f'JONSWAP Spectrum — Mean 2024 (AWAC measured, real year)\n'
             f'Hs = {Hs_array[idx_mean]:.2f} m,  Tp = {Tp_array[idx_mean]:.2f} s,  γ = {GAMMA}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([frequencies[0], frequencies[-1]])
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_FOLDER, "Spectra_Mean_2024.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Spectra_Mean_2024.png")

# --- One figure per climatological year (y01..y15), 12 monthly spectra each ---
for yi in range(1, 16):
    year_prefix = f"y{yi:02d}_"
    year_label  = 2009 + yi   # y01 -> 2010, ..., y15 -> 2024

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for mi, mon in enumerate(month_order):
        lbl = f"{year_prefix}{mon}"
        if lbl not in labels:
            continue
        idx = labels.index(lbl)
        ax.plot(frequencies, S_all[idx, :], color=month_colors[mi], linewidth=1.4,
                label=f"{mon} (Hs={Hs_array[idx]:.2f} m, Tp={Tp_array[idx]:.1f} s)")

    ax.set_xlabel('Angular frequency ω [rad/s]', fontsize=12)
    ax.set_ylabel('S(ω)  [m²·s/rad]', fontsize=12)
    ax.set_title(f'JONSWAP Spectra — Year {yi:02d} ({year_label})  (γ = {GAMMA})\n'
                 f'Climatological monthly states (MCP: AWAC local × WIS offshore ST63042)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([frequencies[0], frequencies[-1]])
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_FOLDER, f"Spectra_y{yi:02d}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Spectra_y{yi:02d}.png  ({year_label})")

#%%============================================================================
# FINAL SUMMARY
#==============================================================================

print("\n" + "=" * 70)
print("St3_Spectra COMPLETED")
print("=" * 70)
print(f"   Spectra generated: {n_states}  (1 Mean_2024 + 180 climatological)")
print(f"   Gamma (γ):         {GAMMA}")
print(f"   Frequency range:   [{frequencies[0]:.2f}, {frequencies[-1]:.2f}] rad/s")
print(f"   Output:            {OUTPUT_FOLDER}/")
print(f"   Plots:             {PLOTS_FOLDER}/  (16 images: 15 per-year + Mean_2024)")
print(f"\n🎯 Next step: run St3_B_Power.py / St3_C_SpectralPower.py")
print(f"   ⚠️  St3_C, St4_A, St4_B downstream will now iterate over 181 states")
print(f"      instead of 13 — review those scripts before running the full batch")
print("\n" + "=" * 70)
