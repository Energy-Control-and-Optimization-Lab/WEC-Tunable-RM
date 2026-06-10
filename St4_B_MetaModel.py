"""
St4_B_MetaModel.py - Enhanced DOE Analysis with Quadratic Response Surface
Performs complete statistical analysis of DOE results with Pareto Charts

Author: Pablo Antonio Matamala Carvajal  
Date: 2025-01-21
Updated: 2025-12-01 - Added quality indicators for R² and MAPE
Description:
- Loads enhanced DOE results from St4_A_ResultsVector (20 response vectors)
- Fits quadratic polynomial metamodel for each response variable
- Performs classical DOE validation (Pure Error, Lack of Fit)
- Calculates standardized effects and generates Pareto Charts
- Generates streamlined plots (2 plots: Actual vs Predicted + Pareto Chart)
- Saves enhanced metamodels with plot data for MATLAB replication
- Cross-response parameter importance analysis
- ADDED: Visual quality indicators (🟢🟡🔴) for R² and MAPE
"""

import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set non-interactive backend for plots
import matplotlib
matplotlib.use('Agg')

#%%============================================================================
# CONFIGURATION
#==============================================================================

# Input configuration
VECTOR_DATA_FILE = "EcoData/St4_ResultVector/VectorValues.pkl"

# Output configuration
OUTPUT_FOLDER = "EcoData/St4_MetaModel/Analysis"
PLOTS_FOLDER = "EcoData/St4_MetaModel/Plots"

# Analysis configuration
ALPHA = 0.05  # Significance level for statistical tests
CONFIDENCE_LEVEL = 0.95  # Confidence level for intervals

# Pareto Chart configuration
N_TOP_EFFECTS = 12  # Number of top effects to show in Pareto Chart
PARETO_COLORS = {
    'main': 'steelblue',
    'interaction': 'orange', 
    'quadratic': 'green'
}

# Parameter mapping (like scientific papers)
PARAMETER_MAPPING = {
    'D1': 'A',  # Float diameter
    'D2': 'B',  # Float draft  
    'D3': 'C',  # Float angle
    'D4': 'D',  # Spar draft
    'D5': 'E',  # Spar diameter
    'D6': 'F'   # PTO damping
}

# Quality assessment thresholds
QUALITY_THRESHOLDS = {
    'r2': {
        'excellent': 0.90,  # R² ≥ 0.90 = 🟢
        'good': 0.75        # R² ≥ 0.75 = 🟡, < 0.75 = 🔴
    },
    'mape': {
        'excellent': 5.0,   # MAPE ≤ 5% = 🟢
        'good': 15.0        # MAPE ≤ 15% = 🟡, > 15% = 🔴
    }
}

#%%============================================================================
# QUALITY ASSESSMENT FUNCTIONS
#==============================================================================

def assess_r2_quality(r2):
    """Assess R² quality with color indicators"""
    if r2 >= QUALITY_THRESHOLDS['r2']['excellent']:
        return '🟢'
    elif r2 >= QUALITY_THRESHOLDS['r2']['good']:
        return '🟡'
    else:
        return '🔴'

def assess_mape_quality(mape):
    """Assess MAPE quality with color indicators"""
    if mape <= QUALITY_THRESHOLDS['mape']['excellent']:
        return '🟢'
    elif mape <= QUALITY_THRESHOLDS['mape']['good']:
        return '🟡'
    else:
        return '🔴'

def assess_model_quality(mape, r2):
    """Assess overall model quality (worst of the two metrics)"""
    r2_quality = assess_r2_quality(r2)
    mape_quality = assess_mape_quality(mape)
    
    # Return the worst quality (red > yellow > green)
    if r2_quality == '🔴' or mape_quality == '🔴':
        return '🔴'
    elif r2_quality == '🟡' or mape_quality == '🟡':
        return '🟡'
    else:
        return '🟢'

#%%============================================================================
# LOAD DOE RESULTS
#==============================================================================

print("="*70)
print("ENHANCED DOE ANALYSIS - QUADRATIC RESPONSE SURFACE")
print("="*70)

print(f"\n📂 Loading enhanced DOE results from: {VECTOR_DATA_FILE}")

try:
    with open(VECTOR_DATA_FILE, 'rb') as f:
        vector_data = pickle.load(f)
    
    # Extract data
    design_matrix = vector_data['design_matrix']  # [n_experiments × 6]
    parameter_names = vector_data['parameter_names']  # ['D1', 'D2', ..., 'D6']
    response_names = vector_data['response_names']    # ['P_avg', 'P_at_1_25', ..., 'phase_at_*', 'P_efficiency']
    experiment_ids = vector_data['experiment_ids']
    n_experiments = vector_data['n_experiments']
    
    print(f"✅ Enhanced data loaded successfully!")
    print(f"   Experiments: {n_experiments}")
    print(f"   Parameters: {len(parameter_names)} {parameter_names}")
    print(f"   Response vectors: {len(response_names)}")
    print(f"   Enhancement features: Phase vectors + Efficiency")
    
except FileNotFoundError:
    print(f"❌ ERROR: Vector data file not found: {VECTOR_DATA_FILE}")
    print(f"   Please run St4_A_ResultsVector.py first")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR loading vector data: {e}")
    sys.exit(1)

#%%============================================================================
# SETUP OUTPUT FOLDERS
#==============================================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)
print(f"\n📁 Enhanced output folders created:")
print(f"   Analysis: {OUTPUT_FOLDER}/")
print(f"   Plots: {PLOTS_FOLDER}/")

#%%============================================================================
# IDENTIFY FACTORIAL vs CENTER POINTS
#==============================================================================

print(f"\n{'='*70}")
print("STEP 1: IDENTIFY FACTORIAL vs CENTER POINTS")
print(f"{'='*70}")

# Last 6 experiments are center points (identical)
n_factorial = n_experiments - 6
n_center = 6

factorial_indices = np.arange(n_factorial)
center_indices = np.arange(n_factorial, n_experiments)

print(f"📊 DOE Structure:")
print(f"   Factorial points: {n_factorial} (experiments 1-{n_factorial})")
print(f"   Center points: {n_center} (experiments {n_factorial+1}-{n_experiments})")

# Verify center points are identical
X_factorial = design_matrix[factorial_indices]
X_center = design_matrix[center_indices]

center_unique = np.unique(X_center, axis=0)
if len(center_unique) == 1:
    print(f"✅ Center points verified: All {n_center} are identical")
    center_point_values = center_unique[0]
    print(f"   Center point: {dict(zip(parameter_names, center_point_values))}")
else:
    print(f"⚠️  WARNING: Center points are not identical!")

#%%============================================================================
# POLYNOMIAL FEATURE ENGINEERING
#==============================================================================

print(f"\n{'='*70}")
print("STEP 2: POLYNOMIAL FEATURE ENGINEERING")
print(f"{'='*70}")

# Create quadratic features (linear + interactions + quadratic)
poly_features = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly_features.fit_transform(design_matrix)

feature_names = poly_features.get_feature_names_out(parameter_names)
n_features = len(feature_names)
n_effects = n_features - 1  # Exclude intercept for effects analysis

print(f"📈 Quadratic Model Features:")
print(f"   Input parameters: {len(parameter_names)}")
print(f"   Total features: {n_features} (including intercept)")
print(f"   Effects for analysis: {n_effects} (excluding intercept)")
print(f"   Feature breakdown:")
print(f"      - Intercept (β₀): 1")
print(f"      - Linear terms: {len(parameter_names)}")
print(f"      - Interaction terms: {int(len(parameter_names)*(len(parameter_names)-1)/2)}")
print(f"      - Quadratic terms: {len(parameter_names)}")

# Print first few feature names for verification
print(f"\n   First 10 features: {feature_names[:10]}")

#%%============================================================================
# STANDARDIZED EFFECTS CALCULATION FUNCTIONS
#==============================================================================

def calculate_standardized_effects(coefficients, X_poly, feature_names, parameter_mapping):
    """
    Calculate standardized effects for Pareto Chart analysis
    
    Parameters:
    -----------
    coefficients : array
        Model coefficients (excluding intercept)
    X_poly : array  
        Polynomial feature matrix
    feature_names : array
        Feature names from PolynomialFeatures
    parameter_mapping : dict
        Mapping from D1-D6 to A-F
    
    Returns:
    --------
    list : Standardized effects sorted by magnitude
    """
    
    standardized_effects = []
    
    # CRITICAL FIX: skip coefficients[0] (bias term coef) AND feature_names[0]
    # ('1') so that coefficients[k] always aligns with feature_names[k] and
    # X_poly[:, k]. The previous code used zip(coefficients, feature_names[1:])
    # which paired the bias coefficient with D1, D1's coef with D2, etc.,
    # making every effect in the Pareto chart wrong by one position.
    for i, (coef, feature_name) in enumerate(zip(coefficients[1:], feature_names[1:]), start=1):
        
        # Calculate standardized effect = |coef| * std(X_poly[:, i])
        std_effect = abs(coef) * np.std(X_poly[:, i])
        
        # Determine effect type and create readable name
        effect_type = 'main'
        readable_name = feature_name
        
        # Parse feature name to create readable version
        if '^2' in feature_name:
            # Quadratic term: D1^2 -> A²
            param = feature_name.replace('^2', '')
            if param in parameter_mapping:
                readable_name = parameter_mapping[param] + '²'
                effect_type = 'quadratic'
                
        elif ' ' in feature_name:
            # Interaction term: D1 D2 -> AB
            params = feature_name.split(' ')
            if len(params) == 2 and all(p in parameter_mapping for p in params):
                readable_name = parameter_mapping[params[0]] + parameter_mapping[params[1]]
                effect_type = 'interaction'
                
        else:
            # Main effect: D1 -> A
            if feature_name in parameter_mapping:
                readable_name = parameter_mapping[feature_name]
                effect_type = 'main'
        
        standardized_effects.append({
            'term': readable_name,
            'effect': std_effect,
            'type': effect_type,
            'abs_effect': std_effect,
            'original_coef': coef,
            'original_term': feature_name
        })
    
    # Sort by absolute effect (descending)
    standardized_effects.sort(key=lambda x: x['abs_effect'], reverse=True)
    
    return standardized_effects

def create_pareto_chart_with_interactions(ax, standardized_effects, response_name, n_top=12):
    """
    Create enhanced Pareto chart with interactions (horizontal bars like paper)
    """
    
    # Take top N effects
    top_effects = standardized_effects[:n_top]
    
    # Extract data for plotting
    effect_names = [e['term'] for e in top_effects]
    effect_values = [e['abs_effect'] for e in top_effects]
    effect_types = [e['type'] for e in top_effects]
    
    # Create colors based on effect type
    colors = [PARETO_COLORS[effect_type] for effect_type in effect_types]
    
    # Create horizontal bar chart (like in paper)
    y_pos = np.arange(len(effect_names))
    bars = ax.barh(y_pos, effect_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Invert y-axis (largest effect at top)
    ax.invert_yaxis()
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(effect_names, fontsize=10)
    ax.set_xlabel('Standardized Effect', fontsize=12)
    ax.set_title(f'Pareto Chart: {response_name}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add reference table in corner
    table_text = "Parameter Mapping:\n"
    for orig, mapped in PARAMETER_MAPPING.items():
        table_text += f"{mapped} = {orig}  "
        if mapped in ['C', 'F']:  # Line break after C and F
            table_text += "\n"
    
    ax.text(0.02, 0.98, table_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Create legend
    legend_elements = []
    for effect_type, color in PARETO_COLORS.items():
        if effect_type in effect_types:
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, 
                                               label=effect_type.title()))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    return top_effects

#%%============================================================================
# METAMODEL FITTING AND ANALYSIS FUNCTION (ENHANCED)
#==============================================================================

def analyze_response_variable_enhanced(response_name, y_data, X_poly, factorial_indices, center_indices, 
                                     feature_names, parameter_names, output_folder, plots_folder):
    """
    Complete enhanced DOE analysis for a single response variable with Pareto Charts
    """
    
    print(f"\n{'='*50}")
    print(f"ANALYZING ENHANCED RESPONSE: {response_name}")
    print(f"{'='*50}")
    
    # Split data
    y_factorial = y_data[factorial_indices]
    y_center = y_data[center_indices]
    X_factorial_poly = X_poly[factorial_indices]
    X_center_poly = X_poly[center_indices]
    
    # Response statistics
    y_mean = np.mean(y_data)
    y_min = np.min(y_data)
    y_max = np.max(y_data)
    y_range = y_max - y_min
    
    print(f"📊 Response Statistics:")
    print(f"   Mean: {y_mean:.2f}")
    print(f"   Range: [{y_min:.2f}, {y_max:.2f}]")
    print(f"   Span: {y_range:.2f} ({y_range/abs(y_mean)*100:.1f}% of |mean|)")
    
    # ========================================================================
    # STEP 1: CALCULATE PURE ERROR FROM CENTER POINTS
    # ========================================================================
    
    center_mean = np.mean(y_center)
    center_std = np.std(y_center, ddof=1)  # Sample standard deviation
    center_var = center_std ** 2
    
    # Pure error percentage
    pure_error_percent = (center_std / abs(center_mean)) * 100 if center_mean != 0 else 0
    
    print(f"\n🎯 PURE ERROR ANALYSIS:")
    print(f"   Center point responses: {y_center}")
    print(f"   Center mean: {center_mean:.2f}")
    print(f"   Pure error (std): {center_std:.2f}")
    print(f"   Pure error %: ±{pure_error_percent:.2f}%")
    
    # ========================================================================
    # STEP 2: FIT QUADRATIC MODEL USING ALL DATA
    # ========================================================================
    
    # Fit model with all experiments
    # CRITICAL FIX: fit_intercept=False because X_poly already contains the bias
    # column of ones (include_bias=True). Using fit_intercept=True (default) would
    # add a second intercept term, distorting all other coefficients.
    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y_data)
    
    # Predictions
    y_pred = model.predict(X_poly)
    y_pred_factorial = y_pred[factorial_indices]
    y_pred_center = y_pred[center_indices]
    
    # Model statistics
    r2 = r2_score(y_data, y_pred)
    r2_adj = 1 - (1 - r2) * (len(y_data) - 1) / (len(y_data) - X_poly.shape[1] - 1)
    rmse = np.sqrt(mean_squared_error(y_data, y_pred))
    mae = mean_absolute_error(y_data, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_data - y_pred) / np.abs(y_data))) * 100
    
    # Model error percentage
    model_error_percent = (rmse / abs(y_mean)) * 100 if y_mean != 0 else 0
    
    print(f"\n📈 MODEL FIT STATISTICS:")
    print(f"   R²: {r2:.4f}")
    print(f"   R²_adjusted: {r2_adj:.4f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Model error %: ±{model_error_percent:.2f}%")
    
    # ========================================================================
    # STEP 3: LACK OF FIT ANALYSIS
    # ========================================================================
    
    # Residuals
    residuals = y_data - y_pred
    residuals_factorial = residuals[factorial_indices]
    
    # Lack of fit calculation
    model_residual_var = np.var(residuals, ddof=X_poly.shape[1])
    
    # Lack of fit variance = Model variance - Pure error variance
    lack_of_fit_var = model_residual_var - center_var if model_residual_var > center_var else 0
    lack_of_fit_std = np.sqrt(lack_of_fit_var)
    lack_of_fit_percent = (lack_of_fit_std / abs(y_mean)) * 100 if y_mean != 0 else 0
    
    # F-test for lack of fit
    df_lof = len(factorial_indices) - X_poly.shape[1] + 1
    df_pe = len(center_indices) - 1
    
    if center_var > 0 and df_lof > 0 and df_pe > 0:
        f_statistic = lack_of_fit_var / center_var if center_var > 1e-10 else 0
        p_value_lof = 1 - stats.f.cdf(f_statistic, df_lof, df_pe) if f_statistic > 0 else 1
    else:
        f_statistic = np.nan
        p_value_lof = np.nan
    
    # Model adequacy based on practical quality metrics (ignoring LoF test)
    model_adequate = (r2 >= 0.85) and (mape <= 10.0)
    
    print(f"\n🔍 LACK OF FIT ANALYSIS:")
    print(f"   Model residual std: {np.sqrt(model_residual_var):.2f}")
    print(f"   Lack of fit std: {lack_of_fit_std:.2f}")
    print(f"   Lack of fit %: ±{lack_of_fit_percent:.2f}%")
    print(f"   F-statistic: {f_statistic:.2f}")
    print(f"   p-value: {p_value_lof:.4f}")
    print(f"   Model adequate: {'✅ YES' if model_adequate else '❌ NO'} (R² ≥ 0.85 AND MAPE ≤ 10%)")
    
    # ========================================================================
    # STEP 4: STANDARDIZED EFFECTS ANALYSIS (NEW)
    # ========================================================================
    
    print(f"\n⚡ STANDARDIZED EFFECTS ANALYSIS:")
    
    # Calculate standardized effects
    standardized_effects = calculate_standardized_effects(
        model.coef_, X_poly, feature_names, PARAMETER_MAPPING
    )
    
    # Show top effects
    print(f"   Top {min(10, len(standardized_effects))} standardized effects:")
    for i, effect in enumerate(standardized_effects[:10], 1):
        print(f"   {i:2d}. {effect['term']:>4s}: {effect['abs_effect']:6.2f} ({effect['type']})")
    
    # Get top effect for summary
    top_effect = standardized_effects[0]['term'] if standardized_effects else 'None'
    
    # ========================================================================
    # STEP 5: GENERATE ENHANCED PLOTS (2 PLOTS ONLY)
    # ========================================================================
    
    # Create enhanced plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Enhanced DOE Analysis: {response_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Actual vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_data, y_pred, alpha=0.7, color='blue', s=50, edgecolors='black', linewidth=0.5)
    ax1.plot([y_min, y_max], [y_min, y_max], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Values', fontsize=12)
    ax1.set_ylabel('Predicted Values', fontsize=12)
    ax1.set_title(f'Actual vs Predicted\nR² = {r2:.3f}, MAPE = {mape:.1f}%', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Pareto Chart (NEW)
    ax2 = axes[1]
    top_effects_data = create_pareto_chart_with_interactions(
        ax2, standardized_effects, response_name, n_top=N_TOP_EFFECTS
    )
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plots_folder, f"{response_name}_enhanced_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Enhanced plot saved: {response_name}_enhanced_analysis.png")
    plt.close()
    
    # ========================================================================
    # STEP 6: PREPARE ENHANCED RESULTS DICTIONARY
    # ========================================================================
    
    results = {
        # Response data
        'response_name': response_name,
        'y_data': y_data,
        'y_mean': y_mean,
        'y_range': [y_min, y_max],
        
        # Model
        'model': model,
        # With fit_intercept=False, sklearn sets intercept_=0.0.
        # The actual intercept is model.coef_[0] (coefficient for the bias column).
        'coefficients': model.coef_,          # [28] — coef_[0] is the intercept
        'intercept': model.coef_[0],          # scalar — true model intercept
        'feature_names': feature_names,
        'polynomial_features': poly_features,
        
        # Predictions
        'y_pred': y_pred,
        'residuals': residuals,
        
        # Model statistics
        'r2': r2,
        'r2_adjusted': r2_adj,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'model_error_percent': model_error_percent,
        
        # Pure error analysis
        'center_responses': y_center,
        'center_mean': center_mean,
        'pure_error_std': center_std,
        'pure_error_percent': pure_error_percent,
        
        # Lack of fit analysis
        'lack_of_fit_std': lack_of_fit_std,
        'lack_of_fit_percent': lack_of_fit_percent,
        'f_statistic': f_statistic,
        'p_value_lof': p_value_lof,
        'model_adequate': model_adequate,
        
        # Enhanced effects analysis (NEW)
        'standardized_effects': standardized_effects,
        'top_effect': top_effect,
        'n_effects_analyzed': len(standardized_effects),
        
        # Plot data for MATLAB replication (NEW)
        'plot_data': {
            'actual_vs_predicted': {
                'y_actual': y_data,
                'y_predicted': y_pred,
                'r2': r2,
                'mape': mape,
                'title': f'Actual vs Predicted - {response_name}'
            },
            'pareto_chart': {
                'effect_names': [e['term'] for e in top_effects_data],
                'effect_values': [e['abs_effect'] for e in top_effects_data],
                'effect_types': [e['type'] for e in top_effects_data],
                'title': f'Pareto Chart - {response_name}'
            }
        },
        
        # Metadata
        'n_experiments': len(y_data),
        'n_factorial': len(factorial_indices),
        'n_center': len(center_indices),
        'confidence_level': CONFIDENCE_LEVEL,
        'alpha': ALPHA,
        'enhancement_version': '2025-11-26'
    }
    
    return results

#%%============================================================================
# ANALYZE ALL RESPONSE VARIABLES (ENHANCED)
#==============================================================================

print(f"\n{'='*70}")
print("STEP 3: ANALYZE ALL RESPONSE VARIABLES (ENHANCED)")
print(f"{'='*70}")

# Storage for enhanced results
all_enhanced_results = {}
enhanced_summary_table = []
parameter_importance_counter = {}

# Analyze each response variable
for response_name in response_names:
    
    # Get response data
    if response_name in vector_data:
        y_data = vector_data[response_name]
        
        # Perform enhanced analysis
        results = analyze_response_variable_enhanced(
            response_name=response_name,
            y_data=y_data,
            X_poly=X_poly,
            factorial_indices=factorial_indices,
            center_indices=center_indices,
            feature_names=feature_names,
            parameter_names=parameter_names,
            output_folder=OUTPUT_FOLDER,
            plots_folder=PLOTS_FOLDER
        )
        
        # Store enhanced results
        all_enhanced_results[response_name] = results
        
        # Count parameter importance across responses
        top_param = results['top_effect']
        if top_param in parameter_importance_counter:
            parameter_importance_counter[top_param] += 1
        else:
            parameter_importance_counter[top_param] = 1
        
        # Add to enhanced summary table (WITH QUALITY INDICATORS)
        enhanced_summary_table.append({
            'Response': f"{assess_model_quality(results['mape'], results['r2'])} {response_name}",
            'R²': results['r2'],
            'R²_Q': assess_r2_quality(results['r2']),
            'R²_adj': results['r2_adjusted'],
            'MAPE (%)': results['mape'],
            'MAPE_Q': assess_mape_quality(results['mape']),
            'Pure Error (%)': results['pure_error_percent'],
            'Lack of Fit (%)': results['lack_of_fit_percent'],
            'Model Adequate': '✅' if results['model_adequate'] else '❌',
            'Top Effect': results['top_effect'],
            'N Effects': results['n_effects_analyzed']
        })
        
    else:
        print(f"⚠️  Response '{response_name}' not found in data")

#%%============================================================================
# ENHANCED SUMMARY REPORT WITH QUALITY INDICATORS
#==============================================================================

print(f"\n{'='*70}")
print("STEP 4: ENHANCED SUMMARY REPORT WITH QUALITY INDICATORS")
print(f"{'='*70}")

print(f"\n📊 ENHANCED METAMODEL QUALITY SUMMARY:")
print(f"{'Response':<20} {'R²':<8} {'R²_Q':<5} {'MAPE(%)':<8} {'MAPE_Q':<7} {'Pure_E(%)':<10} {'LoF(%)':<8} {'Adequate':<10} {'Top_Effect':<12}")
print("-" * 130)

for summary in enhanced_summary_table:
    print(f"{summary['Response']:<20} "
          f"{summary['R²']:<8.3f} "
          f"{summary['R²_Q']:<5} "
          f"{summary['MAPE (%)']:<8.1f} "
          f"{summary['MAPE_Q']:<7} "
          f"{summary['Pure Error (%)']:<10.2f} "
          f"{summary['Lack of Fit (%)']:<8.2f} "
          f"{summary['Model Adequate']:<10} "
          f"{summary['Top Effect']:<12}")

# Enhanced quality assessment with color categories
green_models = sum(1 for s in enhanced_summary_table if '🟢' in s['Response'])
yellow_models = sum(1 for s in enhanced_summary_table if '🟡' in s['Response'])
red_models = sum(1 for s in enhanced_summary_table if '🔴' in s['Response'])

print(f"\n🎯 ENHANCED OVERALL ASSESSMENT:")
print(f"   🟢 Excellent models (MAPE ≤ 5% AND R² ≥ 0.90): {green_models}/{len(enhanced_summary_table)}")
print(f"   🟡 Good models (MAPE ≤ 15% AND R² ≥ 0.75): {yellow_models}/{len(enhanced_summary_table)}")
print(f"   🔴 Poor models (MAPE > 15% OR R² < 0.75): {red_models}/{len(enhanced_summary_table)}")

adequate_models = sum(1 for s in enhanced_summary_table if s['Model Adequate'] == '✅')
print(f"   Statistically adequate models: {adequate_models}/{len(enhanced_summary_table)} (R² ≥ 0.85 AND MAPE ≤ 10%)")

print(f"\n🎨 QUALITY INDICATOR LEGEND:")
print(f"   🟢 Excellent: MAPE ≤ {QUALITY_THRESHOLDS['mape']['excellent']}% AND R² ≥ {QUALITY_THRESHOLDS['r2']['excellent']}")
print(f"   🟡 Good: MAPE ≤ {QUALITY_THRESHOLDS['mape']['good']}% AND R² ≥ {QUALITY_THRESHOLDS['r2']['good']}")
print(f"   🔴 Poor: MAPE > {QUALITY_THRESHOLDS['mape']['good']}% OR R² < {QUALITY_THRESHOLDS['r2']['good']}")
print(f"   ✅ Adequate: R² ≥ 0.85 AND MAPE ≤ 10% (practical quality threshold)")
print(f"   ❌ Not Adequate: R² < 0.85 OR MAPE > 10%")

# Cross-response parameter importance analysis (NEW)
print(f"\n⚡ PARAMETER IMPORTANCE ACROSS ALL RESPONSES:")
sorted_params = sorted(parameter_importance_counter.items(), key=lambda x: x[1], reverse=True)
for param, count in sorted_params:
    percentage = (count / len(enhanced_summary_table)) * 100
    print(f"   {param}: {count}/{len(enhanced_summary_table)} responses ({percentage:.1f}%)")

#%%============================================================================
# SAVE ENHANCED RESULTS
#==============================================================================

print(f"\n{'='*70}")
print("STEP 5: SAVE ENHANCED RESULTS")
print(f"{'='*70}")

# Save complete enhanced results
enhanced_results_file = os.path.join(OUTPUT_FOLDER, "enhanced_metamodel_results.pkl")
enhanced_data = {
    'all_results': all_enhanced_results,
    'summary_table': enhanced_summary_table,
    'design_matrix': design_matrix,
    'parameter_names': parameter_names,
    'response_names': response_names,
    'polynomial_features': poly_features,
    'feature_names': feature_names,
    'factorial_indices': factorial_indices,
    'center_indices': center_indices,
    'parameter_mapping': PARAMETER_MAPPING,
    'parameter_importance': parameter_importance_counter,
    'quality_thresholds': QUALITY_THRESHOLDS,
    'quality_counts': {
        'excellent': green_models,
        'good': yellow_models,
        'poor': red_models
    },
    'metadata': {
        'analysis_type': 'Enhanced DOE with Quadratic Response Surface + Pareto Charts + Quality Indicators',
        'enhancement_date': '2025-12-01',
        'confidence_level': CONFIDENCE_LEVEL,
        'alpha': ALPHA,
        'n_experiments': n_experiments,
        'n_factorial': n_factorial,
        'n_center': n_center,
        'total_features': n_features,
        'n_effects_analyzed': n_effects,
        'n_response_vectors': len(response_names),
        'pareto_config': {
            'n_top_effects': N_TOP_EFFECTS,
            'colors': PARETO_COLORS
        }
    }
}

with open(enhanced_results_file, 'wb') as f:
    pickle.dump(enhanced_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"✅ Enhanced results saved: enhanced_metamodel_results.pkl")

# Save enhanced results as MAT file with plot data (MATLAB compatible)
try:
    from scipy.io import savemat
    
    # Prepare MATLAB-compatible data structure
    enhanced_mat_data = {
        'design_matrix': design_matrix,
        'parameter_names': np.array(parameter_names, dtype=object),
        'response_names': np.array(response_names, dtype=object),
        'factorial_indices': factorial_indices + 1,  # MATLAB indexing (1-based)
        'center_indices': center_indices + 1,        # MATLAB indexing (1-based)
        'n_experiments': n_experiments,
        'n_factorial': n_factorial,
        'n_center': n_center,
        'feature_names': np.array(feature_names, dtype=object),
        'confidence_level': CONFIDENCE_LEVEL,
        'alpha': ALPHA,
        'parameter_mapping_keys': np.array(list(PARAMETER_MAPPING.keys()), dtype=object),
        'parameter_mapping_values': np.array(list(PARAMETER_MAPPING.values()), dtype=object)
    }
    
    # Add enhanced analysis for each response
    for response_name, results in all_enhanced_results.items():
        # Clean response name for MATLAB
        clean_name = response_name.replace('.', '_').replace('-', '_')
        
        # Response data and predictions
        enhanced_mat_data[f'{clean_name}_actual'] = results['y_data']
        enhanced_mat_data[f'{clean_name}_predicted'] = results['y_pred']
        enhanced_mat_data[f'{clean_name}_residuals'] = results['residuals']
        
        # Model coefficients
        enhanced_mat_data[f'{clean_name}_coefficients'] = results['coefficients']
        enhanced_mat_data[f'{clean_name}_intercept'] = results['intercept']
        
        # Enhanced statistics
        enhanced_mat_data[f'{clean_name}_R2'] = results['r2']
        enhanced_mat_data[f'{clean_name}_R2_adj'] = results['r2_adjusted']
        enhanced_mat_data[f'{clean_name}_RMSE'] = results['rmse']
        enhanced_mat_data[f'{clean_name}_MAE'] = results['mae']
        enhanced_mat_data[f'{clean_name}_MAPE'] = results['mape']
        enhanced_mat_data[f'{clean_name}_model_error_percent'] = results['model_error_percent']
        
        # Pure error analysis
        enhanced_mat_data[f'{clean_name}_center_responses'] = results['center_responses']
        enhanced_mat_data[f'{clean_name}_center_mean'] = results['center_mean']
        enhanced_mat_data[f'{clean_name}_pure_error_std'] = results['pure_error_std']
        enhanced_mat_data[f'{clean_name}_pure_error_percent'] = results['pure_error_percent']
        
        # Lack of fit analysis
        enhanced_mat_data[f'{clean_name}_lack_of_fit_std'] = results['lack_of_fit_std']
        enhanced_mat_data[f'{clean_name}_lack_of_fit_percent'] = results['lack_of_fit_percent']
        enhanced_mat_data[f'{clean_name}_f_statistic'] = results['f_statistic'] if not np.isnan(results['f_statistic']) else 0
        enhanced_mat_data[f'{clean_name}_p_value_lof'] = results['p_value_lof'] if not np.isnan(results['p_value_lof']) else 1
        enhanced_mat_data[f'{clean_name}_model_adequate'] = int(results['model_adequate'])
        
        # Enhanced effects analysis (NEW)
        enhanced_mat_data[f'{clean_name}_top_effect'] = results['top_effect']
        
        # Standardized effects (top 20)
        top_20_effects = results['standardized_effects'][:20]
        enhanced_mat_data[f'{clean_name}_effect_names'] = np.array([e['term'] for e in top_20_effects], dtype=object)
        enhanced_mat_data[f'{clean_name}_effect_values'] = np.array([e['abs_effect'] for e in top_20_effects])
        enhanced_mat_data[f'{clean_name}_effect_types'] = np.array([e['type'] for e in top_20_effects], dtype=object)
        
        # Plot data for MATLAB replication (NEW)
        plot_data = results['plot_data']
        
        # Actual vs Predicted plot data
        enhanced_mat_data[f'{clean_name}_plot_actual'] = plot_data['actual_vs_predicted']['y_actual']
        enhanced_mat_data[f'{clean_name}_plot_predicted'] = plot_data['actual_vs_predicted']['y_predicted']
        enhanced_mat_data[f'{clean_name}_plot_R2'] = plot_data['actual_vs_predicted']['r2']
        enhanced_mat_data[f'{clean_name}_plot_MAPE'] = plot_data['actual_vs_predicted']['mape']
        
        # Pareto chart plot data
        enhanced_mat_data[f'{clean_name}_pareto_names'] = np.array(plot_data['pareto_chart']['effect_names'], dtype=object)
        enhanced_mat_data[f'{clean_name}_pareto_values'] = np.array(plot_data['pareto_chart']['effect_values'])
        enhanced_mat_data[f'{clean_name}_pareto_types'] = np.array(plot_data['pareto_chart']['effect_types'], dtype=object)
        
        # Color indices for MATLAB (1=steelblue, 2=orange, 3=green)
        color_map = {'main': 1, 'interaction': 2, 'quadratic': 3}
        color_indices = [color_map.get(t, 1) for t in plot_data['pareto_chart']['effect_types']]
        enhanced_mat_data[f'{clean_name}_pareto_colors'] = np.array(color_indices)
    
    # Save enhanced MAT file
    enhanced_mat_file = os.path.join(OUTPUT_FOLDER, "enhanced_metamodel_results.mat")
    savemat(enhanced_mat_file, enhanced_mat_data, do_compression=True)
    print(f"✅ Enhanced MATLAB results saved: enhanced_metamodel_results.mat")
    print(f"   📊 Includes plot data for MATLAB replication")
    
except ImportError:
    print(f"⚠️  scipy not available - Enhanced MAT file not saved")
except Exception as e:
    print(f"⚠️  Error saving enhanced MAT file: {e}")

# Save enhanced summary table as CSV
try:
    import pandas as pd
    enhanced_df_summary = pd.DataFrame(enhanced_summary_table)
    enhanced_csv_file = os.path.join(OUTPUT_FOLDER, "enhanced_metamodel_summary.csv")
    enhanced_df_summary.to_csv(enhanced_csv_file, index=False)
    print(f"✅ Enhanced summary table saved: enhanced_metamodel_summary.csv")
except ImportError:
    print(f"⚠️  pandas not available - Enhanced CSV not saved")

# Save individual enhanced metamodels
enhanced_metamodels_folder = os.path.join(OUTPUT_FOLDER, "individual_enhanced_metamodels")
os.makedirs(enhanced_metamodels_folder, exist_ok=True)

for response_name, results in all_enhanced_results.items():
    # Save Python enhanced metamodel
    enhanced_metamodel_file = os.path.join(enhanced_metamodels_folder, f"{response_name}_enhanced_metamodel.pkl")
    
    enhanced_metamodel_data = {
        'model': results['model'],
        'polynomial_features': results['polynomial_features'],
        'parameter_names': parameter_names,
        'response_name': response_name,
        'parameter_mapping': PARAMETER_MAPPING,
        'standardized_effects': results['standardized_effects'],
        'plot_data': results['plot_data'],  # NEW: For MATLAB replication
        'statistics': {
            'r2': results['r2'],
            'r2_adjusted': results['r2_adjusted'],
            'mape': results['mape'],
            'model_adequate': results['model_adequate'],
            'top_effect': results['top_effect'],
            'pure_error_percent': results['pure_error_percent'],
            'lack_of_fit_percent': results['lack_of_fit_percent']
        },
        'enhancement_info': {
            'version': '2025-12-01',
            'features': ['standardized_effects', 'pareto_charts', 'plot_data_matlab', 'quality_indicators']
        }
    }
    
    with open(enhanced_metamodel_file, 'wb') as f:
        pickle.dump(enhanced_metamodel_data, f)
    
    # Save enhanced MAT metamodel
    try:
        clean_name = response_name.replace('.', '_').replace('-', '_')
        enhanced_mat_metamodel_file = os.path.join(enhanced_metamodels_folder, f"{clean_name}_enhanced_metamodel.mat")
        
        enhanced_mat_metamodel_data = {
            'coefficients': results['model'].coef_,
            'intercept': results['model'].coef_[0],   # true intercept = coef_[0]
            'parameter_names': np.array(parameter_names, dtype=object),
            'feature_names': np.array(results['feature_names'], dtype=object),
            'response_name': response_name,
            'R2': results['r2'],
            'R2_adjusted': results['r2_adjusted'],
            'MAPE': results['mape'],
            'model_adequate': int(results['model_adequate']),
            'top_effect': results['top_effect'],
            'polynomial_degree': 2,
            'include_bias': 1,
            
            # Enhanced plot data
            'plot_actual': results['plot_data']['actual_vs_predicted']['y_actual'],
            'plot_predicted': results['plot_data']['actual_vs_predicted']['y_predicted'],
            'plot_R2': results['plot_data']['actual_vs_predicted']['r2'],
            'plot_MAPE': results['plot_data']['actual_vs_predicted']['mape'],
            
            'pareto_names': np.array(results['plot_data']['pareto_chart']['effect_names'], dtype=object),
            'pareto_values': np.array(results['plot_data']['pareto_chart']['effect_values']),
            'pareto_types': np.array(results['plot_data']['pareto_chart']['effect_types'], dtype=object),
            
            # Parameter mapping
            'param_mapping_keys': np.array(list(PARAMETER_MAPPING.keys()), dtype=object),
            'param_mapping_values': np.array(list(PARAMETER_MAPPING.values()), dtype=object),
            
            'instructions': np.array([
                'To predict: X_poly = generate_polynomial_features(X, degree=2)',
                'Then: y_pred = coefficients * X_poly + intercept',
                'Plot data included for replication in MATLAB'
            ], dtype=object)
        }
        
        savemat(enhanced_mat_metamodel_file, enhanced_mat_metamodel_data, do_compression=True)
        
    except Exception as e:
        print(f"⚠️  Error saving enhanced MAT metamodel for {response_name}: {e}")

print(f"✅ Individual enhanced metamodels saved in: individual_enhanced_metamodels/")
print(f"   📊 {len(all_enhanced_results)} Python .pkl + {len(all_enhanced_results)} MATLAB .mat files")

#%%============================================================================
# FINAL ENHANCED SUMMARY
#==============================================================================

print(f"\n{'='*70}")
print("🎉 ENHANCED DOE ANALYSIS WITH QUALITY INDICATORS COMPLETED")
print(f"{'='*70}")

print(f"\n📁 Enhanced results saved in: {OUTPUT_FOLDER}/")
print(f"   📊 enhanced_metamodel_results.pkl (complete enhanced analysis)")
print(f"   📊 enhanced_metamodel_results.mat (MATLAB + plot data)")
print(f"   📋 enhanced_metamodel_summary.csv (summary table)")
print(f"   🤖 individual_enhanced_metamodels/ (Python + MATLAB)")

print(f"\n📈 Enhanced plots saved in: {PLOTS_FOLDER}/")
print(f"   📊 {len(response_names)} files: *_enhanced_analysis.png")
print(f"   📊 Each plot: 2 panels (Actual vs Predicted + Pareto Chart)")

print(f"\n🎯 Enhanced Key Findings:")
print(f"   📊 Analyzed {len(response_names)} enhanced response vectors")
print(f"   🟢 {green_models} excellent metamodels (MAPE ≤ 5% AND R² ≥ 0.90)")
print(f"   🟡 {yellow_models} good metamodels (MAPE ≤ 15% AND R² ≥ 0.75)")
print(f"   🔴 {red_models} poor metamodels (MAPE > 15% OR R² < 0.75)")
print(f"   ✅ {adequate_models} statistically adequate models")
print(f"   📈 Quadratic models with {n_features} features each")
print(f"   ⚡ Standardized effects analysis with Pareto Charts")

print(f"\n🆕 NEW ENHANCEMENTS:")
print(f"   ⚡ Pareto Charts with {n_effects} standardized effects")
print(f"   🎨 Streamlined analysis (2 plots vs 4 plots)")
print(f"   📊 Parameter mapping (A-F) like scientific papers")
print(f"   🔄 Cross-response parameter importance ranking")
print(f"   📈 Complete plot data saved for MATLAB replication")
print(f"   🎨 QUALITY INDICATORS: Visual assessment with 🟢🟡🔴")

print(f"\n🎨 QUALITY INDICATOR BENEFITS:")
print(f"   🚀 Instant visual assessment of metamodel quality")
print(f"   🎯 Quick identification of usable models for optimization")
print(f"   📊 Separate R² and MAPE quality indicators")
print(f"   🏆 Combined overall quality assessment")
print(f"   📋 Enhanced table readability for decision making")

print(f"\n💡 Next Steps:")
print(f"   1. Review enhanced plots in {PLOTS_FOLDER}/")
print(f"   2. Focus on 🟢 and 🟡 models for multi-objective optimization")
print(f"   3. Analyze parameter importance trends across responses")
print(f"   4. Replicate/customize plots in MATLAB using saved data")
print(f"   5. Generate response surface contour plots for best models")
print(f"   6. Validate 🔴 models with experimental data before use")

print(f"\n🔧 MATLAB Enhanced Usage:")
print(f"   load('enhanced_metamodel_results.mat')")
print(f"   % All coefficients, statistics, and plot data available")
print(f"   % Individual enhanced metamodels in individual_enhanced_metamodels/")
print(f"   % Plot replication data included")
print(f"   % Quality thresholds defined for filtering")

print(f"\n{'='*70}")
