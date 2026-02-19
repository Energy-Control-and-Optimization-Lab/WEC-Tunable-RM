"""
St6_B_MetamodelAnalysis.py - Metamodel Sensitivity Analysis (HARDCODED VERSION)
Comprehensive analysis of metamodel sensitivity and parameter importance

Author: Pablo Antonio Matamala Carvajal
Date: 2025-12-04
Version: HARDCODED - Simple version without dynamic loading
Description:
- Case 1: Pareto Chart (standardized effects ranking)
- Case 3: Parametric Response Curves (vary 1, fix 5)
- Focus on P_efficiency with proper units [W/(η²kg)]
- All parameters defined directly in script (no file dependencies)
"""

import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Set non-interactive backend for plots
import matplotlib
matplotlib.use('Agg')

#%%============================================================================
# HARDCODED CONFIGURATION (NO FILE DEPENDENCIES)
#%%============================================================================

# Response variable
RESPONSE_VARIABLE = "P_efficiency"
RESPONSE_DISPLAY_NAME = "Power-Mass ratio"  # Display name for plots
RESPONSE_UNITS = "W/(η²kg)"

# Folders
METAMODEL_FOLDER = "EcoData/MetaModel/Analysis/individual_enhanced_metamodels"
OUTPUT_FOLDER = "EcoData/Borrador"
PLOTS_FOLDER = os.path.join(OUTPUT_FOLDER, "sensitivity_plots")

# Analysis settings
N_TOP_EFFECTS = 15
N_POINTS_CURVE = 50
FIGURE_SIZE = (15, 10)
DPI = 300

# Parameter information
PARAMETER_NAMES = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
PARAMETER_DESCRIPTIONS = [
    'Float diameter',
    'Float draft', 
    'Float angle',
    'Spar draft',
    'Spar plate diameter',
    'PTO damping'
]
PARAMETER_UNITS = ['m', 'm', '°', 'm', 'm', 'kg/s']

# HARDCODED: Parameter ranges (from DOE)
PARAMETER_RANGES = {
    'D1': [0.6, 1.0],        # Float diameter [m]
    'D2': [0.05, 0.15],      # Float draft [m]  
    'D3': [0.0, 20.0],       # Float angle [deg]
    'D4': [1.2, 1.6],        # Spar draft [m]
    'D5': [0.6, 1.0],        # Spar plate diameter [m]
    'D6': [300.0, 700.0]    # PTO damping [kg/s]
}

# HARDCODED: Middle point (center of parameter ranges)
OPTIMAL_POINT = np.array([
    0.8,       # D1 = (0.6 + 1.0) / 2
    0.1,       # D2 = (0.05 + 0.15) / 2  
    10.0,      # D3 = (0.0 + 20.0) / 2
    1.4,       # D4 = (1.2 + 1.6) / 2
    0.8,       # D5 = (0.6 + 1.0) / 2
    500.0      # D6 = (300.0 + 700.0) / 2
])

OPTIMAL_VALUE = 4.5  # P_efficiency [W/(η²kg)] at middle point

# Color scheme
COLORS = {
    'main': 'steelblue',
    'interaction': 'orange', 
    'quadratic': 'green',
    'curve': 'blue',
    'optimal': 'red'
}

#%%============================================================================
# FUNCTIONS
#%%============================================================================

def create_output_folders():
    """Create output directories"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    print(f"📁 Output folders created:")
    print(f"   - {OUTPUT_FOLDER}/")
    print(f"   - {PLOTS_FOLDER}/")

def load_metamodel(response_name):
    """Load metamodel for specified response variable"""
    
    print(f"📂 Loading metamodel for: {response_name}")
    
    # Try enhanced metamodel first
    enhanced_file = os.path.join(METAMODEL_FOLDER, f"{response_name}_enhanced_metamodel.pkl")
    regular_file = os.path.join(METAMODEL_FOLDER, f"{response_name}_metamodel.pkl")
    
    for file_path in [enhanced_file, regular_file]:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    metamodel_data = pickle.load(f)
                
                print(f"✅ Metamodel loaded from: {file_path}")
                
                # Verify required components
                required_keys = ['model', 'polynomial_features', 'parameter_names', 'statistics']
                missing_keys = [key for key in required_keys if key not in metamodel_data]
                
                if missing_keys:
                    print(f"⚠️  Missing keys: {missing_keys}")
                    continue
                
                # Display metamodel info
                stats = metamodel_data['statistics']
                print(f"   R²: {stats['r2']:.4f}")
                print(f"   MAPE: {stats['mape']:.2f}%")
                print(f"   Model adequate: {'✅' if stats['model_adequate'] else '❌'}")
                
                return metamodel_data
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue
    
    print(f"❌ No valid metamodel found for {response_name}")
    return None

def calculate_standardized_effects(model, poly_features, parameter_names):
    """Calculate standardized effects for Pareto analysis"""
    
    # Get feature names and coefficients
    feature_names = poly_features.get_feature_names_out(parameter_names)
    coefficients = model.coef_[1:]  # Skip intercept
    
    # Create design matrix for effects calculation
    n_samples = 1000
    X_samples = []
    
    for param in parameter_names:
        if param in PARAMETER_RANGES:
            min_val, max_val = PARAMETER_RANGES[param]
            samples = np.random.uniform(-1, 1, n_samples)
            physical_samples = min_val + (max_val - min_val) * (samples + 1) / 2
            X_samples.append(physical_samples)
        else:
            raise ValueError(f"Parameter {param} not found in PARAMETER_RANGES")
    
    X_samples = np.column_stack(X_samples)
    X_poly = poly_features.transform(X_samples)
    
    # Calculate standardized effects
    standardized_effects = []
    
    for i, (coef, feature_name) in enumerate(zip(coefficients, feature_names[1:]), start=1):
        std_effect = abs(coef) * np.std(X_poly[:, i])
        
        # Clean up effect name with subscripts
        effect_type = 'main'
        clean_name = feature_name
        
        if '^2' in feature_name:
            # Quadratic: D1^2 -> D₁²
            param = feature_name.replace('^2', '')
            param_idx = int(param[1:])
            subscripts = '₁₂₃₄₅₆'
            clean_name = f'D{subscripts[param_idx-1]}²'
            effect_type = 'quadratic'
        elif ' ' in feature_name:
            # Interaction: D1 D2 -> D₁D₂
            params = feature_name.split(' ')
            if len(params) == 2:
                param1_idx = int(params[0][1:])
                param2_idx = int(params[1][1:])
                subscripts = '₁₂₃₄₅₆'
                clean_name = f'D{subscripts[param1_idx-1]}D{subscripts[param2_idx-1]}'
                effect_type = 'interaction'
        else:
            # Main effect: D1 -> D₁
            if feature_name.startswith('D'):
                param_idx = int(feature_name[1:])
                subscripts = '₁₂₃₄₅₆'
                clean_name = f'D{subscripts[param_idx-1]}'
                effect_type = 'main'
        
        standardized_effects.append({
            'name': clean_name,
            'effect': std_effect,
            'type': effect_type,
            'coefficient': coef,
            'original_name': feature_name
        })
    
    # Sort by effect magnitude
    standardized_effects.sort(key=lambda x: x['effect'], reverse=True)
    
    return standardized_effects

def create_pareto_chart(standardized_effects, response_name, response_units):
    """Create Pareto chart (Case 1)"""
    
    print(f"📊 Creating Pareto Chart for {response_name}...")
    
    # Take top effects
    top_effects = standardized_effects[:N_TOP_EFFECTS]
    
    names = [effect['name'] for effect in top_effects]
    values = [effect['effect'] for effect in top_effects]
    types = [effect['type'] for effect in top_effects]
    
    colors = [COLORS[effect_type] for effect_type in types]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=16)  # Increased font size for y-axis labels
    ax.invert_yaxis()
    # Use display name for response variable
    display_name = RESPONSE_DISPLAY_NAME if response_name == "P_efficiency" else response_name
    
    ax.set_xlabel(f'Standardized Effect [{response_units}]', fontsize=18)  # Increased font size for x-axis label
    ax.set_title(f'Pareto Chart: {display_name} Sensitivity Analysis\n' +
                f'Top {N_TOP_EFFECTS} Standardized Effects', 
                fontsize=20, fontweight='bold', pad=20)  # Increased title font size
    ax.grid(True, alpha=0.3, axis='x')
    
    # Increase tick label sizes
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=16)
    
    # Parameter mapping legend (positioned at bottom center-right with larger font)
    legend_text = "Parameter Mapping:\n"
    subscripts = '₁₂₃₄₅₆'
    for i, (param, desc, unit) in enumerate(zip(PARAMETER_NAMES, PARAMETER_DESCRIPTIONS, PARAMETER_UNITS)):
        legend_text += f"D{subscripts[i]} = {param} ({desc} [{unit}])\n"
    
    ax.text(0.45, 0.32, legend_text, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Effect type legend (positioned at bottom right, lower than parameter mapping)
    legend_elements = []
    for effect_type, color in COLORS.items():
        if effect_type in ['main', 'interaction', 'quadratic'] and effect_type in types:
            from matplotlib.patches import Rectangle
            legend_elements.append(Rectangle((0,0),1,1, facecolor=color, alpha=0.8, 
                                           label=effect_type.title()))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower right', fontsize=12, 
                 bbox_to_anchor=(0.98, 0.02))  # Positioned at very bottom right
    
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_FOLDER, f"{response_name}_pareto_analysis.png")
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    print(f"✅ Pareto chart saved: {plot_path}")
    plt.close()
    
    return top_effects

def create_pareto_chart_by_effect_type(standardized_effects, response_name, response_units):
    """Create Pareto charts separated by effect type (linear, quadratic, interactions)"""
    
    print(f"📊 Creating Pareto Charts by Effect Type for {response_name}...")
    
    # Separate effects by type
    main_effects = [e for e in standardized_effects if e['type'] == 'main']
    quadratic_effects = [e for e in standardized_effects if e['type'] == 'quadratic']
    interaction_effects = [e for e in standardized_effects if e['type'] == 'interaction']
    
    # Use display name for response variable
    display_name = RESPONSE_DISPLAY_NAME if response_name == "P_efficiency" else response_name
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Helper function to create individual pareto chart
    def plot_effects(ax, effects, effect_type, color, title):
        if not effects:
            ax.text(0.5, 0.5, f'No {effect_type} effects found', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=16, fontweight='bold')
            return
        
        # Take top N effects for this type
        top_effects_type = effects[:min(10, len(effects))]  # Max 10 per chart
        
        names = [effect['name'] for effect in top_effects_type]
        values = [effect['effect'] for effect in top_effects_type]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, values, color=color, alpha=0.8, 
                      edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=14)
        ax.invert_yaxis()
        ax.set_xlabel(f'Standardized Effect [{response_units}]', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Increase tick label sizes
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=14)
        
        return top_effects_type
    
    # Plot each type
    top_main = plot_effects(ax1, main_effects, 'linear', COLORS['main'], 
                           f'Linear Effects\n{display_name}')
    
    top_quadratic = plot_effects(ax2, quadratic_effects, 'quadratic', COLORS['quadratic'], 
                                f'Quadratic Effects\n{display_name}')
    
    top_interaction = plot_effects(ax3, interaction_effects, 'interaction', COLORS['interaction'], 
                                  f'Interaction Effects\n{display_name}')
    
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_FOLDER, f"{response_name}_pareto_by_effect_type.png")
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    print(f"✅ Pareto charts by effect type saved: {plot_path}")
    plt.close()
    
    # Print summary
    print(f"📊 Effect type summary:")
    print(f"   Linear effects: {len(main_effects)}")
    print(f"   Quadratic effects: {len(quadratic_effects)}")
    print(f"   Interaction effects: {len(interaction_effects)}")
    
    return {
        'main_effects': top_main if top_main else [],
        'quadratic_effects': top_quadratic if top_quadratic else [],
        'interaction_effects': top_interaction if top_interaction else []
    }

def create_parametric_curves(metamodel_data, response_name, response_units):
    """Create parametric response curves (Case 3: vary 1, fix 5)"""
    
    print(f"📈 Creating parametric curves for {response_name}...")
    
    model = metamodel_data['model']
    poly_features = metamodel_data['polynomial_features']
    
    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
    axes = axes.flatten()
    
    curve_data = {}
    
    for i, param in enumerate(PARAMETER_NAMES):
        ax = axes[i]
        
        if param in PARAMETER_RANGES:
            min_val, max_val = PARAMETER_RANGES[param]
        else:
            print(f"⚠️  Parameter {param} not found in ranges, skipping...")
            continue
        
        # Parameter sweep
        param_values = np.linspace(min_val, max_val, N_POINTS_CURVE)
        response_values = []
        
        for param_val in param_values:
            # Fix 5 parameters at middle values, vary 1
            design_point = OPTIMAL_POINT.copy()
            design_point[i] = param_val
            
            X_poly = poly_features.transform([design_point])
            response_pred = model.predict(X_poly)[0]
            response_values.append(response_pred)
        
        response_values = np.array(response_values)
        
        # Store curve data
        curve_data[param] = {
            'param_values': param_values,
            'response_values': response_values,
            'middle_value': OPTIMAL_POINT[i],  # Changed from optimal_value
            'range': [min_val, max_val],
            'impact_range': np.max(response_values) - np.min(response_values)
        }
        
        # Plot curve with different line styles based on Y-axis values (response)
        # Continuous line when response is between 3 and 4.5, dashed elsewhere
        
        # Find segments where response is in different ranges
        mask_low = response_values < 2.4      # Below 3: dashed
        mask_optimal = (response_values >= 2.4) & (response_values <= 4.5)  # 3-4.5: continuous  
        mask_high = response_values > 4.5     # Above 4.5: dashed
        
        # Plot segments with different line styles
        if np.any(mask_low):
            ax.plot(param_values[mask_low], response_values[mask_low], 
                   COLORS['curve'], linewidth=2, linestyle='--', alpha=0.7)
        
        if np.any(mask_optimal):
            ax.plot(param_values[mask_optimal], response_values[mask_optimal], 
                   COLORS['curve'], linewidth=2, linestyle='-', 
                   label='Response curve (continuous: 3.0-4.5)')
        
        if np.any(mask_high):
            ax.plot(param_values[mask_high], response_values[mask_high], 
                   COLORS['curve'], linewidth=2, linestyle='--', alpha=0.7)
        
        # Add connecting segments at boundaries for visual continuity
        for j in range(len(response_values) - 1):
            y1, y2 = response_values[j], response_values[j+1]
            
            # Check if we're crossing boundaries
            crosses_3 = (y1 < 2.4 < y2) or (y2 < 2.4 < y1)
            crosses_4_5 = (y1 < 4.5 < y2) or (y2 < 4.5 < y1)
            
            if crosses_3 or crosses_4_5:
                # Draw connecting segment at boundary
                ax.plot([param_values[j], param_values[j+1]], 
                       [response_values[j], response_values[j+1]], 
                       COLORS['curve'], linewidth=2, linestyle='-', alpha=0.5)
        
        # If no points in optimal range, use regular dashed line
        if not np.any(mask_optimal):
            if not (np.any(mask_low) or np.any(mask_high)):
                ax.plot(param_values, response_values, COLORS['curve'], linewidth=2, 
                       linestyle='--', alpha=0.7, label='Response curve')
        
        # Mark middle point (changed from optimal)
        middle_response = model.predict(poly_features.transform([OPTIMAL_POINT]))[0]
        ax.axvline(x=OPTIMAL_POINT[i], color=COLORS['optimal'], linestyle='--', linewidth=2, 
                  label='Middle point')
        ax.plot(OPTIMAL_POINT[i], middle_response, 'o', color=COLORS['optimal'], 
                markersize=8, label=f'Middle: {OPTIMAL_POINT[i]:.3f}')
        
        # Customize subplot (removed title as requested)
        ax.set_xlabel(f'{param} [{PARAMETER_UNITS[i]}]', fontsize=11)
        # Use display name for response variable
        display_name = RESPONSE_DISPLAY_NAME if response_name == "P_efficiency" else response_name
        ax.set_ylabel(f'{display_name} [{response_units}]', fontsize=11)
        # Removed title line: ax.set_title(...)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([2.5, 5.0])  # Fixed Y-axis limits: 2.5 to 5.0
        
        print(f"   {param}: Impact range = {curve_data[param]['impact_range']:.3f} {response_units}")
    
    # Use display name for response variable
    display_name = RESPONSE_DISPLAY_NAME if response_name == "P_efficiency" else response_name
    
    plt.suptitle(f'Parametric Response Curves: {display_name}\n' +
                f'Methodology: Vary 1 parameter, fix 5 at middle values', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_FOLDER, f"{response_name}_parametric_curves.png")
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    print(f"✅ Parametric curves saved: {plot_path}")
    plt.close()
    
    return curve_data

def analyze_metamodel_sensitivity(response_name=None, response_units=None):
    """Main sensitivity analysis function"""
    
    if response_name is None:
        response_name = RESPONSE_VARIABLE
    if response_units is None:
        response_units = RESPONSE_UNITS
    
    print("="*70)
    print("St6_B - METAMODEL SENSITIVITY ANALYSIS (HARDCODED)")
    print("="*70)
    
    # Display hardcoded configuration
    print(f"📊 Configuration:")
    print(f"   Response: {response_name} [{response_units}]")
    print(f"   Middle value: {OPTIMAL_VALUE:.6f} {response_units}")
    print(f"   Parameter ranges:")
    for param, range_vals in PARAMETER_RANGES.items():
        print(f"      {param}: [{range_vals[0]:6.3f}, {range_vals[1]:6.3f}] {PARAMETER_UNITS[PARAMETER_NAMES.index(param)]}")
    print(f"   Middle point:")
    for i, (param, value, unit) in enumerate(zip(PARAMETER_NAMES, OPTIMAL_POINT, PARAMETER_UNITS)):
        print(f"      {param} = {value:8.3f} {unit}")
    
    # Setup
    create_output_folders()
    
    # Load metamodel
    metamodel_data = load_metamodel(response_name)
    if metamodel_data is None:
        print(f"❌ Cannot proceed without metamodel")
        return None
    
    print(f"\n🔬 Starting analysis for: {response_name}")
    
    # Case 1: Pareto Chart
    print(f"\n{'='*50}")
    print("CASE 1: PARETO CHART ANALYSIS")
    print(f"{'='*50}")
    
    try:
        standardized_effects = calculate_standardized_effects(
            metamodel_data['model'], 
            metamodel_data['polynomial_features'],
            metamodel_data['parameter_names']
        )
        
        top_effects = create_pareto_chart(standardized_effects, response_name, response_units)
        
        # Create Pareto charts by effect type
        effects_by_type = create_pareto_chart_by_effect_type(standardized_effects, response_name, response_units)
        
        print(f"📊 Top {len(top_effects)} standardized effects:")
        for i, effect in enumerate(top_effects[:5], 1):
            print(f"   {i}. {effect['name']:>4s}: {effect['effect']:8.3f} {response_units} ({effect['type']})")
        if len(top_effects) > 5:
            print(f"   ... (showing top 5 of {len(top_effects)})")
            
    except Exception as e:
        print(f"❌ Error in Pareto analysis: {e}")
        standardized_effects = None
        top_effects = None
        effects_by_type = None
    
    # Case 3: Parametric Curves
    print(f"\n{'='*50}")
    print("CASE 3: PARAMETRIC RESPONSE CURVES")
    print(f"{'='*50}")
    
    try:
        curve_data = create_parametric_curves(metamodel_data, response_name, response_units)
        
        print(f"📈 Parameter impact ranking:")
        impact_ranking = sorted(curve_data.items(), 
                              key=lambda x: x[1]['impact_range'], reverse=True)
        
        for i, (param, data) in enumerate(impact_ranking, 1):
            print(f"   {i}. {param}: {data['impact_range']:8.3f} {response_units}")
            
    except Exception as e:
        print(f"❌ Error in parametric curves: {e}")
        curve_data = None
    
    # Save summary
    print(f"\n💾 Saving summary...")
    
    summary_data = {
        'response_name': response_name,
        'response_units': response_units,
        'middle_point': OPTIMAL_POINT.tolist(),  # Changed from optimal_point
        'middle_value': OPTIMAL_VALUE,  # Changed from optimal_value
        'parameter_ranges': PARAMETER_RANGES,
        'parameter_names': PARAMETER_NAMES,
        'parameter_descriptions': PARAMETER_DESCRIPTIONS,
        'parameter_units': PARAMETER_UNITS,
        'standardized_effects': standardized_effects if standardized_effects else [],
        'top_effects': top_effects if top_effects else [],
        'effects_by_type': effects_by_type if effects_by_type else {},
        'curve_data': curve_data if curve_data else {},
        'metamodel_info': {
            'r2': metamodel_data['statistics']['r2'] if metamodel_data else None,
            'mape': metamodel_data['statistics']['mape'] if metamodel_data else None,
            'model_adequate': metamodel_data['statistics']['model_adequate'] if metamodel_data else None
        },
        'analysis_config': {
            'n_top_effects': N_TOP_EFFECTS,
            'n_points_curve': N_POINTS_CURVE,
            'version': 'hardcoded_simple'
        }
    }
    
    summary_file = os.path.join(OUTPUT_FOLDER, f"{response_name}_sensitivity_analysis.pkl")
    with open(summary_file, 'wb') as f:
        pickle.dump(summary_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✅ Summary saved: {summary_file}")
    
    # Final summary
    print(f"\n🎉 SENSITIVITY ANALYSIS COMPLETED")
    print("="*70)
    print(f"📊 Results:")
    print(f"   - Pareto Chart: {PLOTS_FOLDER}/{response_name}_pareto_analysis.png")
    print(f"   - Pareto by Effect Type: {PLOTS_FOLDER}/{response_name}_pareto_by_effect_type.png")
    print(f"   - Parametric Curves: {PLOTS_FOLDER}/{response_name}_parametric_curves.png")
    print(f"   - Summary: {OUTPUT_FOLDER}/{response_name}_sensitivity_analysis.pkl")
    
    if standardized_effects and curve_data:
        print(f"\n🎯 Key Insights:")
        print(f"   - Most important effect: {standardized_effects[0]['name']} ({standardized_effects[0]['type']})")
        print(f"   - Highest parameter impact: {max(curve_data.items(), key=lambda x: x[1]['impact_range'])[0]}")
        print(f"   - Metamodel R²: {metamodel_data['statistics']['r2']:.3f}")
    
    print("="*70)
    
    return summary_data

#%%============================================================================
# MAIN EXECUTION
#%%============================================================================

if __name__ == "__main__":
    # Show configuration
    print("📋 HARDCODED CONFIGURATION:")
    print(f"   Response: {RESPONSE_VARIABLE} (Display: {RESPONSE_DISPLAY_NAME}) [{RESPONSE_UNITS}]")
    print(f"   Middle value: {OPTIMAL_VALUE:.6f}")
    print(f"   Middle point: {OPTIMAL_POINT}")
    
    # Run analysis
    results = analyze_metamodel_sensitivity()
    
    print(f"\n💡 To analyze other variables:")
    print(f"   analyze_metamodel_sensitivity('P_avg', 'W')")
    print(f"   analyze_metamodel_sensitivity('phase_at_1_50', 'degrees')")
