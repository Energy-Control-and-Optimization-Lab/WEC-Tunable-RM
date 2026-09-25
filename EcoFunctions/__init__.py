# EcoFunctions/__init__.py (English)
"""
EcoFunctions package - Hydrodynamic analysis tools with DOE, RAO and Power capabilities
Author: Pablo Antonio Matamala Carvajal
"""

__version__ = "2.3.0"
__author__ = "Pablo Antonio Matamala Carvajal"

from .Eco_StlRev import generate_revolution_solid_stl
from .Eco_Cap2B import analyze_two_body_hydrodynamics
from .Eco_Cap1B import analyze_single_body_hydrodynamics
from .Eco_DOE import generate_doe_vectors, print_design_summary
from .Eco_RAO import calculate_rao_heave
from .Eco_Power import calculate_power

__all__ = [
    'generate_revolution_solid_stl',
    'analyze_two_body_hydrodynamics',
    'analyze_single_body_hydrodynamics',
    'generate_doe_vectors',
    'print_design_summary',
    'calculate_rao_heave',
    'calculate_power',
]
