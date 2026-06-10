"""
Replica el tutorial rm3.py de WEC-Sim para mallas STL.

Diferencias respecto al rm3.py original:
  - Importa call_capytaine_stl en lugar de call_capytaine
  - Mallas en formato .stl en lugar de .dat
  - CoG, profundidad y frecuencias ajustados a la geometria del caso
  Todo lo demas es identico al original.

CORRECCIONES v2:
  1. Densidad consistente: 1025.0 kg/m3 pasada explicitamente a call_capy
     (antes hydrostatics() usaba rho=1023.0 hardcodeado — K_hs incorrecto)
  2. Rango de frecuencias extendido: desde 0.1 rad/s en lugar de 0.5
     (el IRF de regularCIC requiere que B_rad -> 0 en ambos extremos)
  3. Mayor resolucion en zona de bajas frecuencias (omega < 1.5 rad/s)
     donde ocurrian las amplificaciones erroneas
"""

# setup environment
import os
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import sys

# Add directory with the call_capytaine_stl.py file to the system path.
currentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentdir)
import call_capytaine_stl as cc

# Define parameters --------------------------------------------------------- #
bem_file = (os.getcwd() + os.path.sep + 'float.stl',   # mesh files (.stl)
            os.getcwd() + os.path.sep + 'spar.stl')

bem_cg = ((0, 0, 0),      # CoG float  — verificado con Capytaine
          (0, 0, 0))   # CoG spar   — corregido desde Capytaine (antes -0.65)

bem_name = ('float',        # body names — generan DOFs: float__Heave, etc.
            'spar')

# CORRECCION 2 y 3: rango extendido con mayor resolucion a bajas frecuencias
# Regla: omega_min < 0.5 * omega_min_simulacion (omega_min_sim = 1.6 rad/s)
# Regla: B_rad debe -> 0 claramente en ambos extremos del rango
# Regla: A(omega) debe converger a A_inf antes de omega_max (verificado en ~10-12 rad/s)
bem_w = np.concatenate([
    np.linspace(0.05, 1.5,  30),   # bajas frecuencias — zona critica, densa
    np.linspace(1.5,  8.0,  80),   # rango medio
    np.linspace(8.0,  12.0, 30),   # altas frecuencias — para convergencia de A_inf
])
bem_w = np.unique(np.round(bem_w, 4))  # elimina duplicados en la union

bem_headings = np.linspace(0, 0, 1)    # wave headings (rad)
bem_depth    = 2.4                      # water depth (m)
bem_density  = 1025.0                  # CORRECCION 1: densidad explicita y consistente

bem_ncFile   = os.getcwd() + os.path.sep + 'ECO_RM_WEC.nc'
num_threads  = 2
# --------------------------------------------------------------------------- #

# Run Capytaine
if __name__ == '__main__':
    cc.call_capy(meshFName   = bem_file,
                 wCapy       = bem_w,
                 CoG         = bem_cg,
                 headings    = bem_headings,
                 ncFName     = bem_ncFile,
                 body_name   = bem_name,
                 depth       = bem_depth,
                 density     = bem_density,   # CORRECCION 1: pasado explicitamente
                 num_threads = num_threads)
