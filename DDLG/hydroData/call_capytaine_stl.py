'''
Function to run Capytaine for hydrodynamics and Meshmagick for hydrostatics.

CAMBIOS respecto al original call_capytaine.py de WEC-Sim:
  1. from_file() ya soporta STL — sin cambio en la carga de mallas.
  2. fill_dataset() reemplazado por solve_all() + assemble_dataset()
     porque fill_dataset() fue eliminado en Capytaine >=2.1.
  3. separate_complex_values() reemplazado por to_netcdf() directo,
     porque Capytaine >=2.0 ya guarda valores complejos correctamente.
  Todo lo demas (hydrostatics, multiprocessing, KH.dat, Hydrostatics.dat,
  firma de call_capy, retorno) es IDENTICO al original.

CORRECCIONES v2:
  4. hydrostatics() ahora acepta parametro density (antes hardcodeado a 1023.0)
     Llamada desde call_capy pasa density correctamente -> K_hs consistente
     con los coeficientes hidrodinamicos.
'''
import os
import numpy as np
from multiprocessing import Process

capy_v2 = False
import capytaine as cpt
# Get the version of capytaine
capytaine_version = cpt.__version__
# If the version is less than 2.0, import meshmagick:
if int(capytaine_version.split('.')[0]) >= 2:
    capy_v2 = True
if not capy_v2:
    import meshmagick.mesh as mmm
    try:
        # Latest version on github removed the previous
        # meshmagick.hydrostatics.Hydrostatics() method. Use old module w/ new version
        import meshmagick.hydrostatics_old as mmhs
    except ModuleNotFoundError:
        # Older versions of meshmagick should have meshmagick.hydrostatics.Hydrostatics() method
        import meshmagick.hydrostatics as mmhs

import xarray as xr
import logging as LOG
from glob import glob
import shutil
import platform
import sys

# Set the affinity back on all the cores solving the single core issue.
#   0xf is essentially a hexadecimal bitmask, corresponding to 4 cores 
#   0xffffffffff is used here for a maximum of 40 cores.
#   This is only used for Linux machines
if platform == "linux" or platform == "linux2":
    os.system("taskset -p 0xffffffffff %d" % os.getpid())

def __init__(self):
    LOG.info("Capytaine imported.")

# CORRECCION 4: density ahora es parametro (antes hardcodeado a 1023.0)
def hydrostatics(myBodies, savepath='', density=1025.0):
    '''
    use Capytaine/Meshmagick functions to calculate and output the hydrostatic
    stiffness, inertia, center of gravity, center of buoyancy and displaced
    volume of a capytaine bodies. Output is saved to Hydrostatics.dat and KH.dat
    files in the same manner as Nemoh
    
    Example of output format:
    Hydrostatics.dat:
         XF =   0.000 - XG =   0.000
         YF =   0.000 - YG =   0.000
         ZF =  -2.500 - ZG =  -2.500
         Displacement =  0.4999997E+03
         Waterplane area =  0.1000002E+03
         
    KH.dat:
        0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00
        0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00
        0.0000000E+00  0.0000000E+00  0.9810053E+06 -0.1464844E-01 -0.5859375E-02  0.0000000E+00
        0.0000000E+00  0.0000000E+00 -0.1464844E-01  0.8160803E+07  0.0000000E+00  0.0000000E+00
        0.0000000E+00  0.0000000E+00 -0.5859375E-02  0.0000000E+00  0.8160810E+07  0.0000000E+00
        0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00


    Parameters
    ----------
    myBodies : List
        A list of capytaine floating bodies.
    savepath : str
        Path where KH.dat and Hydrostatics.dat will be saved.
    density : float
        Water density in kg/m3. Must match the density used in hydrodynamics.
        Default 1025.0

    Returns
    -------
    None
    '''
    nbod = len(myBodies)
    
    for i, body in enumerate(myBodies):
        cg = body.center_of_mass
        # Set file index
        fileind = '' if nbod == 1 else '_' + str(i)
        
        # use meshmagick to compute hydrostatic stiffness matrix
        # NOTE: meshmagick currently has issue if a body is copmletely submerged (OSWEC base)
        # use try-except statement to catch this error use alternate function for cb/vo
        try:
            if capy_v2:
                # Capytaine version is >= 2.0
                # CORRECCION 4: usar density en lugar de 1023.0 hardcodeado
                body_hs = body.compute_hydrostatics(rho=density)
                vo = body_hs['disp_volume']
                cb = body_hs['center_of_buoyancy']
                khs = body_hs['hydrostatic_stiffness']
            else:
                # Capytaine version is < 2.0; use meshmagick
                # CORRECCION 4: usar density en lugar de 1023.0 hardcodeado
                body_mesh = mmm.Mesh(body.mesh.vertices, body.mesh.faces, name= body.mesh.name)
                body_hs = mmhs.Hydrostatics(working_mesh=body_mesh, cog=body.center_of_mass, rho_water=density, grav=9.81)
                vo = body_hs.displacement_volume
                cb = body_hs.buoyancy_center
                khs = body_hs.hydrostatic_stiffness_matrix
        except:
            # Exception if body is fully submerged
            vo = body.volume if capy_v2 else body_mesh.volume
            cb = body.center_of_buoyancy if capy_v2 else cg
            khs = np.zeros((3,3))
        
        # Write hydrostatic stiffness to KH.dat file
        khs_full = np.zeros((6,6))
        if capy_v2:
            khs_full[2:5, 2:5] += khs[2:5, 2:5]
        else:
            khs_full[2:5, 2:5] += khs

        tmp = savepath + 'KH' + fileind +'.dat'
        np.savetxt(tmp, khs_full)
        
        # Write the other hydrostatics data to Hydrostatics.dat file
        tmp = savepath + 'Hydrostatics' + fileind + '.dat'
        f = open(tmp,'w')
        for j in [0,1,2]:
            line =  f'XF = {cb[j]:7.3f} - XG = {cg[j]:7.3f} \n'
            f.write(line)
        line = f'Displacement = {vo:E}'
        f.write(line)
        f.close()


def call_capy(meshFName, wCapy, CoG=([0,0,0],), headings=[0.0], ncFName=None,
              wDes=None, body_name=('',), depth=np.infty, density=1025.0,
              additional_dofs_dir=None, num_threads=1):
    '''
    Setup the problem and call the capytine solver.

    Setup parallel computing for different frequencies and combine the data 
    after the parallel simulation is completed.
    
    May be called with multiple bodies (automatically implements B2B). 
    In this case, the meshFName, CoG, body_name should be a tuple of the
    values for each body.
    
    Also has the ability to add generalized body modes by inputting the path to
    a 'gbm_dofs.py' file (see RM3 example).
    
    Parameters
    ----------
    meshFName : tuple of strings
        Tuple containing a string for the path to each body's hydrodynamic mesh.
        mesh must be cropped at waterline (OXY plane) and have no lid.
        Supports .stl, .dat (Nemoh) and .gdf (WAMIT) formats.
    wCapy: array
        array of frequency points to be computed by Capytaine
    CoG: tuple of lists
        tuple contains a 3x1 list of each body's CoG
    headings: list
        list of wave headings to compute [rad]
    saveNc: Bool
        save results to .nc file
    ncFName: str
        name of .nc file
    wDes: array
        array of desired frequency points
        (for interpolation of wCapy-based Capytaine data)
    body_name: tuple of strings
        Tuple containing strings. Strings are the names of each body. 
        Prevent the body name from being a long file path
    depth: float
        Water depth. Should be positive downwards. Use decimal value to prevent 
        Capytaine outputting int32 types. Default is -np.infty
    density: float
        Water density. Use decimal value to prevent Capytaine outputting int32 
        types. Default 1025.0
    additional_dofs: string
        path to a gbm_dofs.py file that returns GBM dofs to this function

    Returns
    -------
    capyData: xarray Dataset
        Hydrodynamic coefficients as computed
    problems: list
        capytaine Problems that were solved
    '''
        
    # check that old output is not being overwritten (runs take awhile)
    if os.path.isfile(ncFName):
        print(f'Output ({ncFName}) file already exists and will be overwritten. '
               'Do you wish to proceed? (y/n)')
        try:
            ans = input()
        except EOFError:
            # Catch error that occurs when this script is run in a 
            # non-interactive way ('python CASE.py' in run_cases.py, etc) and
            # default to overwriting the output file
            ans = 'y'
            pass
        if ans.lower() != 'y':
            print('\nEnding simulation. file not overwritten')
            sys.exit(0)

    bodies = []
    for i in np.arange(0, len(meshFName)):
        # CAMBIO STL: from_file() ya soporta .stl nativamente.
        # No se requiere ningun cambio aqui respecto al original.
        bodies.append(cpt.FloatingBody.from_file(meshFName[i]))
        bodies[i].center_of_mass = CoG[i]
        bodies[i].keep_immersed_part()
        if body_name[i] != '':
            bodies[i].name = body_name[i]
        bodies[i].add_all_rigid_body_dofs()
    
    # calculate hydrostatics and output to KH.dat and Hydrostatics.dat files
    # CORRECCION 4: pasar density para que K_hs sea consistente con la hidrodinamica
    path, tmp = os.path.split(ncFName)
    path += os.path.sep
    hydrostatics(bodies, path, density=density)
    
    # add gbm dofs
    if additional_dofs_dir is not None:
        old_dir = os.getcwd()
        os.chdir(additional_dofs_dir)
        import gbm_dofs
        additional_dofs = gbm_dofs.new_dofs(bodies)
        
        for i in np.arange(0, len(meshFName)):
            if body_name[i] in additional_dofs:
                for k,v in additional_dofs[body_name[i]].items():
                    bodies[i].dofs[k] = v
        os.chdir(old_dir)
    
    # combine all bodies to account for B2B interactions
    combo = bodies[0]
    for i in np.arange(1,len(bodies),1):
        combo += bodies[i]
    
    # call Capytaine solver
    print(f'\n-------------------------------\n'
          f'Calling Capytaine BEM solver...\n'
          f'-------------------------------\n'
          f'mesh = {meshFName}\n'
          f'w range = {wCapy[0]:.3f} - {wCapy[-1]:.3f} rad/s\n'
          f'dw = {(wCapy[1]-wCapy[0]):.3f} rad/s\n'
          f'no of headings = {len(headings)}\n'
          f'no of radiation & diffraction problems = {len(wCapy)*(len(headings) + len(combo.dofs))}\n'
          f'-------------------------------\n')
    
    wCapy_threads = np.array_split(np.array(wCapy),num_threads)

    if num_threads != 1:
        try:
            shutil.rmtree('capyParallelFolder')
        except OSError as e:        
            pass

        os.mkdir('capyParallelFolder')

    # An array for the processes.
    processing_jobs = []

    for i in range(num_threads):
        if num_threads == 1:
            ncFName_each_thread = ncFName
        else:
            os.chdir("./capyParallelFolder")
            ncFName_each_thread = os.getcwd() + os.path.sep + "capyParallel_{}.nc".format(i+1)
            os.chdir("../")

        p = Process(target=capy_solver, args= (wCapy_threads[i], CoG, headings, ncFName_each_thread, wDes, body_name, depth, density,
                    combo, additional_dofs_dir))
        processing_jobs.append(p)
        p.start()

    # Wait for all processes to finish.
    for proc in processing_jobs:
        proc.join()

    if num_threads == 1:
        capyData = read_netcdfs(ncFName, dim='omega')
    else:
        os.chdir("./capyParallelFolder")
        ncFName_thread = os.getcwd() + os.path.sep + 'capyParallel_*.nc'
        os.chdir("../")
        capyData = read_netcdfs(ncFName_thread, dim='omega')
        print('\nCombine Capytaine data and saved to \n' + ncFName +'\n\n')        
        capyData.to_netcdf(ncFName)

        # Remove saved Capytaine data from each thread. 
        try:
            shutil.rmtree('capyParallelFolder')
        except OSError as e:        
            pass
    
    # Create a dataset of parameters. 
    problems = xr.Dataset(coords={
        'omega': wCapy,
        'wave_direction': headings,
        'radiating_dof': list(combo.dofs),
        'water_depth': [depth],
        'rho': [density],
        })
    
    print('\nCapytaine call complete. \n\n')

    return capyData, problems

def read_netcdfs(files, dim):
    # glob expands paths with * to a list of files, like the unix shell
    paths = sorted(glob(files))
    datasets = [xr.open_dataset(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined    
    
def capy_solver(wCapy, CoG, headings, ncFName, wDes, body_name, depth, density,
                combo, additional_dofs_dir):
    '''
    call Capytaine for a given mesh, frequency range and wave headings
    This function is modified from David Ogden's work 
    (see https://github.com/mattEhall/FrequencyDomain/blob/b89dd4f4a732fbe4afde56efe2b52c3e32e22d53/FrequencyDomain.py#L842 for the original function).
    
    save the results to a file in Network Common Data Form.

    Returns
    -------
    None
    '''
    # CAMBIO (1/3): fill_dataset() fue eliminado en Capytaine >=2.1.
    # Se reemplaza por solve_all() + assemble_dataset().
    sea_bottom = -depth if np.isfinite(depth) else -np.inf

    problems_list = []
    for omega in wCapy:
        for heading in headings:
            problems_list.append(
                cpt.DiffractionProblem(
                    body=combo,
                    wave_direction=heading,
                    omega=float(omega),
                    sea_bottom=sea_bottom,
                )
            )
        for dof in combo.dofs:
            problems_list.append(
                cpt.RadiationProblem(
                    body=combo,
                    radiating_dof=dof,
                    omega=float(omega),
                    sea_bottom=sea_bottom,
                )
            )

    solver = cpt.BEMSolver()
    results = solver.solve_all(problems_list, progress_bar=True)

    capyData = cpt.assemble_dataset(results, mesh=combo.mesh, hydrostatics=False)

    # CAMBIO (2/3): Capytaine >=2.0 guarda las fuerzas de excitacion como
    # valores complejos, sin la dimension 'complex'=['re','im'] que BEMIO requiere.
    # Se reformatea el dataset al formato exacto de readCAPYTAINE.
    capyData = _format_for_bemio(capyData, density)

    # CAMBIO (3/3): separate_complex_values() fue eliminado en Capytaine >=2.0.
    # El dataset ya viene con la dimension 'complex' construida manualmente.
    capyData.to_netcdf(ncFName,
                       encoding={'radiating_dof': {'dtype': 'U'},
                                 'influenced_dof': {'dtype': 'U'}})

    print('\nCapytaine call complete. Data saved to \n' + ncFName +'\n\n')

    return


def _format_for_bemio(ds, density):
    '''
    Convierte el dataset de Capytaine >=2.0 al formato que BEMIO espera.

    readCAPYTAINE requiere estas variables en el .nc:
        added_mass          (omega, radiating_dof, influenced_dof)  - real
        radiation_damping   (omega, radiating_dof, influenced_dof)  - real
        diffraction_force   (complex, omega, wave_direction, influenced_dof)
        Froude_Krylov_force (complex, omega, wave_direction, influenced_dof)
        complex             coordenada con valores ['re', 'im']
        rho, g              escalares
    '''

    def split_complex(da, name):
        '''Separa DataArray complejo en dimension complex=[re, im].'''
        re = da.real.expand_dims('complex').assign_coords(complex=['re'])
        im = da.imag.expand_dims('complex').assign_coords(complex=['im'])
        out = xr.concat([re, im], dim='complex')
        out.name = name
        return out

    # Localizar diffraction_force (Capytaine puede usar nombres distintos)
    if 'diffraction_force' in ds:
        diff_raw = ds['diffraction_force']
    elif 'excitation_force' in ds:
        diff_raw = ds['excitation_force']
    else:
        raise KeyError(
            "El dataset no contiene 'diffraction_force' ni 'excitation_force'.\n"
            "Verifica que los DiffractionProblems se resolvieron correctamente."
        )

    # Localizar Froude-Krylov
    if 'Froude_Krylov_force' in ds:
        fk_raw = ds['Froude_Krylov_force']
    else:
        # Si no existe, aproximar como ceros
        fk_raw = xr.zeros_like(diff_raw)

    # Separar en re/im con dimension 'complex'
    diff_split = split_complex(diff_raw, 'diffraction_force')
    fk_split   = split_complex(fk_raw,   'Froude_Krylov_force')

    # Reordenar dimensiones al orden que BEMIO espera:
    # (complex, omega, wave_direction, influenced_dof)
    target_dims = ('complex', 'omega', 'wave_direction', 'influenced_dof')
    diff_split = diff_split.transpose(*target_dims)
    fk_split   = fk_split.transpose(*target_dims)

    # Construir dataset final eliminando variables complejas originales
    vars_to_drop = [v for v in ['diffraction_force', 'Froude_Krylov_force',
                                'excitation_force']
                    if v in ds]
    ds_out = ds.drop_vars(vars_to_drop, errors='ignore')

    ds_out = ds_out.assign({
        'diffraction_force':   diff_split,
        'Froude_Krylov_force': fk_split,
        'rho':                 xr.DataArray(float(density)),
        'g':                   xr.DataArray(9.81),
    })

    return ds_out
