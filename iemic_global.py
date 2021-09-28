"""
  example for running I-EMIC in a global configuration using OMUSE
  
"""

import numpy

from omuse.units import units, constants

from omuse.community.iemic.interface import iemic

from omuse.io import write_set_to_file

from fvm import Continuation

from matplotlib import pyplot

from bstream import barotropic_streamfunction, overturning_streamfunction,z_from_cellcenterz


"""
OCEAN parameters:

Ocean__Analyze_Jacobian: True
Ocean__Belos_Solver__FGMRES_explicit_residual_test: False
Ocean__Belos_Solver__FGMRES_iterations: 500
Ocean__Belos_Solver__FGMRES_output: 100
Ocean__Belos_Solver__FGMRES_restarts: 0
Ocean__Belos_Solver__FGMRES_tolerance: 1e-08
Ocean__Input_file: ocean_input.h5
Ocean__Load_mask: True
Ocean__Load_salinity_flux: False
Ocean__Load_state: False
Ocean__Load_temperature_flux: False
Ocean__Max_mask_fixes: 5
Ocean__Output_file: ocean_output.h5
Ocean__Save_column_integral: False
Ocean__Save_frequency: 0
Ocean__Save_mask: True
Ocean__Save_salinity_flux: True
Ocean__Save_state: True
Ocean__Save_temperature_flux: True
Ocean__THCM__Compute_salinity_integral: True
Ocean__THCM__Coriolis_Force: 1
Ocean__THCM__Coupled_Salinity: 0
Ocean__THCM__Coupled_Sea_Ice_Mask: 1
Ocean__THCM__Coupled_Temperature: 0
Ocean__THCM__Depth_hdim: 4000.0
Ocean__THCM__Fix_Pressure_Points: False
Ocean__THCM__Flat_Bottom: False
Ocean__THCM__Forcing_Type: 0
Ocean__THCM__Global_Bound_xmax: 350.0
Ocean__THCM__Global_Bound_xmin: 286.0
Ocean__THCM__Global_Bound_ymax: 74.0
Ocean__THCM__Global_Bound_ymin: 10.0
Ocean__THCM__Global_Grid_Size_l: 16
Ocean__THCM__Global_Grid_Size_m: 16
Ocean__THCM__Global_Grid_Size_n: 16
Ocean__THCM__Grid_Stretching_qz: 1.0
Ocean__THCM__Inhomogeneous_Mixing: 0
Ocean__THCM__Integral_row_coordinate_i: -1
Ocean__THCM__Integral_row_coordinate_j: -1
Ocean__THCM__Land_Mask: no_mask_specified
Ocean__THCM__Levitus_Internal_T_S: False
Ocean__THCM__Levitus_S: 1
Ocean__THCM__Levitus_T: 1
Ocean__THCM__Linear_EOS:_alpha_S: 0.00076
Ocean__THCM__Linear_EOS:_alpha_T: 0.0001
Ocean__THCM__Local_SRES_Only: False
Ocean__THCM__Mixing: 1
Ocean__THCM__Periodic: False
Ocean__THCM__Problem_Description: Unnamed
Ocean__THCM__Read_Land_Mask: False
Ocean__THCM__Read_Salinity_Perturbation_Mask: False
Ocean__THCM__Restoring_Salinity_Profile: 1
Ocean__THCM__Restoring_Temperature_Profile: 1
Ocean__THCM__Rho_Mixing: True
Ocean__THCM__Salinity_Forcing_Data: levitus/new/s00an1
Ocean__THCM__Salinity_Integral_Sign: -1
Ocean__THCM__Salinity_Perturbation_Mask: no_mask_specified
Ocean__THCM__Scaling: THCM
Ocean__THCM__Taper: 1
Ocean__THCM__Temperature_Forcing_Data: levitus/new/t00an1
Ocean__THCM__Topography: 1
Ocean__THCM__Wind_Forcing_Data: wind/trtau.dat
Ocean__THCM__Wind_Forcing_Type: 2
Ocean__Use_legacy_fort.3_output: False
Ocean__Use_legacy_fort.44_output: True

starting (derived parameters):
#~ Ocean__THCM__Starting_Parameters__ALPC: nan
#~ Ocean__THCM__Starting_Parameters__AL_T: nan
#~ Ocean__THCM__Starting_Parameters__ARCL: nan
#~ Ocean__THCM__Starting_Parameters__CMPR: nan
#~ Ocean__THCM__Starting_Parameters__CONT: nan
#~ Ocean__THCM__Starting_Parameters__Combined_Forcing: nan
#~ Ocean__THCM__Starting_Parameters__Energy: nan
#~ Ocean__THCM__Starting_Parameters__Flux_Perturbation: nan
#~ Ocean__THCM__Starting_Parameters__Horizontal_Ekman_Number: nan
#~ Ocean__THCM__Starting_Parameters__Horizontal_Peclet_Number: nan
#~ Ocean__THCM__Starting_Parameters__IFRICB: nan
#~ Ocean__THCM__Starting_Parameters__LAMB: nan
#~ Ocean__THCM__Starting_Parameters__MIXP: nan
#~ Ocean__THCM__Starting_Parameters__MKAP: nan
#~ Ocean__THCM__Starting_Parameters__NLES: nan
#~ Ocean__THCM__Starting_Parameters__Nonlinear_Factor: nan
#~ Ocean__THCM__Starting_Parameters__P_VC: nan
#~ Ocean__THCM__Starting_Parameters__RESC: nan
#~ Ocean__THCM__Starting_Parameters__Rayleigh_Number: nan
#~ Ocean__THCM__Starting_Parameters__Rossby_Number: nan
#~ Ocean__THCM__Starting_Parameters__SPL1: nan
#~ Ocean__THCM__Starting_Parameters__SPL2: nan
#~ Ocean__THCM__Starting_Parameters__Salinity_Forcing: nan
#~ Ocean__THCM__Starting_Parameters__Salinity_Homotopy: nan
#~ Ocean__THCM__Starting_Parameters__Salinity_Perturbation: nan
#~ Ocean__THCM__Starting_Parameters__Solar_Forcing: nan
#~ Ocean__THCM__Starting_Parameters__Temperature_Forcing: nan
#~ Ocean__THCM__Starting_Parameters__Vertical_Ekman_Number: nan
#~ Ocean__THCM__Starting_Parameters__Vertical_Peclet_Number: nan
#~ Ocean__THCM__Starting_Parameters__Wind_Forcing: nan
"""

def initialize_global_iemic():

    i = iemic(redirection="none", number_of_workers=1)

    i.parameters.Ocean__Belos_Solver__FGMRES_tolerance=1e-03
    i.parameters.Ocean__Belos_Solver__FGMRES_iterations=800

    i.parameters.Ocean__Save_state=False
    
    i.parameters.Ocean__THCM__Global_Bound_xmin=0
    i.parameters.Ocean__THCM__Global_Bound_xmax=360
    i.parameters.Ocean__THCM__Global_Bound_ymin=-85.5
    i.parameters.Ocean__THCM__Global_Bound_ymax=85.5
    
    i.parameters.Ocean__THCM__Periodic=True
    i.parameters.Ocean__THCM__Global_Grid_Size_n=96
    i.parameters.Ocean__THCM__Global_Grid_Size_m=38 
    i.parameters.Ocean__THCM__Global_Grid_Size_l=12
    
    i.parameters.Ocean__THCM__Grid_Stretching_qz=2.25
    i.parameters.Ocean__THCM__Depth_hdim=5000.

    i.parameters.Ocean__THCM__Topography=0
    i.parameters.Ocean__THCM__Flat_Bottom=False
  
    i.parameters.Ocean__THCM__Read_Land_Mask=True
    i.parameters.Ocean__THCM__Land_Mask="global_96x38x12.mask"
    
    i.parameters.Ocean__THCM__Rho_Mixing=False
    
    i.parameters.Ocean__THCM__Starting_Parameters__Combined_Forcing=0.
    i.parameters.Ocean__THCM__Starting_Parameters__Salinity_Forcing=1.
    i.parameters.Ocean__THCM__Starting_Parameters__Solar_Forcing=0.
    i.parameters.Ocean__THCM__Starting_Parameters__Temperature_Forcing=10.
    i.parameters.Ocean__THCM__Starting_Parameters__Wind_Forcing=1.
       
    i.parameters.Ocean__Analyze_Jacobian=True

    return i
    
if __name__=="__main__":
    instance=initialize_global_iemic()

    print("starting")

    #print out all parameters
    print(instance.parameters)
    
    x = instance.get_state()

    print(instance.Ocean__THCM__Starting_Parameters)
    
    # numerical parameters for the continuation
    parameters={"Newton Tolerance" : 1.e-2, "Verbose" : True,
                "Minimum Step Size" : 0.001,
                "Maximum Step Size" : 0.2,
                "Delta" : 1.e-6 }

    # setup continuation object
    continuation=Continuation(instance, parameters)
    
    # Converge to an initial steady state
    x = continuation.newton(x, 1e-10)
      
    cp=0.2  
      
    print("start continuation, this may take a while")
  
    print("running to continuation parameter", cp)

    x, mu, data = continuation.continuation(x, 'Ocean->THCM->Starting Parameters->Combined Forcing', 0., cp, 0.005)

    write_set_to_file(x.grid, "global_96x38x12.amuse","amuse", overwrite_file=True)

    print("continuation done")

    instance.stop()
        
    print("done")
