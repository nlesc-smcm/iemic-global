import numpy

from omuse.units import units, constants

from omuse.community.iemic.interface import iemic
from omuse.community.iemic.implicit_utils import newton, continuation

from matplotlib import pyplot

from bstream import barotropic_streamfunction, overturning_streamfunction,z_from_cellcenterz


"""
default OCEAN parameters:

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

    i.parameters.Ocean__Save_state=False

    i.parameters.Ocean__THCM__Global_Bound_xmax=360.
    i.parameters.Ocean__THCM__Global_Bound_xmin=0.
    i.parameters.Ocean__THCM__Global_Bound_ymax=85.5
    i.parameters.Ocean__THCM__Global_Bound_ymin=-85.5

    i.parameters.Ocean__THCM__Periodic=True
    i.parameters.Ocean__THCM__Global_Grid_Size_n=96
    i.parameters.Ocean__THCM__Global_Grid_Size_m=38 
    i.parameters.Ocean__THCM__Global_Grid_Size_l=12

    #~ i.parameters.Ocean__THCM__Levitus_Internal_T_S=False

    #~ i.parameters.Ocean__THCM__Grid_Stretching_qz=1.

    i.parameters.Ocean__THCM__Topography=1
    i.parameters.Ocean__THCM__Flat_Bottom=True
    #~ i.parameters.Ocean__THCM__Coriolis_Force=1

    i.parameters.Ocean__THCM__Read_Land_Mask=True
    i.parameters.Ocean__THCM__Land_Mask="global_96x38x12.mask"
    #~ i.parameters.Ocean__THCM__Land_Mask="mask_global_48x19x12"
    #~ i.parameters.Ocean__THCM__Land_Mask="mask_natl16"

    #~ i.parameters.Ocean__THCM__Depth_hdim=5000.

    i.parameters.Ocean__THCM__Rho_Mixing=True

    #~ i.parameters.Ocean__THCM__Mixing=1
    #~ i.parameters.Ocean__THCM__Taper=1

    #~ i.parameters.Ocean__THCM__Forcing_Type=0
    #~ i.parameters.Ocean__THCM__Wind_Forcing_Type=2

    i.parameters.Ocean__THCM__Starting_Parameters__SPL1=1.
    i.parameters.Ocean__THCM__Starting_Parameters__SPL2=0.01

    i.parameters.Ocean__THCM__Starting_Parameters__Combined_Forcing=0.
    i.parameters.Ocean__THCM__Starting_Parameters__Salinity_Forcing=0.
    i.parameters.Ocean__THCM__Starting_Parameters__Solar_Forcing=0.
    i.parameters.Ocean__THCM__Starting_Parameters__Temperature_Forcing=0.
    i.parameters.Ocean__THCM__Starting_Parameters__Wind_Forcing=1.

    # matching Dijkstra paper:
    i.parameters.Ocean__THCM__Starting_Parameters__Rossby_Number=1.1e-4

    #~ i.parameters.Ocean__THCM__Starting_Parameters__Horizontal_Ekman_Number=2.7e-3  
    i.parameters.Ocean__THCM__Starting_Parameters__Vertical_Ekman_Number=4.3e-7

    i.parameters.Ocean__THCM__Starting_Parameters__Horizontal_Peclet_Number=1.5e-3
    i.parameters.Ocean__THCM__Starting_Parameters__Vertical_Peclet_Number=3.9e-4

    i.parameters.Ocean__THCM__Starting_Parameters__Rayleigh_Number=4.2e-2

    i.parameters.Ocean__THCM__Starting_Parameters__AL_T=2.7e-2
    i.parameters.Ocean__THCM__Starting_Parameters__LAMB=7.6


    i.parameters.Ocean__Analyze_Jacobian=True

    return i

if __name__=="__main__":
    instance=initialize_global_iemic()

    xmin=instance.parameters.Ocean__THCM__Global_Bound_xmin
    xmax=instance.parameters.Ocean__THCM__Global_Bound_xmax
    ymin=instance.parameters.Ocean__THCM__Global_Bound_ymin
    ymax=instance.parameters.Ocean__THCM__Global_Bound_ymax


    print(instance.parameters)

    #~ instance.parameters.Continuation__destination_0=1.0

    # Converge to an initial steady state
    x = instance.get_state()

    print(
    instance.Ocean__THCM__Starting_Parameters)

    x = newton(instance, x, 1e-10)

    #~ input()

    lat=instance.grid.lat
    lon=instance.grid.lon
    zc=instance.grid[0,0,:].z

    z=z_from_cellcenterz(zc)
    #~ yvar=get_yvar(instance.grid)


    x = continuation(instance, x, 'Ocean->THCM->Starting Parameters->Combined Forcing', 1., 0.2, 20, tol=1.e-6)

    """
    print(instance.Continuation)

    instance.Continuation.destination_0=1.

    instance.step_continuation()
    x=instance.grid
    print(x)
    """

    uvel=x[:,:,:].u_velocity
    vvel=x[:,:,:].v_velocity

    umax=numpy.abs(uvel).max()
    vmax=numpy.abs(vvel).max()

    print("umax, vmax:", umax,vmax)

    #~ fig, axs=pyplot.subplots(2,1,figsize=(8,8))

    #~ im=axs[0].imshow(uvel.T, origin="lower", cmap="seismic", vmax=umax, vmin=-umax, extent=[xmin,xmax,ymin,ymax], interpolation="none")
    #~ fig.colorbar(im,ax=axs[0],label="uvel")

    #~ im=axs[1].imshow(vvel.T, origin="lower", cmap="seismic", vmax=vmax, vmin=-vmax, extent=[xmin,xmax,ymin,ymax], interpolation="none")
    #~ fig.colorbar(im,ax=axs[1],label="vvel")

    #~ pyplot.savefig("test.png")

    #~ pyplot.show()

    fig, ax=pyplot.subplots(2,1,figsize=(8,8))

    dz=z[1:]-z[:-1]
    dy=constants.Rearth*x[...].cellsize()[1]

    psib=barotropic_streamfunction(x[...].u_velocity | 0.1*units.m/units.s,dz,dy)

    vmax=numpy.abs(psib).max().value_in(units.Sv)

    im=ax[0].imshow(psib.value_in(units.Sv).T, origin="lower", cmap="seismic", vmax=vmax, vmin=-vmax, extent=[xmin,xmax,ymin,ymax], interpolation="none")
    fig.colorbar(im,ax=ax[0],label="psib [Sv]")

    dx=constants.Rearth*x[...].cellsize()[0]*numpy.cos(x[0,:,0].lat.value_in(units.rad))

    psim=overturning_streamfunction(x[...].v_velocity | 0.1*units.m/units.s,dz,dx)

    instance.stop()
    print("psim")
    print(psim[:,-1])

    vmax=numpy.abs(psim).max().number#value_in(units.Sv)
    zmin=z.min().value_in(units.m)
    zmax=z.max().value_in(units.m)

    im=ax[1].imshow(psim.number.T, origin="lower", cmap="seismic", vmax=vmax, vmin=-vmax, extent=[ymin,ymax,zmin,zmax], interpolation="none", aspect="auto")
    fig.colorbar(im,ax=ax[1],label="psim [Sv]")


    pyplot.savefig("psib_psim.png")

    pyplot.show()


    print("done")
