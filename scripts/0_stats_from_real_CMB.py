import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline
import healpy as hp
import CMBanom

# Parameters
mu = 0.5
summation = True
save_Smu = False
mask = True
save_downgrade = False
fn_corr = "../data/real/"
fn_cls = "../data/real/"
fn_Smu = "../data/Smu/"

# mask files
fn_mask = "../data/masks/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
mask_label = "commask"
#fn_mask = "/home/lherold/data/masks/Pessimistic_common-Mask-Int_Nside_128.fits"
#mask_label = "pess_common_mask"
#fn_mask = "/home/lherold/data/masks/Cutoff_09_common-Mask-Int_Nside_128.fits"
#mask_label = "09mask"

# Planck maps
fn_maps = "/tank/data/planck/"
names_map = ["planck_pr3/COM_CMB_IQU-smica_2048_R3.00_full.fits", "planck_pr3/COM_CMB_IQU-commander_2048_R3.00_full.fits", "planck_pr3/COM_CMB_IQU-nilc_2048_R3.00_full.fits", "planck_pr3/COM_CMB_IQU-sevem_2048_R3.01_full.fits", "planck_pr2/COM_CMB_IQU-commander_1024_R2.02_full.fits", "planck_pr2/COM_CMB_IQU-nilc-field-Int_2048_R2.01_full.fits", "planck_pr2/COM_CMB_IQU-sevem-field-Int_2048_R2.01_full.fits", "planck_pr2/COM_CMB_IQU-smica-field-Int_2048_R2.00.fits"]
labels = ["smica_2048", "commander_2048", "nilc_2048", "sevem_2048", "commander_pr2_2048", "nilc_pr2_2048", "sevem_pr2_2048", "smica_pr2_2048"]

# Planck maps downgraded
#fn_maps = "../data/maps/"
#names_map = ["commander_pr3_nside_128.fits", "nilc_pr3_nside_128.fits", "sevem_pr3_nside_128.fits", "smica_pr3_nside_128.fits"]
#labels = ["commander_128", "nilc_128", "sevem_128", "smica_128"]

# Hayley's maps
#fn_maps = "/tank/data/cmb_anom/hayleys_maps/"
#names_map = ["cleaned_100GHz_v1.fits", "cleaned_100GHz_v2.fits", "cleaned_143GHz_v1.fits", "cleaned_143GHz_v2.fits", "cleaned_70GHz_v1.fits", "cleaned_70GHz_v2.fits", "cleaned_94GHz_v1.fits", "cleaned_94GHz_v2.fits"]
#labels = ["cleaned_100GHz_v1", "cleaned_100GHz_v2", "cleaned_143GHz_v1", "cleaned_143GHz_v2", "cleaned_70GHz_v1", "cleaned_70GHz_v2", "cleaned_94GHz_v1", "cleaned_94GHz_v2"]

# Save downgraded maps 
if save_downgrade == True:
    for i in range(len(names_map)):
        highres_map = hp.read_map(fn_maps+names_map[i])
        lowres_map = CMBanom.downgrade_map(highres_map, NSIDEout=128)
        hp.write_map("../data/maps/"+labels[i]+"_nside_128.fits", lowres_map)
    exit()

os.system('echo "Computing Smu from real maps"')
for i in range(len(names_map)):
    name_map = names_map[i]
    label = labels[i]

    # Compute C(\theta)
    if mask:
        name_corr = 'corr_'+mask_label+"_"+label+'.txt'
        name_pcl = 'cl_'+mask_label+"_"+label+'.txt'
        os.system('spice -mapfile '+fn_maps+name_map+' -maskfile '+fn_mask+'  -corfile '+fn_corr+name_corr+' -clfile '+fn_cls+name_pcl)
    else:
        name_corr = 'corr_'+label+'.txt'
        name_pcl = 'cl_'+label+'.txt'
        os.system('spice -mapfile '+fn_maps+name_map+'  -corfile '+fn_corr+name_corr+' -clfile '+fn_cls+name_pcl)

    # Read C(\theta)'s and compute Smu
    #C_theta = 1e12*np.loadtxt(fn_corr+name_corr).T[2]
    C_theta = 1e6*np.loadtxt(fn_corr+name_corr).T[2] 
    theta = np.loadtxt(fn_corr+name_corr).T[0]
    cos_theta = np.loadtxt(fn_corr+name_corr).T[1]
    dtheta = np.append(theta[:-1] - theta[1:], np.zeros(1))
    dcos_theta = np.append(cos_theta[1:] - cos_theta[:-1], np.zeros(1))
    
    if summation:
        index_mu = np.nonzero(cos_theta>mu)[0][0]-1
        C_theta_mu = np.zeros(len(C_theta))
        C_theta_mu[:index_mu] = C_theta[:index_mu]
        S_mu = np.sum(C_theta_mu**2*dcos_theta)

    else:
        C_theta_int = UnivariateSpline(cos_theta, C_theta**2)
        S_mu = integrate.quad(C_theta_int, -1, mu)[0]

    # Save Smu                          
    if save_Smu:
        if mask:
            np.savetxt(fn_Smu+"Smu_"+mask_label+"_"+label+".npy", np.array([S_mu]))
            print("Saved Smu for "+label+" as "+fn_Smu+"Smu_"+mask_label+"_"+label+".npy")
        else:
            np.savetxt(fn_Smu+"Smu_"+label+".npy", np.array([S_mu]))
            print("Saved Smu for "+label+" as "+fn_Smu+"Smu_"+label+".npy")
