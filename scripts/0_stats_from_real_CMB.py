import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline
import healpy as hp
import CMBanom

# Parameters
mu = 0.5
compute_cl_corr = True
compute_Smu = False
summation = True
save_downgrade = False
fn_real = "../data/real/"
fn_Smu = "../data/stats/"
fn_mask = "../data/masks/"

# full sky, common mask, std. mask

mask_labels = ["fullsky", "commask", "stdmask"]
names_mask = [None, "common-Mask-Int_cutoff0.9_Nside128.fits", "stdv_mask_1percent_cutoff0.9_Nside128.fits"]

# Planck maps full resolution
#fn_maps = "/tank/data/planck/"
#names_map = ["planck_pr3/COM_CMB_IQU-smica_2048_R3.00_full.fits", "planck_pr3/COM_CMB_IQU-commander_2048_R3.00_full.fits", "planck_pr3/COM_CMB_IQU-nilc_2048_R3.00_full.fits", "planck_pr3/COM_CMB_IQU-sevem_2048_R3.01_full.fits", "planck_pr2/COM_CMB_IQU-commander_1024_R2.02_full.fits", "planck_pr2/COM_CMB_IQU-nilc-field-Int_2048_R2.01_full.fits", "planck_pr2/COM_CMB_IQU-sevem-field-Int_2048_R2.01_full.fits", "planck_pr2/COM_CMB_IQU-smica-field-Int_2048_R2.00.fits"]
#labels = ["smica_2048", "commander_2048", "nilc_2048", "sevem_2048", "commander_pr2_2048", "nilc_pr2_2048", "sevem_pr2_2048", "smica_pr2_2048"]

# All maps Nside=128
fn_maps = "../data/maps/"
names_map = ["map_commander_nside_128.fits", "map_nilc_nside_128.fits", "map_sevem_nside_128.fits", "map_smica_nside_128.fits", "map_cleaned_70GHz_v3.fits", "map_cleaned_94GHz_v3.fits", "map_cleaned_100GHz_v3.fits", "map_cleaned_143GHz_v3.fits"]
map_labels = ["commander", "nilc", "sevem", "smica", "70GHz", "94GHz", "100GHz", "143GHz"]

for i in range(len(names_map)):
    name_map = names_map[i]

    for j in range(len(names_mask)):
        name_mask = names_mask[j]

        if compute_cl_corr:
            os.system('echo "Computing power spectra and correlation functions"')
            name_corr = 'corr_'+map_labels[i]+'_'+mask_labels[j]+'.txt'
            name_pcl = 'cl_'+map_labels[i]+'_'+mask_labels[j]+'.txt'
        
            if name_mask==None:
                os.system('spice -mapfile '+fn_real+name_map+'  -corfile '+fn_real+name_corr+' -clfile '+fn_real+name_pcl)
            else:
                os.system('spice -mapfile '+fn_real+name_map+' -maskfile '+fn_mask+name_mask+'  -corfile '+fn_real+name_corr+' -clfile '+fn_real+name_pcl)

        if compute_Smu:
            os.system('echo "Computing Smu from real maps"')
    
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
            if mask:
                np.savetxt(fn_Smu+"Smu_"+mask_label+"_"+label+".npy", np.array([S_mu]))
                print("Saved Smu for "+label+" as "+fn_Smu+"Smu_"+mask_label+"_"+label+".npy")
            else:
                np.savetxt(fn_Smu+"Smu_"+label+".npy", np.array([S_mu]))
                print("Saved Smu for "+label+" as "+fn_Smu+"Smu_"+label+".npy")


# Save downgraded maps                                                                                                           
if save_downgrade == True:
    for i in range(len(names_map)):
        highres_map = hp.read_map(fn_maps+names_map[i])
        lowres_map = CMBanom.downgrade_map(highres_map, NSIDEout=128)
        hp.write_map(fn_real+labels[i]+"_nside_128.fits", lowres_map)
    exit()
