import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline
import CMBanom

# Parameters                                                                                                              
Nsims    = 10000
Nside_in = 128
fn_maps  = "/tank/NoBackup/lherold/sims/" 
fn_corrs = "/tank/NoBackup/lherold/sims/"
fn_cls   = "/tank/NoBackup/lherold/sims/"
fn_masks = "../data/masks/"
fn_stats = "../data/stats/"
names_mask = ["fullsky", "stdmask", "commask"]
mask_files = [None, "stdv_mask_1percent_cutoff0.9_Nside16.fits", "common-Mask-Int_cutoff0.9_Nside16.fits"]

## Cl's and corr's function
compute_envelopes = False

## Low correlation, Smu
compute_Smu = False
summation = True
mu = 0.5

# Parity asymmetry, R
compute_R = False
lmax_R = 60

# Hemispherical asymmetry, sigma16
compute_sigma16 = False
ecliptic_coords = True
mask_fn_south_ecl = "mask_south_ecl_Nside16.fits"

# Quadrupole-octopole alignment, SQO
compute_SQO = False


if compute_Smu:
    print("Computing Smu:")
    for m in range(len(names_mask)):
        name_mask = names_mask[m]
        print(name_mask, "...")

        # Load corrs
        theta, cos_theta, corrs = CMBanom.load_corrs(fn_corrs, name_mask, Nsims)

        # Compute & save Smu
        S_mu = CMBanom.S_mu_many(corrs, cos_theta, mu, summation=summation)
        np.savetxt(fn_stats+f'Smu_sims_{name_mask}_Nsims_{Nsims}.npy', S_mu)

        
if compute_R:
    print("Computing R:")
    for m in range(len(names_mask)):
        name_mask = names_mask[m]
        print(name_mask, "...")

        # Load Cls (correcting for pixel window fct. & beam)
        cl_wf_factor = CMBanom.get_cl_wf_factor(Nside_in)
        cls = CMBanom.load_cls(fn_cls, name_mask, Nsims, cl_wf_factor)

        # Compute and save R
        R = np.array([[CMBanom.get_Rassymstat(cls[n], lmax=l) for l in range(lmax_R)] for n in range(Nsims)])
        np.savetxt(fn_stats+f'R_sims_{name_mask}_Nsims_{Nsims}.npy', R)

        
if compute_sigma16:
    Nside_out = 16
    print("Computing sigma_16:")
    for m in range(len(names_mask)):
        name_mask = names_mask[m]
        print(name_mask, "...")

        # Load masks
        mask_com = np.where(hp.read_map(masks_dir+mask_fns[2])==0, np.NaN, 1)
        mask_std = np.where(hp.read_map(masks_dir+mask_fns[1])==0, np.NaN, 1)
        if ecliptic_coords:
            mask_for_north = np.where(hp.read_map(masks_dir+mask_fn_south_ecl)==0, np.NaN, 1)
            mask_for_south = np.where(hp.read_map(masks_dir+mask_fn_south_ecl)==1, np.NaN, 1)
        else:
            mask_for_north = np.append(np.ones(len(mask_com)), np.full(len(mask_com), np.NaN))
            mask_for_north = np.append(np.full(len(mask_com), np.NaN), np.ones(len(mask_com)))
    
        # Read maps
        

if compute_SQO:
    for m in range(len(names_mask)):
        name_mask = names_mask[m]
        print(name_mask, "...")

        # ...
        
# Compute Cl and corr envelopes
if compute_envelopes:
    for m in range(len(names_mask)):
        name_mask = names_mask[m]
        print(name_mask, "...")
        
        # Load corrs
        theta, cos_theta, corrs = CMBanom.load_corrs(fn_corrs, name_mask, Nsims)

        # Save corr envelope
        mean_corr = np.mean(corrs, axis=0)
        std_corr = np.std(corrs, axis=0)
        np.savetxt(fn_stats+"corr_mean_std_"+name_mask+".npy", np.array([mean_corr, std_corr]))

        # Load Cls and correct for window function
        cl_wf_factor = CMBanom.get_cl_wf_factor(Nside_in)
        cls = CMBanom.load_cls(fn_cls, name_mask, Nsims, cl_wf_factor)

        # Save Cl envelope
        mean_cls = np.mean(cls, axis=0)
        std_cls = np.std(cls, axis=0)
        np.savetxt(fn_stats+"cls_mean_std_"+name_mask+".npy", np.array([mean_cls, std_cls]))
