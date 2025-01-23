import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline
import CMBanom

# Parameters                                                                                                              
Nsims     = 10000
Nside_in  = 128
maps_dir  = "/tank/NoBackup/lherold/sims/" 
corrs_dir = "/tank/NoBackup/lherold/sims/"
cls_dir   = "/tank/NoBackup/lherold/sims/"
masks_dir = "../data/masks/"
stats_dir = "../data/stats/"
names_mask = ["fullsky", "stdmask", "commask"]
mask_files = ["stdv_mask_1percent_cutoff0.9_Nside16.fits", "common-Mask-Int_cutoff0.9_Nside16.fits"]

## Cl's and corr's function
compute_envelopes = True

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
mask_dir_south_ecl = "mask_south_ecl_Nside16.fits"

# Quadrupole-octopole alignment, SQO
compute_SQO = False


if compute_Smu:
    print("Computing Smu:")
    for m in range(len(names_mask)):
        name_mask = names_mask[m]
        print(name_mask, "...")

        # Load corrs
        theta, cos_theta, corrs = CMBanom.load_corrs(corrs_dir, name_mask, Nsims)

        # Compute & save Smu
        S_mu = CMBanom.S_mu_many(corrs, cos_theta, mu, summation=summation)
        np.savetxt(stats_dir+f'Smu_sims_{name_mask}_Nsims_{Nsims}.npy', S_mu)

        
if compute_R:
    print("Computing R:")
    for m in range(len(names_mask)):
        name_mask = names_mask[m]
        print(name_mask, "...")

        # Load Cls (correcting for pixel window fct. & beam)
        cl_wf_factor = CMBanom.get_cl_wf_factor(Nside_in)
        cls = CMBanom.load_cls(cls_dir, name_mask, Nsims, cl_wf_factor)

        # Compute and save R
        R = np.array([[CMBanom.get_Rassymstat(cls[n], lmax=l) for l in range(lmax_R)] for n in range(Nsims)])
        np.savetxt(stats_dir+f'R_sims_{name_mask}_Nsims_{Nsims}.npy', R)

        
if compute_sigma16:
    Nside_out = 16
    len_mask_16 = 3072
    print("Computing sigma_16:")

    # Load masks and convert zeros to NaNs
    if ecliptic_coords: mask_for_north = hp.read_map(masks_dir+mask_dir_south_ecl)
    else: mask_for_north = np.append(np.ones(len_mask_16/2), np.zeros(len_mask_16/2))
    masks_01 = CMBanom.read_masks(masks_dir, mask_files, Nside_out)
    masks = np.where(np.array([mask*mask_for_north for mask in masks_01]), np.NaN, 1)
        
    for m in range(len(names_mask)):
        mask = masks[m]
        name_mask = names_mask[m]
        print(name_mask, "...")

        # Read maps
        maps_128 = [hp.read_map(sims_dir+f"map__{n}.fits") for n in range(Nsims)]
        maps_16  = [CMBanom.downgrade_map(map, Nside_out) for map in maps_128]

        # Compute & save sigma_16
        sigma16 = [CMBanom.sigma_16(map, mask) for map in maps_16]
        np.savetxt(stats_dir+f'sigma16_sims_{name_mask}_Nsims_{Nsims}.npy', sigma16)

        
if compute_SQO:
    lmax_QO = 4
    print("Computing SQO:")

    # Load masks
    masks = CMBanom.read_masks(masks_dir, mask_files, Nside_in)

     # Load maps                                                                                                                  
     maps = [hp.read_map(map_dir+f"map__{n}.fits")*1e3 for n in range(Nsims)]
        
    for m in range(len(names_mask)):
        mask = masks[m]
        name_mask = names_mask[m]
        print(name_mask, "...")

        print("Compute multipole vectors...")
        mvs = CMBanom.compute_MVs(maps_sims, mask, lmax)
    
        print("Compute oriented-area vectors...")
        ws = CMBanom.compute_Ws(mvs, lmax)

        print("Compute SQO...")
        SQO = np.array([CMBanom.S_QO(ws[n]) for n in range(Nsims)])
        np.savetxt(stats_dir+f'SQO_sims_{name_mask}_Nsims_{Nsims}.npy', SQO)
        
# Compute Cl and corr envelopes
if compute_envelopes:
    for m in range(len(names_mask)):
        name_mask = names_mask[m]
        print(name_mask, "...")
        
        # Load corrs
        theta, cos_theta, corrs = CMBanom.load_corrs(corrs_dir, name_mask, Nsims)

        # Save corr envelope
        mean_corr = np.mean(corrs, axis=0)
        std_corr = np.std(corrs, axis=0)
        np.savetxt(stats_dir+"corr_mean_std_"+name_mask+".npy", np.array([mean_corr, std_corr]))

        # Load Cls and correct for window function
        cl_wf_factor = CMBanom.get_cl_wf_factor(Nside_in)
        cls = CMBanom.load_cls(cls_dir, name_mask, Nsims, cl_wf_factor)

        # Save Cl envelope
        mean_cls = np.mean(cls, axis=0)
        std_cls = np.std(cls, axis=0)
        np.savetxt(stats_dir+"cls_mean_std_"+name_mask+".npy", np.array([mean_cls, std_cls]))
