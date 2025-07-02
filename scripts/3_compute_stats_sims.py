import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline
import CMBanom

# Parameters                                                                                                              
Nsims     = 100000
Nside_in  = 128
maps_dir  = "/tank/NoBackup/lherold/maps_100k/" 
corrs_dir = "/tank/NoBackup/lherold/"
cls_dir   = "/tank/NoBackup/lherold/"
masks_dir = "../data/masks/"
stats_dir = "../data/stats/"
names_mask = ["fullsky", "stdmask", "commask"]
mask_files = ["stdv_mask_1percent_v7.fits", "com_mask_cutoff_0.9_nside_128.fits"]
Nmasks     = len(names_mask)

# Modes
compute_Smu       = False
compute_R         = False
compute_sigma16   = False
compute_SQO       = True
compute_envelopes = False
compute_ALV       = False

## Cl's and corr's function
percentiles = False

## Low correlation, Smu
summation = True
mu = 0.5

# Parity asymmetry, R
lmax_R = 60

# Low northern variance, sigma16
ecliptic_coords = True
if compute_sigma16:
    mask_dir_south_ecl = "mask_south_ecl_nside_16.fits"
    mask_files = ["stdv_mask_1percent_cutoff_0.9_nside_16.fits", "com_mask_cutoff_0.9_nside_16.fits"]

# Hemispherical asymmetry, ALV
theta_deg = 8
frac_to_be_masked = 0.1

    
if compute_Smu:
    print("Computing Smu:")
    for m in range(Nmasks):
        name_mask = names_mask[m]
        print(name_mask, "...")
        
        # Load corrs
        theta, cos_theta, corrs = CMBanom.load_corrs(corrs_dir+f"corrs_{name_mask}_100k/", name_mask, Nsims)

        # Compute & save Smu
        S_mu = CMBanom.S_mu_many(corrs, cos_theta, mu, summation=summation)
        np.savetxt(stats_dir+f'Smu_sims_{name_mask}_Nsims_{Nsims}.npy', S_mu)

        
if compute_R:
    print("Computing R:")
    for m in range(Nmasks):
        name_mask = names_mask[m]
        print(name_mask, "...")

        # Load Cls (correcting for pixel window fct. & beam)
        cl_wf_factor = CMBanom.get_cl_wf_factor(Nside_in)
        cls = CMBanom.load_cls(cls_dir+f"cls_{name_mask}_100k/", name_mask, Nsims, cl_wf_factor)

        # Compute and save R
        R = np.array([[CMBanom.get_Rassymstat(cls[n], lmax=l) for l in range(lmax_R)] for n in range(Nsims)])
        np.savetxt(stats_dir+f'R_sims_{name_mask}_Nsims_{Nsims}.npy', R)

        
if compute_sigma16:
    Nside_out = 16
    len_mask_16 = 3072
    print("Computing sigma_16:")

    # Load masks and convert zeros to nans
    if ecliptic_coords: mask_for_north = hp.read_map(masks_dir+mask_dir_south_ecl)
    else: mask_for_north = np.append(np.ones(len_mask_16/2), np.zeros(len_mask_16/2))
    masks_01 = CMBanom.read_masks(masks_dir, mask_files, Nside_out)
    masks = np.where(np.array([mask*mask_for_north for mask in masks_01])==0, np.nan, 1)

    # Read maps                                                                                                               
    print("Downgrading maps")
    maps_128 = [hp.read_map(maps_dir+f"map__{n}.fits") for n in range(Nsims)]
    maps_16  = [CMBanom.downgrade_map(map, Nside_out) for map in maps_128]
                                                                                                     
    print("Computing sigma_16")
    for m in range(Nmasks):
        mask = masks[m]
        name_mask = names_mask[m]
        print("-", name_mask, "...")

        # Compute & save sigma_16
        sigma16 = [CMBanom.sigma_16(map, mask) for map in maps_16]
        np.savetxt(stats_dir+f'sigma16_sims_{name_mask}_Nsims_{Nsims}.npy', sigma16)

        
if compute_SQO:
    lmax_QO = 3
    print("Computing SQO:")

    # Load masks
    masks = CMBanom.read_masks(masks_dir, mask_files, Nside_in)

    # Load maps                                                                                                                  
    maps = [hp.read_map(maps_dir+f"map__{n}.fits")*1e3 for n in range(Nsims)]
        
    for m in range(Nmasks):
        mask = masks[m]
        name_mask = names_mask[m]
        print(name_mask, "...")

        print("- Computing multipole vectors")
        mvs = CMBanom.compute_MVs(maps, mask, lmax_QO)
    
        print("- Computing oriented-area vectors")
        ws = CMBanom.compute_Ws(mvs, lmax_QO)

        print("- Computing SQO")
        SQO = np.array([CMBanom.S_QO(ws[n]) for n in range(Nsims)])
        np.savetxt(stats_dir+f'SQO_sims_{name_mask}_Nsims_{Nsims}.npy', SQO)


if compute_ALV:
    Nside_in  = 128
    Nside_out = 16
    Npix_in  = hp.nside2npix(Nside_in)
    Npix_out = hp.nside2npix(Nside_out)
    print("Computing ALV:")
    
    # Load maps and masks
    masks = CMBanom.read_masks(masks_dir, mask_files, Nside_in)
    maps  = [hp.read_map(maps_dir+f"map__{n}.fits") for n in range(Nsims)]

    for m in range(Nmasks):
        mask = masks[m]
        name_mask = names_mask[m]
        print(name_mask, "...")

        print("- Computing pixlists and lvmasks")
        pixlist = []
        pixlist = CMBanom.get_pixlist(theta_deg, mask, Nside_in, Nside_out)
        lvmask  = CMBanom.get_lvmask(pixlist, theta_deg, frac_to_be_masked, Nside_in, Nside_out)
        lvmask_nan = np.where(lvmask==0, np.nan, 1.)

        print("- Computing LV maps")
        lvmaps = np.zeros((Nsims, Npix_out))
        for n in range(Nsims):
            lvmaps[n] = CMBanom.get_lvmap(maps[n], mask, pixlist, Nside_out)

        print("- Computing ALV")
        ALV = np.zeros(Nsims)
        for n in range(Nsims):
            ALV[n] = CMBanom.ALV(lvmaps[n], lvmaps, lvmask_nan)
            
        np.savetxt(stats_dir+f'ALV_sims_{name_mask}_Nsims_{Nsims}.npy', ALV)
    
# Compute Cl and corr envelopes
if compute_envelopes:
    for m in range(len(names_mask)):
        name_mask = names_mask[m]
        print(name_mask, "...")
        
        # Load corrs
        theta, cos_theta, corrs = CMBanom.load_corrs(corrs_dir+f"corrs_{name_mask}_100k/", name_mask, Nsims)

        # Save corr envelope
        mean_corr = np.mean(corrs, axis=0)
        if percentiles:
            perc_68_lower = np.percentile(corrs, (100-68.27)/2, axis=0)
            perc_68_upper = np.percentile(corrs, 68.27+(100-68.27)/2, axis=0)
            perc_95_lower = np.percentile(corrs, (100-95.45)/2, axis=0)
            perc_95_upper = np.percentile(corrs, 95.45+(100-95.45)/2, axis=0)
            np.savetxt(stats_dir+"corr_mean_perc_"+name_mask+".npy", np.array([mean_corr, perc_68_lower, perc_68_upper, perc_95_lower, perc_95_upper]))
        else:
            std_corr = np.std(corrs, axis=0)
            np.savetxt(stats_dir+"corr_mean_std_"+name_mask+".npy", np.array([mean_corr, std_corr]))            

        # Load Cls and correct for window function
        cl_wf_factor = CMBanom.get_cl_wf_factor(Nside_in)
        cls = CMBanom.load_cls(cls_dir+f"cls_{name_mask}_100k/", name_mask, Nsims, cl_wf_factor)

        # Save Cl envelope
        mean_cls = np.mean(cls, axis=0)
        if percentiles:
            perc_68_lower = np.percentile(cls, (100-68.27)/2, axis=0)
            perc_68_upper = np.percentile(cls, 68.27+(100-68.27)/2, axis=0)
            perc_95_lower = np.percentile(cls, (100-95.45)/2, axis=0)
            perc_95_upper = np.percentile(cls, 95.45+(100-95.45)/2, axis=0)
            np.savetxt(stats_dir+"cls_mean_perc_"+name_mask+".npy", np.array([mean_cls, perc_68_lower, perc_68_upper, perc_95_lower, perc_95_upper]))
        else:
            std_cls = np.std(cls, axis=0)
            np.savetxt(stats_dir+"cls_mean_std_"+name_mask+".npy", np.array([mean_cls, std_cls]))
