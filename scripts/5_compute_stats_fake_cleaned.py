import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline
import CMBanom

# Parameters                                                                                                              
Nsims     = 10000
Nside_in  = 128
label_sim = "143GHz"
maps_dir  = "/tank/NoBackup/hnofi/sim_maps/LCDM/cleaned143GHz/143GHz_LCDM_"
#maps_dir  = "/tank/NoBackup/hnofi/sim_maps/LCDM/pureCMB/pureCMB_LCDM_" 
corrs_dir = "/tank/NoBackup/lherold/cleaned_sims_test/"
cls_dir   = "/tank/NoBackup/lherold/cleaned_sims_test/"
masks_dir = "../data/masks/"
stats_dir = "../data/stats/"
names_mask = ["fullsky"]
mask_files = []
#["1percent_mask_v9.fits", "com_mask_cutoff_0.9_nside_128.fits"]
Nmasks     = len(names_mask)
fullsky = True

# Modes
compute_envelopes = True
compute_Smu       = True
compute_R         = True
compute_sigma16   = True
compute_SQO       = True
compute_ALV       = True

## Cl's and corr's function
percentiles = True

## Low correlation, Smu
mus    = np.linspace(-1, 1, 41)

# Parity asymmetry, R
lmax_R = 60

# Low northern variance, sigma16
ecliptic_coords = True
if compute_sigma16:
    mask_dir_south_ecl = "mask_south_ecl_nside_16.fits"
    mask_files = []
    #["stdv_mask_1percent_cutoff_0.9_nside_16.fits", "com_mask_cutoff_0.9_nside_16.fits"]

# Hemispherical asymmetry, ALV
theta_deg = 8
frac_to_be_masked = 0.1

    
if compute_Smu:
    print("Computing Smu:")
    for m in range(Nmasks):
        name_mask = names_mask[m]
        print(name_mask, "...")
        
        # Load corrs
        theta, cos_theta, corrs = CMBanom.load_corrs(corrs_dir, name_mask, Nsims, name_corr=label_sim+"_LCDM_corr")

        # Compute & save Smu
        S_mu = np.array([CMBanom.S_mu_many(1e6*corrs, cos_theta, mu, method='summation') for mu in mus])
        np.savetxt(stats_dir+f'Smu_sims_{label_sim}_{name_mask}_Nsims_{Nsims}.npy', S_mu)

        
if compute_R:
    print("Computing R:")
    for m in range(Nmasks):
        name_mask = names_mask[m]
        print(name_mask, "...")

        # Load Cls (correcting for pixel window fct. & beam)
        cl_wf_factor = CMBanom.get_cl_wf_factor(Nside_in)
        cls = CMBanom.load_cls(cls_dir, name_mask, Nsims, cl_wf_factor, name_cl=label_sim+"_LCDM_cl")

        # Compute and save R
        R = np.array([[CMBanom.get_Rlmax(cls[n], lmax=l) for l in range(lmax_R)] for n in range(Nsims)])
        np.savetxt(stats_dir+f'R_sims_{label_sim}_{name_mask}_Nsims_{Nsims}.npy', R)

        
if compute_sigma16:
    Nside_out = 16
    len_mask_16 = 3072
    print("Computing sigma_16:")

    # Load masks and convert zeros to nans
    if ecliptic_coords: mask_for_north = hp.read_map(masks_dir+mask_dir_south_ecl)
    else: mask_for_north = np.append(np.ones(len_mask_16/2), np.zeros(len_mask_16/2))
    masks_01 = CMBanom.read_masks(masks_dir, mask_files, Nside_out, fullsky=fullsky)
    masks = np.where(np.array([mask*mask_for_north for mask in masks_01])==0, np.nan, 1)

    # Read maps                                                                                                               
    print("Downgrading maps")
    maps_128 = [hp.read_map(maps_dir+f"{n:05}.fits") for n in range(Nsims)]
    maps_16  = [CMBanom.downgrade_map(map, Nside_out) for map in maps_128]
                                                                                                     
    print("Computing sigma^2_16")
    for m in range(Nmasks):
        mask = masks[m]
        name_mask = names_mask[m]
        print("-", name_mask, "...")

        # Compute & save sigma_16
        sigma16 = [CMBanom.sigma2_16(map*1.e3, mask) for map in maps_16]
        np.savetxt(stats_dir+f'sigma16_sims_{label_sim}_{name_mask}_Nsims_{Nsims}.npy', sigma16)

        
if compute_SQO:
    lmax_QO = 3
    print("Computing SQO:")

    # Load masks
    masks = CMBanom.read_masks(masks_dir, mask_files, Nside_in, fullsky=fullsky)

    # Load maps                                                                                                                  
    maps = [hp.read_map(maps_dir+f"{n:05}.fits") for n in range(Nsims)]
        
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
        np.savetxt(stats_dir+f'SQO_sims_{label_sim}_{name_mask}_Nsims_{Nsims}.npy', SQO)


if compute_ALV:
    Nside_in  = 128
    Nside_out = 16
    Npix_in  = hp.nside2npix(Nside_in)
    Npix_out = hp.nside2npix(Nside_out)
    print("Computing ALV:")
    
    # Load maps and masks
    masks = CMBanom.read_masks(masks_dir, mask_files, Nside_in, fullsky=fullsky)
    maps  = [hp.read_map(maps_dir+f"{n:05}.fits") for n in range(Nsims)]

    for m in range(Nmasks):
        mask = masks[m]
        name_mask = names_mask[m]
        print(name_mask, "...")

        print("- Computing pixlist and lvmask")
        pixlist = CMBanom.get_pixlist(theta_deg, mask, Nside_in, Nside_out)
        lvmask  = CMBanom.get_lvmask(pixlist, theta_deg, frac_to_be_masked, Nside_in, Nside_out)

        print("- Computing LV maps")
        lvmaps = np.array([CMBanom.get_lvmap(maps[n], mask, pixlist, Nside_out) for n in range(Nsims)])

        print("- Compute mean and var of lvmaps")
        mean_lvmap = CMBanom.get_meanlvmap(lvmaps, lvmask, f"{stats_dir}meanlvmap_{label_sim}_{name_mask}_Nsims_{Nsims}.npy")
        var_lvmap  = CMBanom.get_varlvmap(lvmaps, lvmask, mean_lvmap, f"{stats_dir}varlvmap_{label_sim}_{name_mask}_Nsims_{Nsims}.npy")
        
        print("- Computing ALV")
        ALV = np.array([CMBanom.ALV_vec(lvmaps[n], lvmask, mean_lvmap, var_lvmap)[0] for n in range(Nsims)])
            
        np.savetxt(stats_dir+f'ALV_sims_{label_sim}_{name_mask}_Nsims_{Nsims}.npy', ALV)
    
# Compute Cl and corr envelopes
if compute_envelopes:
    print("Computing envelopes:")
    for m in range(len(names_mask)):
        name_mask = names_mask[m]
        print(name_mask, "...")
        
        # Load corrs
        theta, cos_theta, corrs = CMBanom.load_corrs(corrs_dir, name_mask, Nsims, name_corr=label_sim+"_LCDM_corr")
        corrs = 1e6*corrs
        
        # Save corr envelope
        mean_corr = np.mean(corrs, axis=0)
        if percentiles:
            perc_68_lower = np.percentile(corrs, (100-68.27)/2, axis=0)
            perc_68_upper = np.percentile(corrs, 68.27+(100-68.27)/2, axis=0)
            perc_95_lower = np.percentile(corrs, (100-95.45)/2, axis=0)
            perc_95_upper = np.percentile(corrs, 95.45+(100-95.45)/2, axis=0)
            np.savetxt(stats_dir+f"corr_{label_sim}_mean_perc_{name_mask}.npy", np.array([mean_corr, perc_68_lower, perc_68_upper, perc_95_lower, perc_95_upper]))
        else:
            std_corr = np.std(corrs, axis=0)
            np.savetxt(stats_dir+f"corr_mean_{label_sim}_std_{name_mask}.npy", np.array([mean_corr, std_corr]))            

        # Load Cls and correct for window function
        cl_wf_factor = CMBanom.get_cl_wf_factor(Nside_in)
        cls = CMBanom.load_cls(cls_dir, name_mask, Nsims, cl_wf_factor, name_cl=label_sim+"_LCDM_cl")

        # Save Cl envelope
        mean_cls = np.mean(cls, axis=0)
        if percentiles:
            perc_68_lower = np.percentile(cls, (100-68.27)/2, axis=0)
            perc_68_upper = np.percentile(cls, 68.27+(100-68.27)/2, axis=0)
            perc_95_lower = np.percentile(cls, (100-95.45)/2, axis=0)
            perc_95_upper = np.percentile(cls, 95.45+(100-95.45)/2, axis=0)
            np.savetxt(stats_dir+f"cls_{label_sim}_mean_perc_{name_mask}.npy", np.array([mean_cls, perc_68_lower, perc_68_upper, perc_95_lower, perc_95_upper]))
        else:
            std_cls = np.std(cls, axis=0)
            np.savetxt(stats_dir+"fcls_{label_sim}_mean_std_"+name_mask+".npy", np.array([mean_cls, std_cls]))
