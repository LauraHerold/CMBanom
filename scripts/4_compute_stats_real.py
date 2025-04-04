import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline
import CMBanom

# Parameters
Nside_in  = 128
real_dir  = "../data/real/"
masks_dir = "../data/masks/"
stats_dir = "../data/stats/"
names_mask = ["fullsky", "stdmask", "commask"]
mask_files = ["stdv_mask_1percent_v4.fits", "common-Mask-Int_cutoff0.9_Nside128.fits"]
names_maps = ["commander_nside_128", "nilc_nside_128", "sevem_nside_128", "smica_nside_128", "cleaned_70GHz_v4", "cleaned_94GHz_v4", "cleaned_100GHz_v4", "cleaned_143GHz_v4"]
names_real = ["commander", "nilc", "sevem", "smica", "v4_70GHz", "v4_94GHz", "v4_100GHz", "v4_143GHz"] 
Nmasks = len(names_mask)
Nmaps = len(names_maps)

# Modes
compute_Smu       = True 
compute_R         = False #not implemented
compute_sigma16   = False
compute_SQO       = False #not implemented

## Low correlation, Smu
summation = True
mu = 0.5

# Parity asymmetry, R
lmax_R = 60

# Hemispherical asymmetry, sigma16
ecliptic_coords = True

if compute_sigma16:
    mask_dir_south_ecl = "mask_south_ecl_Nside16.fits"
    mask_files = ["stdv_mask_1percent_cutoff0.9_Nside16.fits", "common-Mask-Int_cutoff0.9_Nside16.fits"]

if compute_Smu:
    print("Computing Smu:")
    for m in range(len(names_mask)):
        name_mask = names_mask[m]
        print(name_mask, "...")

        # Load corrs
        #theta, cos_theta, corrs = CMBanom.load_corrs(corrs_dir+f"corrs_{name_mask}_100k/", name_mask, Nsims)
        theta, cos_theta = np.loadtxt(real_dir+f"corr_{names_real[0]}_{name_mask}.txt").T[:2]
        corrs = np.array([np.loadtxt(real_dir+f"corr_{names_real[n]}_{name_mask}.txt").T[2] for n in range(Nmaps)])
        corrs[4:] *= 1e6

        # Compute & save Smu
        S_mu = CMBanom.S_mu_many(corrs, cos_theta, mu, summation=summation)
        np.savetxt(stats_dir+f'Smu_real_{name_mask}.npy', S_mu)

        
if compute_R:
    print("Computing R:")
    for m in range(len(names_mask)):
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
    masks = np.where(np.array([mask*mask_for_north for mask in masks_01]), np.nan, 1)

    # Read maps
    print("Downgrading maps")
    maps_128 = [hp.read_map(real_dir+f"map_{name}.fits") for name in names_maps]
    maps_128[4:] = [map * 1.e3 for map in maps_128[4:]] # Convert units to muK
    maps_16  = [CMBanom.downgrade_map(map, Nside_out) for map in maps_128]

    print("Computing sigma_16")
    for m in range(len(names_mask)):
        mask = masks[m]
        name_mask = names_mask[m]
        print("-", name_mask, "...")

        # Compute & save sigma_16
        sigma16 = [CMBanom.sigma_16(map, mask) for map in maps_16]
        np.savetxt(stats_dir+f'sigma16_real_{name_mask}.npy', sigma16)

if compute_SQO:
    lmax_QO = 3
    print("Computing SQO:")

    # Load masks
    masks = CMBanom.read_masks(masks_dir, mask_files, Nside_in)

    # Load maps
    maps = [hp.read_map(maps_dir+f"map__{n}.fits")*1e3 for n in range(Nsims)]

    for m in range(len(names_mask)):
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

