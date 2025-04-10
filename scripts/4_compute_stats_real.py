import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline
import os
import CMBanom

# Parameters
Nside_in  = 128
real_dir  = "../data/real/"
masks_dir = "../data/masks/"
stats_dir = "../data/stats/"
names_mask = ["fullsky", "stdmask", "commask"]
mask_files = ["stdv_mask_1percent_v4.fits", "common-Mask-Int_cutoff0.9_Nside128.fits"]
names_maps = ["commander_nside_128_1deg", "nilc_nside_128_1deg", "sevem_nside_128_1deg", "smica_nside_128_1deg", "cleaned_70GHz_focused", "cleaned_94GHz_focused", "cleaned_100GHz_focused", "cleaned_143GHz_focused"]
#names_maps = ["commander_zodi_removed_L", "nilc_zodi_removed_L", "sevem_zodi_removed_L", "smica_zodi_removed_L", "cleaned_70GHz_focused", "cleaned_94GHz_focused", "cleaned_100GHz_focused", "cleaned_143GHz_focused"]
names_real = ["commander_1deg", "nilc_1deg", "sevem_1deg", "smica_1deg", "focused_70GHz", "focused_94GHz", "focused_100GHz", "focused_143GHz"]
#names_real = ["commander_zodi_removed", "nilc_zodi_removed", "sevem_zodi_removed", "smica_zodi_removed", "focused_70GHz", "focused_94GHz", "focused_100GHz", "focused_143GHz"] 
Nmasks = len(names_mask)
Nmaps = len(names_maps)

# Modes
compute_cl_corr = True
compute_Smu     = False
compute_R       = False
compute_sigma16 = False
compute_SQO     = False

## Low correlation, Smu
summation = True
mu = 0.5

# Parity asymmetry, R
lmax_R = 60

# Hemispherical asymmetry, sigma16
ecliptic_coords = True

##### Compute stats

if compute_cl_corr:
    print("Computing Cls and corrs")
    for i in range(Nmaps):
        fn_map = f"{real_dir}map_{names_maps[i]}.fits"

        for j in range(Nmasks):
            fn_corr = f'{real_dir}corr_{names_real[i]}_{names_mask[j]}.txt'
            fn_pcl  = f'{real_dir}cl_{names_real[i]}_{names_mask[j]}.txt'

            if j==0:
                print("-fullsky")
                os.system(f'spice -mapfile {fn_map} -corfile {fn_corr} -clfile {fn_pcl}')
            else:
                print(names_mask[j])
                fn_mask = f"{masks_dir}{mask_files[j-1]}"
                os.system(f'spice -mapfile {fn_map} -maskfile {fn_mask} -corfile {fn_corr} -clfile {fn_pcl}')

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

        # Load Planck Cl (begins with l=0) and correct window fcts. and units, shape: (Nmasks, Nmaps, lmax)
        cl_wf_128  = CMBanom.get_cl_wf_factor(Nside_in)
        cl_wf_1deg = CMBanom.get_cl_wf_factor(Nside_in, deg=1)
        #cls = CMBanom.load_cls(cls_dir+f"cls_{name_mask}_100k/", name_mask, Nsims, cl_wf_factor)
        cls = np.array([np.loadtxt(real_dir+f"cl_{names_real[n]}_{names_mask[m]}.txt").T[1] for n in range(Nmaps)])
        for m in range(Nmasks):
            cls[:4] *= cl_wf_128
            cls[4:] *= 1e6*cl_wf_1deg

        # Compute and save R
        R = np.array([[CMBanom.get_Rassymstat(cls[n], lmax=l) for l in range(lmax_R)] for n in range(Nmaps)])
        np.savetxt(stats_dir+f'R_real_{name_mask}.npy', R)

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
    maps = [hp.read_map(real_dir+f"map_{names_maps[n]}.fits") for n in range(Nmaps)]
    maps[4:] = [map * 1.e3 for map in maps[4:]] # Convert units to muK   
    
    for m in range(len(names_mask)):
        mask = masks[m]
        name_mask = names_mask[m]
        print(name_mask, "...")

        print("- Computing multipole vectors")
        mvs = CMBanom.compute_MVs(maps, mask, lmax_QO)

        print("- Computing oriented-area vectors")
        ws = CMBanom.compute_Ws(mvs, lmax_QO)

        print("- Computing SQO")
        SQO = np.array([CMBanom.S_QO(ws[n]) for n in range(Nmaps)])
        np.savetxt(stats_dir+f'SQO_real_{name_mask}.npy', SQO)

