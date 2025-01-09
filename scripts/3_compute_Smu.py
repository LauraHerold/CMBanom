import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline
import CMBanom

# Parameters
N_maps = 10000
mu = 0.5
summation = True
compute_Smu = False
compute_envelopes = True
names_mask = ["fullsky", "stdmask", "commask"]
fn_corrs = "../data/sims/"
fn_Smu = "../data/stats/"

for m in range(len(names_mask)):
    name_mask = names_mask[m]
    # Read C(\theta)'s                                                              
    C_theta = np.zeros((N_maps, 384))
    for n in range(0, N_maps):
        name = "corr_"+name_mask+"__"+str(n)+".txt"
        C_theta[n] = np.loadtxt(fn_corrs+name).T[2]
    theta = np.loadtxt(fn_corrs+name).T[0]
    cos_theta = np.loadtxt(fn_corrs+name).T[1]
    dtheta = np.append(theta[:-1] - theta[1:], np.zeros(1))
    dcos_theta = np.append(cos_theta[1:] - cos_theta[:-1], np.zeros(1))

    # Compute Smu
    if compute_Smu:
        if summation:
            S_mu = CMBanom.S_mu_many(C_theta, cos_theta, mu)

        else:
            S_mu = np.zeros(N_maps)
            for n in np.arange(0, N_maps):
                C_theta_int = UnivariateSpline(cos_theta, C_theta[n]**2)
                S_mu[n] = integrate.quad(C_theta_int, -1, mu)[0]

        name_Smu = "Smu_sims_"+name_mask+"_Nmaps_"+str(N_maps)+".npy"
        np.savetxt(fn_Smu+name_Smu, S_mu)


    if compute_envelopes:
        mean_corr = np.mean(C_theta, axis=0)
        std_corr = np.std(C_theta, axis=0)
        np.savetxt(fn_Smu+"corr_mean_std_"+name_mask+".npy", np.array([mean_corr, std_corr]))

        # Correct for pixel window fct. & beam (smoothing window fct.)
        pixwin_128 = hp.pixwin(128)[:384]
        beam_128 = hp.sphtfunc.gauss_beam(fwhm=80*np.pi/(60.*180.), lmax=383, pol=False)

        cls = np.zeros((N_maps, 384))
        for n in range(0, N_maps):
            name_cl = "cl_"+name_mask+"__"+str(n)+".txt"
            cls[n] = np.loadtxt(fn_corrs+name_cl).T[1]/(beam_128**2*pixwin_128**2)

        mean_cls = np.mean(cls, axis=0)
        std_cls = np.std(cls, axis=0)
        np.savetxt(fn_Smu+"cls_mean_std_"+name_mask+".npy", np.array([mean_cls, std_cls]))
