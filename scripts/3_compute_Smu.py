import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline
import CMBanom

# Parameters
N_maps = 10000
mu = 0.5
summation = True
save_Smu = True
fn_corrs = "../data/sims/"
name_corr = "corr_commask__"
fn_Smu = "../data/stats/"
name_Smu = "Smu_sims_commask_Nmaps_"

# Read C(\theta)'s                                                              
C_theta = np.zeros((N_maps, 384))
for n in range(0, N_maps):
    name = name_corr+str(n)+".txt"
    C_theta[n] = np.loadtxt(fn_corrs+name).T[2]
theta = np.loadtxt(fn_corrs+name).T[0]
cos_theta = np.loadtxt(fn_corrs+name).T[1]
dtheta = np.append(theta[:-1] - theta[1:], np.zeros(1))
dcos_theta = np.append(cos_theta[1:] - cos_theta[:-1], np.zeros(1))

# Compute Smu
if summation:
    S_mu = CMBanom.S_mu_many(C_theta, cos_theta, mu)

else:
    S_mu = np.zeros(N_maps)
    for n in np.arange(0, N_maps):
        C_theta_int = UnivariateSpline(cos_theta, C_theta[n]**2)
        S_mu[n] = integrate.quad(C_theta_int, -1, mu)[0]

# Save Smu
if save_Smu:
    np.savetxt(fn_Smu+name_Smu+str(N_maps)+".npy", S_mu)
