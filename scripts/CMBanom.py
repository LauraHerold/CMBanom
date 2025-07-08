# This script is an adaption of Jessica Muir's github https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer/tree/master and paper https://arxiv.org/abs/1806.02354
import numpy as np
import healpy as hp
import scipy
import polymv

# File locations
## Theory Cl file used to generate simulations
cldatfile = "../../data/Cls/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
mask_fn_south_ecl = "../../data/masks/mask_south_ecl_Nside16.fits"
outdir_simmaps = "../../data/sims/"

#Conversion Nside to FWHMarcmin from Tab. 1 of Planck 2015 Isotropy and Statistics paper arXiv:1506.07135
#Exception: we smooth Nside=128 to 1 deg = 60 arcmin (Planck col. smoothes to 80 arcmin)
NSIDEtoFWHMarcmin = {2048:5, 128:60, 64:160, 16:640}
NSIDEfid = 128

##################################################################
# UTILITY FUNCTIONS FOR MAPS/MASKS
##################################################################

# Taken from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer  
def arcmin2rad(angle):
    """
    Given angle in arcmin, convert into radians
    """
    return angle*np.pi/(60.*180.)


# Adapated from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def downgrade_map(inmap, NSIDEout, DEGin=None, DEGout=None):
    """
    Downgrades map, scaling by appropriate beam and pixel window
    functions, as discussed in Planck isotropy paper.
    """
    #get coefficent to covolve with beam and pixel window func
    plout = hp.sphtfunc.pixwin(NSIDEout)
    lmax = plout.size-1
    NSIDEin = hp.get_nside(inmap)
    plin = hp.sphtfunc.pixwin(NSIDEin)[:lmax+1]
    if DEGin == None: DEGin = NSIDEtoFWHMarcmin[NSIDEin]/60.      # Use Planck smoothing convention arXiv:1506.07135 except for Nside=128 (arcmin to degree)
    fwhmin = DEGin*np.pi/180.                                   # degree to radians
    if DEGout == None: DEGout = NSIDEtoFWHMarcmin[NSIDEout]/60.    # arcmin to degree
    fwhmout = DEGout*np.pi/180.                                 # degree to radians
    blin = hp.sphtfunc.gauss_beam(fwhmin,lmax=lmax)
    blout = hp.sphtfunc.gauss_beam(fwhmout,lmax=lmax)
    multby = blout*plout/(blin*plin) #one number per ell

    #turn map to spherical harmonics, colvolve, then turn back into map
    alm = hp.sphtfunc.map2alm(inmap,lmax)
    alm = hp.almxfl(alm,multby)  #colvolve w/window funcs
    outmap = hp.sphtfunc.alm2map(alm,NSIDEout)
    return outmap

# Adapted from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def gen_maps_from_cls(cldatfile=cldatfile, outdir=outdir_simmaps, Nside=NSIDEfid, N_start=0, N_maps=1, lmax=200, regen=True, returnoutf=True):
    """                                                                                                                                    
    Given Cl data filename, desired output file lcoation and name, and some other map properties,                                          
    generates Nmaps .fits files consistent with input C_ls                                                                              
    If regen = False, doesn't generate maps, just returns filenames                                                                        
    """
    data = np.loadtxt(cldatfile, skiprows=1)
    llist = np.arange(lmax+1)
    Clist = np.zeros(lmax+1)
    Clist[2:] = data[:lmax-1, 1]*2.*np.pi/(llist[2:]*(llist[2:] + 1))
    outfiles = []

    # Generate and save maps 
    seeds = np.arange(N_start,N_start+N_maps)
    for seed in seeds:
        np.random.seed(seed)
        outf = outdir+"map__"+str(seed)+".fits"
        if regen:
            m = hp.sphtfunc.synfast(Clist, nside=Nside, fwhm=arcmin2rad(NSIDEtoFWHMarcmin[Nside]), pixwin=True)
            hp.write_map(outf, m, overwrite=True, dtype=np.float64)
            print(outf)
        if returnoutf:
            outfiles.append(outf)
    return outfiles


def get_cl_wf_factor(Nside, deg=None, lmax=384):
    if deg == None: deg = NSIDEtoFWHMarcmin[Nside]/60.       # arcmin to degree
    deg_rad = deg*np.pi/180.                                 # degree to radians
    pixwin = hp.pixwin(Nside)[:lmax]                         # pixwin takes Nside
    beam = hp.sphtfunc.gauss_beam(fwhm=deg_rad, lmax=lmax-1) # gauss_beam takes fwhm in radians
    return 1./pixwin**2/beam**2

def read_masks(dir_mask, names_mask, Nside):
    masks = np.ones((3, hp.nside2npix(Nside)))
    for m in range(len(names_mask)): masks[m+1] = hp.read_map(dir_mask+names_mask[m])
    return masks
            

def pval_lower(val_real, vals_sims):
    return len(np.where(vals_sims<val_real)[0])/len(vals_sims)
    
    
def pval_higher(val_real, vals_sims):
    return len(np.where(vals_sims>val_real)[0])/len(vals_sims)


##################################################################
# Low correlation, S_1/2
##################################################################


def corr_from_cl(theta, C_l, lmax=30):
    # Cl's starting from l=0
    ll = np.arange(2,np.minimum(len(C_l),lmax))
    cos = np.cos(theta)
    corr = np.zeros(cos.shape)
    legendre = scipy.special.legendre
    for l in ll:
        corr+= (2.*l + 1.)/(4*np.pi) * C_l[int(l)] * legendre(l)(cos)
    return corr

def load_corrs(fn_corrs, name_mask, Nsims):
    theta, cos_theta = np.loadtxt(fn_corrs+f'corr_{name_mask}__1.txt').T[0:2]
    corrs = np.array([np.loadtxt(fn_corrs+f'corr_{name_mask}__{n}.txt').T[2] for n in range(Nsims)])
    return theta, cos_theta, corrs

def S_mu_many(C_theta, cos_theta, mu, summation=True):
    """
    Compute S_mu via naive summation of C_theta_i**2 * cos_theta_i
    """
    if summation:
        dcos_theta = np.append(cos_theta[1:] - cos_theta[:-1], np.zeros(1))
            
        # Sum only over C_theta where cos_theta<mu
        C_theta_mu = np.where(cos_theta<mu, C_theta, 0)
        S_mu = np.sum(C_theta_mu**2*dcos_theta, axis=1)
        
    return S_mu

def S_mu_sum(corr_file, mu, summation=True):
    """
    Compute S_mu via naive summation of C_theta_i**2 * cos_theta_i
    """
    if summation:
        if len(corr_file.shape)==2:
            cos_theta = corr_file[1]
            dcos_theta = np.append(cos_theta[1:] - cos_theta[:-1], np.zeros(1))
            C_theta = 1e12*corr_file[2]
        
            # Sum only over C_theta where cos_theta<mu
            C_theta_mu = np.where(cos_theta<mu, C_theta, 0)
            S_mu = np.sum(C_theta_mu**2*dcos_theta)
        
        elif len(corr_file.shape)==3:
            cos_theta = corr_file[0,1]
            dcos_theta = np.append(cos_theta[1:] - cos_theta[:-1], np.zeros(1))
            C_theta = 1e12*corr_file[:,2]
        
            # Sum only over C_theta where cos_theta<mu
            C_theta_mu = np.where(cos_theta<mu, C_theta, 0)
            S_mu = np.sum(C_theta_mu**2*dcos_theta, axis=1)
        else:
            print("Unexpected dimension of corr_file.")   
    return S_mu

def S_mu_simps(corr_file, mu):
    """
    Integration using Simpson's rule (quadratic interpolation procedure)   
    """
    if len(corr_file.shape)==2:
        cos_theta = corr_file[1]
        dcos_theta = np.append(cos_theta[1:] - cos_theta[:-1], np.zeros(1))
        C_theta = 1e12*corr_file[2]
        
        # Integrate only over C_theta where cos_theta<mu
        C_theta_mu = np.where(cos_theta<mu, C_theta, 0)
        S_mu = scipy.integrate.simps(C_theta_mu**2, x=cos_theta)
        
    elif len(corr_file.shape)==3:
        cos_theta = corr_file[0,1]
        dcos_theta = np.append(cos_theta[1:] - cos_theta[:-1], np.zeros(1))
        C_theta = 1e12*corr_file[:,2]
        
        # Sum only over C_theta where cos_theta<mu
        C_theta_mu = np.where(cos_theta<mu, C_theta, 0)
        S_mu = np.zeros(len(C_theta_mu))
        for n in np.arange(0, len(C_theta_mu)):
            S_mu[n] = scipy.integrate.simps(C_theta_mu[n]**2, x=cos_theta)
    else:
        print("Unexpected dimension of corr_file.")   
    return S_mu



# Adapted from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def S_mu_Itab(clin,mu=0.5,LMAX=100,Itab = np.array([])): 
    """
    Computes measure of large scale power S(x)
    cl = array of C_l's in muK**2, mu= upper bound on cos(theta) integral
    """
    if (not LMAX) or (LMAX>clin.size -1):
        LMAX = clin.size -1
        cl = clin
    else:
        cl = clin[:LMAX+1]
    #print '  Computing S('+str(x)+') with LMAX=',LMAX
    if not Itab.shape==(LMAX+1,LMAX+1):
        Itab = tabulate_Ifunc(x = mu, LMAX = LMAX)
        
    cldat = (2.*np.arange(LMAX+1) + 1)*cl
    clascol = np.tile(cldat.reshape((cldat.size,1)),cldat.size) 
    clasrow = clascol.T
    Sval = np.sum((clascol*clasrow*Itab)[2:,2:])/(16.*np.pi*np.pi)
    return Sval

# Taken from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def tabulate_Ifunc(x,LMAX):
    """
    This is the matrix you get when integrating two legengre 
    polynomials with l=m,n from -1 to x

    Will return array of shape (LMAX+1)x(LMAX+1)
    """
    legPx = np.zeros(LMAX+2)
    for i in range(LMAX+2):
        Pcoef = np.zeros(i+1)
        Pcoef[-1] = 1
        legPx[i] = np.polynomial.legendre.legval(x,Pcoef)
    Imat = np.zeros((LMAX+2,LMAX+2))
    # need the LMAX+1 index for last diagonal entry, will
    # slice of last row and column before returning
    for m in range(LMAX+1):
        for n in range(m+1,LMAX+1):
            #do off diagonals first, as diagonals depend on them
            if m==0:
                A = 0.
            else:
                A = m*legPx[n]*(legPx[m-1]-x*legPx[m])
            if n==0:
                B = 0.
            else:
                B = -n*legPx[m]*(legPx[n-1]-x*legPx[n])
            Imat[n,m] = (A+B)/(n*n+n - m*m-m)
            Imat[m,n] = Imat[n,m] #symmetric
    for m in range(LMAX+1):

        if m==0:
            Imat[m,m] = x+1.
        elif m==1:
            Imat[m,m] = (x**3+1.)/3.
        else:
            A = (legPx[m+1]-legPx[m-1])*(legPx[m]-legPx[m-2])
            B = -(2*m-1)*Imat[m+1,m-1]+(2*m+1)*Imat[m,m-2]
            C = (2*m-1)*Imat[m-1,m-1]
            Imat[m,m] = (A+B+C)/(2*m+1)
    
    return Imat[:-1,:-1]

##################################################################                                                                         
# Parity asymmetry
##################################################################

def load_cls(fn_cls, name_mask, Nsims, cl_wf_factor=1.):
    cls = np.array([np.loadtxt(fn_cls+f'cl_{name_mask}__{n}.txt').T[1] for n in range(Nsims)])*cl_wf_factor
    return cls

# Adapted from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer 
def get_Rassymstat(cl,lmax=27,clstartsat=0):
    """
    Given C_ell data and lmax, computes R stat as on page 25 of
    https://arxiv.org/abs/1506.07135, which measures amount of parity
    assymetry. If 1, no assymetry, >1 even parity pref, <1 odd parity pref
    """

    if lmax<3:
        R = 1.
    else:
        LMIN=2
        ell = np.arange(LMIN,lmax+1)
        isodd = (ell%2).astype(bool)
        iseven = np.logical_not(isodd)

        Dl = ell*(ell+1.)*cl[LMIN-clstartsat:lmax+1-clstartsat]
        R = np.mean(Dl[iseven])/np.mean(Dl[isodd])
    
    return R
    
##################################################################
# Low northern variance, sigma^2_16
##################################################################
    
# This is \sigma^2_{16}, the variance at Nside = 16
def sigma2_16(inmap, mask):
    sigma2_16 = np.nanvar(inmap*mask)
    #sigma2_16 = np.nansum((inmap*mask)**2)/len(np.nonzero(mask==1.)[0])
    return sigma2_16
    
##################################################################
# Multipole alignments, S_QO
##################################################################

def compute_MVs(maps, mask, lmax):

    # Compute alms                                                                                                               
    alms = [hp.sphtfunc.map2alm(maps[n]*mask, lmax) for n in range(len(maps))]

    # Compute MVs with polymv
    mvs = []
    for n in range(len(maps)):
        mvs_all = [polymv.mvs.m_vectors(alms[n], ell) for ell in range(lmax+1)]
        mvs.append([polymv.otherfuncs.mvs_north(mvs_all[ell]) for ell in range(lmax+1)])
    return mvs
    
                
def compute_Ws(mvs, lmax):
    ws = []
    for n in range(len(mvs)):
        ws_n = [0,0]
        for l in range(2,lmax+1):
            ws_l = []
            for i in range(len(mvs[n][l])-1):
                mv1 = polymv.otherfuncs.to_cart(mvs[n][l])[i]
                for j in range(i+1, len(mvs[n][l])):
                    mv2 = polymv.otherfuncs.to_cart(mvs[n][l])[j]
                    ws_l.append(np.cross(mv1, mv2))
            ws_n.append(ws_l)
        ws.append(ws_n)
    return ws


def S_QO(ws):
    S = 0
    for i in range(3):
        S+= np.absolute(np.dot(ws[2], ws[3][i]))[0]
    return S/3.


##################################################################
# Hemispherical asymmetry, ALV
##################################################################


# All inspired by https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def get_pixlist(theta_deg, mask, Nside_in, Nside_out):
    pixlist = []
    unmasked = np.nonzero(mask)[0]
    Npix_out = hp.nside2npix(Nside_out)
    
    for p in range(Npix_out):
        alldisk = hp.query_disc(nside=Nside_in, vec=hp.pix2vec(Nside_out, p), radius=np.deg2rad(theta_deg))
        unmaskeddisk = np.intersect1d(alldisk, unmasked, assume_unique=True)
        pixlist.append(unmaskeddisk)
    
    return pixlist
        
def get_lvmask(pixlist, theta_deg, frac_to_be_masked, Nside_in, Nside_out):
    Npix_out = hp.nside2npix(Nside_out)
    fracunmasked = np.array([float(pixlist[i].size)/float(hp.query_disc(nside=Nside_in, vec=hp.pix2vec(Nside_out, i), radius=np.deg2rad(theta_deg)).size) for i in range(Npix_out)])
    lvmask = (fracunmasked > frac_to_be_masked).astype(bool)
    
    return lvmask

def get_lvmap(inmap, mask, pixlist, Nside_out):
    Npix_out = hp.nside2npix(Nside_out)
    inmap = hp.ma(inmap)
    inmap.mask = np.logical_not(mask)
    inmap = hp.remove_dipole(inmap)
    inmap_nanmask = np.where(mask==0, np.nan, inmap.data) # avoiding warning: converting a masked element to nan
        
    lvmap = np.zeros(Npix_out)
    for i in range(Npix_out):
        #lvmap[i] = np.var(inmap_nanmask[pixlist[i]])
        #if len(pixlist[i])!= 0: lvmap[i] = np.sum(inmap[pixlist[i]]**2)/len(pixlist[i])
        if len(pixlist[i])!= 0: lvmap[i] = np.sum((inmap[pixlist[i]]-np.nanmean(inmap_nanmask))**2)/len(pixlist[i])
        
    return lvmap

def ALV_vec(lvmap, lvmaps_sims, lvmask):

    mean_lvmap = np.where(lvmask==1., np.mean(lvmaps_sims, axis=0), 1.)
    var_lvmap  = np.where(lvmask==1., np.sum(lvmaps_sims**2, axis=0)/len(lvmaps_sims)/mean_lvmap**2, 1.)
    meanvar_lvmap = np.nanmean(var_lvmap*lvmask)
    normlvmap = (meanvar_lvmap/var_lvmap)*(lvmap - mean_lvmap)/mean_lvmap
    normlvmap = hp.ma(normlvmap)
    normlvmap.mask = np.logical_not(lvmask)
    dipolevec = hp.remove_dipole(normlvmap, fitval = True)[2]
    ALV = np.linalg.norm(dipolevec)
    
    return ALV, dipolevec
 
#def ALV_vec(lvmap, lvmaps_sims, lvmask):
#    mean_lvmap = np.mean(lvmaps_sims, axis=0)
#    var_lvmap = np.sum(lvmaps_sims**2, axis=0)/len(lvmaps_sims)/mean_lvmap**2
#    #var_lvmap = np.var(lvmaps_sims, axis=0, ddof=1)/mean_lvmap**2
#    meanvar_lvmap = np.nanmean(var_lvmap*lvmask)
#    normlvmap = (meanvar_lvmap/var_lvmap)*(lvmap - mean_lvmap)/mean_lvmap
#    normlvmap = hp.ma(normlvmap)
#    normlvmap.mask = np.logical_not(lvmask)
#    dipolevec = hp.remove_dipole(normlvmap, fitval = True)[2]
#    
#    return dipolevec
