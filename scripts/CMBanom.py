
# This script is an adaption of Jessica Muir's github https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer/tree/master and paper https://arxiv.org/abs/1806.02354
import numpy as np
import healpy as hp
import scipy

# File locations
## Theory Cl file used to generate simulations
cldatfile = "../../data/Cls/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
mask_fn_south_ecl = "../../data/masks/mask_south_ecl_Nside16.fits"
outdir_simmaps = "../../data/sims/"

#Conversion Nside to FWHMarcmin from Tab. 1 of Planck 2015 Isotropy and Statistics paper arXiv:1506.07135
NSIDEtoFWHMarcmin = {2048:5, 1024:10, 512:20, 256:40, 128:80, 64:160, 32:320, 16:640}
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

# Taken from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def downgrade_map(inmap, NSIDEout):
    """
    Downgrades map, scaling by appropriate beam and pixel window
    functions, as discussed in Planck isotropy paper.
    """
    #get coefficent to covolve with beam and pixel window func
    plout = hp.sphtfunc.pixwin(NSIDEout)
    lmax = plout.size-1
    #print("lmax: ", lmax)
    NSIDEin = hp.get_nside(inmap)
    plin = hp.sphtfunc.pixwin(NSIDEin)[:lmax+1]
    fwhmin = arcmin2rad(NSIDEtoFWHMarcmin[NSIDEin])
    blin = hp.sphtfunc.gauss_beam(fwhmin,lmax=lmax)
    fwhmout = arcmin2rad(NSIDEtoFWHMarcmin[NSIDEout])
    blout = hp.sphtfunc.gauss_beam(fwhmout,lmax=lmax)
    multby = blout*plout/(blin*plin) #one number per ell

    #turn map to spherical harmonics, colvolve, then turn back into map
    alm = hp.sphtfunc.map2alm(inmap,lmax)
    alm = hp.almxfl(alm,multby)  #colvolve w/window funcs
    outmap = hp.sphtfunc.alm2map(alm,NSIDEout)
    return outmap

# Adapted from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def get_filename_testcase(datadir,mapbase,nside,number,stattype = 'map', maskname='',extratag = ''):
    if stattype == 'map':
        ending = '.fits'
    elif stattype == 'cl':
        ending = '.cl.dat'
    elif stattype == 'ct':
        ending = '.ct.dat'
    elif stattype == 'lvmap':
        ending = '.{0:s}.fits'.format(extratag)
    elif stattype == 'statsummary':
        ending = '.stats.dat'
    elif stattype == 'Rall':
        ending = '.Rall.dat'
    elif stattype == 'Rall-contours':
        ending = '.Rall-contours.dat'
    elif stattype == 'Rall-contours-singletail':
        ending = '.Rall-contours-singletail.dat'
    elif stattype == 'Rall-contours-hist':
        ending = '.Rall-contours-hist.dat'
    if maskname:
        maskstr = '-'+maskname
    else:
        maskstr = ''
    return ''.join([datadir,mapbase, '_', str(nside), '_', maskstr,'_{0:d}'.format(number), ending])


# Adapted from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer 
def gen_maps_from_cls(cldatfile=cldatfile, outdir=outdir_simmaps, Nside=NSIDEfid, N_start=0, N_maps=1, lmax=200, regen=True, returnoutf=True):
    """                                                                                                                                    
    Given Cl data filename, desired output file lcoation and name, and some other map properties,                                          
    generates Nmaps .fits files consistant with input C_ls                                                                              
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
        #outf = get_filename_testcase(outdir,'map', Nside, seed,'map')
        outf = outdir+"map__"+str(seed)+".fits"
        if regen:
            m = hp.sphtfunc.synfast(Clist, nside=Nside, fwhm=arcmin2rad(NSIDEtoFWHMarcmin[Nside]), pixwin=True)
            hp.write_map(outf, m, overwrite=True, dtype=np.float64)
            print(outf)
        if returnoutf:
            outfiles.append(outf)
    return outfiles
    
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
    
def S_mu_many(C_theta, cos_theta, mu):
    """
    Compute S_mu via naive summation of C_theta_i**2 * cos_theta_i
    """
    dcos_theta = np.append(cos_theta[1:] - cos_theta[:-1], np.zeros(1))
            
    # Sum only over C_theta where cos_theta<mu
    C_theta_mu = np.where(cos_theta<mu, C_theta, 0)
    S_mu = np.sum(C_theta_mu**2*dcos_theta, axis=1)
        
    return S_mu

def S_mu_sum(corr_file, mu):
    """
    Compute S_mu via naive summation of C_theta_i**2 * cos_theta_i
    """
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


# Taken from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer 
def get_Rassymstat(cl,lmax=27,clstartsat=0):
    """
    Given C_ell data and lmax, computes R stat as on page 25 of
    https://arxiv.org/abs/1506.07135, which measures amount of parity
    assymetry. If 1, no assymetry, >1 even parity pref, <1 odd parity pref
    """
   
    LMIN=2
    ell = np.arange(LMIN,lmax+1)
    isodd = (ell%2).astype(bool)
    iseven = np.logical_not(isodd)

    Dl = ell*(ell+1.)*cl[LMIN-clstartsat:lmax+1-clstartsat]
    R = np.mean(Dl[iseven])/np.mean(Dl[isodd])
    
    return R
    
##################################################################
# Hemispherical asymmetry, sigma_16
##################################################################
    
def sigma_16(map, mask):
    return np.nanstd(map*mask)**2
    
##################################################################
# Multipole alignments, S_QO
##################################################################
    
def S_QO(ws):
    S = 0
    for i in range(3):
        S+= np.absolute(np.dot(ws[2], ws[3][i]))[0]
    return S/3.
    
##################################################################
# ALV -- needs adaption
##################################################################

# Taken from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def get_ALV_onemap_externalmean(datadir, mapbase, meanmapfile, varmapfile, lvmapdir = '', maskfile='', maskname='', disksizedeg = 8, NSIDELV = 16, overwrite = False):
    """
    Make LV map for a single map, then use externally saved mean and
    variance maps to measure ALV.
    (e.g. When measuring ALV for planck, need to get mean and variance
    from some set of simulations)

    datadir, mapbase, maskfile, maskname - info about input map
    lvmapdir - where to put local variance maps; defaults to datadir
    meanmapfile - filename of mean LV map to use when measuring LV
    variancemapfile - filename of variance LV map to use when measuring LV

    if overwrite, remeasure LV map, otherwise, read in file if it exists
    """
    if not lvmapdir:
        lvmapdir = datadir
    
    extractLVmap_forlist(0,0, datadir = datadir, lvmapdir = lvmapdir, mapbase = mapbase, maskfile = maskfile, maskname = maskname, disksizedeg = 8, NSIDELV = 16, overwrite = overwrite)

    lvbasestr ='LVd{0:02d}n{1:02d}'.format(int(disksizedeg),NSIDELV)
    lvmapf = get_filename_testcase(datadir = lvmapdir, mapbase = mapbase, number =0, stattype = 'lvmap', maskname = maskname,extratag = lvbasestr)
    lvmap = hp.read_map(lvmapf, verbose=False)
    if maskname:
        lvmaskfile= "{0:s}{1:s}_for{2:s}.fits".format(lvmapdir,maskname,lvbasestr)
        if  os.path.isfile(lvmaskfile): #make lv mask file if we don't have it
            lvmask = hp.read_map(lvmaskfile, verbose = False)
        else:
            mask = hp.read_map(maskfile, verbose = False)
            diskpixlist = get_diskpixlist(NSIDEfid,NSIDELV,disksizedeg,mask)
            lvmask = getLVmap_mask(mask,disksizedeg , NSIDELV,diskpixlist)
            hp.write_map(lvmaskfile,lvmask)
    else:
        lvmask = None
    
    #use mean and var from set of simulations
    meanmap = hp.read_map(meanmapfile, verbose=False)
    varmap = hp.read_map(varmapfile, verbose=False)
    zerovar = np.where(varmap==0)[0] #pixel indices
    varmap[zerovar] = 1
    weightmap = (meanmap**2)/varmap #meanmap factor is to make this dimensionless
    if lvmask is None:
        meanweights =  np.mean(weightmap)
    else:
        meanweights = np.mean(weightmap[lvmask==1])
    weightmap = weightmap/meanweights #avg weight is 1
    weightmap[zerovar] = 0
    
    ALV = get_ALVstat_foronemap(lvmap,meanmap,weightmap,lvmask)
    #note, if we save this, keep track of where the mean came from
    return ALV


# Taken from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def get_manyALVstats(realmin=0,realmax=100, mapbase = 'map',\
                     maskfile='',maskname='', \
                     datadir = 'output/lcdm-map-testspace/',\
                     lvmapdir = 'output/lvmap-testspace/', \
                     statdir = 'output/stat-testspace/', \
                     disksizedeg = 8, NSIDELV=16,redoLVmaps = True, Ncore=0):
    """
    Given parameters re. input names, output dir, etc.,
    runs procedures necessary to create local variance maps
    and measure A_LV statistics from them. 
    """
    #if lvmapdir or statdir aren't given, default to put them in datadir
    if not lvmapdir:
        lvmapdir = datadir
    if not statdir:
        stadir = datadir
    
    #first, make the LV maps, measuring mean and variance
    meanfile, varfile = get_LVmaps_formaplist(realmin = realmin, \
                                              realmax = realmax,\
                                              maskfile=maskfile,\
                                              maskname=maskname, \
                                              datadir = datadir,\
                                              lvmapdir = lvmapdir , \
                                              mapbase = mapbase, \
                                              disksizedeg = disksizedeg, \
                                              NSIDELV= NSIDELV,\
                                              overwrite = redoLVmaps,\
                                              Ncore = Ncore)

    #then parallelize and measure stats
    print("Computing ALV for list of existing LV maps; parallelizing.")
    availcore = multiprocessing.cpu_count()
    if not Ncore or (availcore<Ncore):
        Ncore = availcore
    print("Using {0} cores.".format(Ncore))
    edges = np.linspace(realmin,realmax,num=Ncore+1,dtype=int)
    print("Splitting realizations into chunks with edges:\n",edges)
    rmins = edges[:-1]
    rmaxs = edges[1:]-1
    rmaxs[-1]+=1
    #start processes for each chunk of realizations
    jobs = []
    for i in xrange(Ncore):
        p = multiprocessing.Process(target = computeALV_forlist, args=(rmins[i],rmaxs[i], lvmapdir, statdir, mapbase, maskname, disksizedeg, NSIDELV, meanfile, varfile))
        jobs.append(p)
        print("Starting ALV meas for rlzns {0:d}-{1:d}".format(rmins[i],rmaxs[i]))
        p.start()
    #wait until all are done before moving on
    for j in jobs:
        j.join()


# Taken from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def computeALV_forlist(realmin,realmax,\
                       lvmapdir = '', statdir = '',\
                       mapbase='map', maskname='',\
                       disksizedeg = 8, NSIDELV=16, \
                       meanmapfile='',varmapfile = '',lvmaskfile=''):
    """
    Given range of realizations and info about input LV map filenames,
    reads in local variance maps, and uses meanmap and varmap to extract
    ALV.

    lvmapdir is where the local variance maps are stored
    statdir is where to put the extracted ALV files

    diskpixlist - array of arrays; computed if emtpy, but can be passed
            to save time

    if meanmapfile ,varmapfile, lvmaskfile string are passed, reads from them
           otherwise assumes their format matches the LV maps
    meanmapfile contains mean map of set of simulations
    varmap file contains variance map of set of simulations
    lvmaskfile contains mask for LV map where pixels are masked if less than 
       10% of pixels in its associated disk were unmasked
    """
    if not statdir:
        statdir = datadir
    lvbasestr = 'LVd{0:02d}n{1:02d}'.format(int(disksizedeg),NSIDELV)

    #get the lv mask
    if maskname and (not lvmaskfile):
        lvmaskfile= "{0:s}{1:s}_for{2:s}.fits".format(lvmapdir,maskname,lvbasestr)
    if lvmaskfile:
        print("Reading in LV mask file",lvmaskfile)
        lvmask = hp.read_map(lvmaskfile, verbose = False)
    else:
        lvmask = None
        
    #get the mean and variance maps, as well as the LV map mask
    if not meanmapfile:
        meantag = '{2:s}_MEANr{0:d}-{1:d}'.format(realmin,realmax,lvbasestr)
        vartag = '{2:s}_VARr{0:d}-{1:d}'.format(realmin,realmax,lvbasestr)
        meanmapfile = get_filename_testcase(datadir = lvmapdir, mapbase = mapbase, number =0, stattype = 'lvmap', maskname = maskname,extratag = meantag)
        varmapfile = get_filename_testcase(datadir = lvmapdir, mapbase = mapbase, number =0, stattype = 'lvmap', maskname = maskname,extratag = vartag)
    print("Getting mean and variance of LV maps from",meanmapfile,' and ', varmapfile)
    meanmap = hp.read_map(meanmapfile, verbose = False)
    varmap = hp.read_map(varmapfile, verbose = False)
    zerovar = np.where(varmap==0)[0] #pixel indices
    varmap[zerovar] = 1
    weightmap = (meanmap**2)/varmap #should be dimensionless
    # ^this is equivalent to taking the variance of map/meanmap
    if lvmask is None:
        meanweights = np.mean(weightmap)
    else:
        meanweights = np.mean(weightmap[lvmask==1])
    weightmap = weightmap/meanweights #avg weight is 1
    weightmap[zerovar] = 0

    #set up output file and data structures
    rlzns = np.arange(realmin,realmax+1)
    Nmap = rlzns.size
    ALVf = get_filename_forstat(datadir = statdir, mapbase = mapbase, rmin = realmin, rmax = realmax, stattype = 'ALV', maskname = maskname)
    ALVdat = np.ones((Nmap,2))*np.nan
    ALVdat[:,0]=rlzns
    lvbasestr = 'LVd{0:02d}n{1:02d}'.format(int(disksizedeg),NSIDELV)
    inmaplist = [get_filename_testcase(datadir = lvmapdir, mapbase = mapbase, number =i, stattype = 'lvmap', maskname = maskname,extratag = lvbasestr) for i in rlzns]
    for i in xrange(Nmap):
        m = hp.read_map(inmaplist[i], verbose = False)
        ALVdat[i,1] = get_ALVstat_foronemap(m,meanmap,weightmap,lvmask)
    print("Saving ",ALVf)
    np.savetxt(ALVf,ALVdat,header = 'realization, ALV for '+lvbasestr)


# Taken from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def get_ALVstat_foronemap(lvmap,meanmap,weights,lvmask = None):
    normlvmap = ((lvmap - meanmap)/meanmap)*weights
    # normelvmap should be dimensionless
    if (lvmask is not None):
        normlvmap = hp.ma(normlvmap)
        normlvmap.mask = np.logical_not(lvmask)
    dipolevec = hp.remove_dipole(normlvmap,fitval = True, verbose=False)[2]
    ALV = np.linalg.norm(dipolevec)
    return ALV



# Taken from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def get_LVmaps_formaplist(realmin,realmax,maskfile='',maskname='', \
                          datadir = 'output/lcdm-map-testspace/',\
                          lvmapdir = '',  mapbase = 'map',\
                          disksizedeg = 8, NSIDELV=16, overwrite=False,\
                          Ncore = 0):

    """
    Measures dipole amplitude of local variance maps. Steps:
    - remove monopole and dipole from masked sky
    - for each realization, make NSIDE=lvnside map, for each pixel, 
      store value of variance of unmasked pixels within a disk of size 
      disksizedeg (units=degrees) (save these in lvmapdir)
    - given all simulated LV maps of a set, find mean and variance maps 
      (avg over realizations)
    - subtract mean map from all simulated and observed LV maps
    - measure dipole amplidude from each  map, weighting pixels with inverse
      of variance accross simulated realizations

    If maskfile is empty string, assumes full sky.
    maskname is a shorter string used to indicate which mask was used
    in the output filenames

    datadir is where to find input maps
    lvmapdir is where to put output files for local variance maps; if empty
            is made equal to datadir
    disksizedeg - is the size of disks to use when measuring local variance
            in units of degrees
    NSIDELV - is the NSIDE paramter to use when making local variance masks
    overwrite - if true, redo all maps; if false
                only do if files don't already exist
    """
    
    lvbasestr = 'LVd{0:02d}n{1:02d}'.format(int(disksizedeg),NSIDELV)
    if not lvmapdir:
        lvmapdir = datadir

    #check input NSIDE
    firstmap = hp.read_map(get_filename_testcase(datadir = datadir, mapbase = mapbase, number =realmin, stattype = 'map'), verbose = False)
    NSIDEIN = hp.get_nside(firstmap)
    
    #set up mask if we have one, get low res version
    if maskfile:
        #check that the file exists
        assert os.path.isfile(maskfile), "Mask file doesn't exist!"
        #get mask and use it to get disk pixel lists
        mask = hp.read_map(maskfile, verbose = False)
        diskpixlist = get_diskpixlist(NSIDEIN,NSIDELV,disksizedeg,mask)
        #if we haven't already done so, get low res mask for LV maps
        if not maskname:
            maskname = maskfile[maskfile.rfind('/')+1:]
        maskstr = '-'+maskname

        lowresmaskf = "{0:s}{1:s}_for{2:s}.fits".format(lvmapdir,maskname,lvbasestr)
        if overwrite or not os.path.isfile(lowresmaskf):
            lowresmask = getLVmap_mask(mask,disksizedeg , NSIDELV,diskpixlist)
            hp.write_map(lowresmaskf,lowresmask)
            print("Saving mask for LV maps to",lowresmaskf)
        else:
            print("Reading in mask for LV maps from",lowresmaskf)
            lowreskmask = hp.read_map(lowresmaskf, verbose = False)
        
    else:
        maskstr = ''
        lowresmaskf = ''
        diskpixlist = get_diskpixlist(NSIDEIN,NSIDELV,disksizedeg)

    #using multiprocessing, go make all non-norm LV maps
    #split up realization numbers into chunks
    availcore = multiprocessing.cpu_count()
    if not Ncore or (availcore<Ncore):
        Ncore = availcore
    print("Using {0} cores for LV map making.".format(Ncore))
    edges = np.linspace(realmin,realmax,num=Ncore+1,dtype=int)
    print("Splitting realizations into chunks with edges:\n",edges)
    rmins = edges[:-1]
    rmaxs = edges[1:]-1
    rmaxs[-1]+=1
    #start processes for each chunk of realizations
    jobs = []
    print("LV map extraction, creating QUEUE")
    queue = multiprocessing.Queue()
    print("LV map extraction, Starting processe")
    for i in xrange(Ncore):
        p = multiprocessing.Process(target = extractLVmap_forlist, args=(rmins[i], rmaxs[i], datadir, lvmapdir, mapbase, maskfile, maskname, disksizedeg, NSIDELV, overwrite, diskpixlist, NSIDEIN, queue))
        jobs.append(p)
        print("LV map making: Starting rlzns {0:d}-{1:d}".format(rmins[i],rmaxs[i]))
        p.start()
        
    print("map-making jobs done, getting mean and variance info from queue")
    meanmaps = []
    varmaps = []
    counts = []
    for j in jobs:
        dat = queue.get()
        #print '  dat=',dat
        meanmaps.append(dat[0])
        varmaps.append(dat[1])
        counts.append(dat[2])
    #wait until all are done before moving on
    print("waiting until all jobs are done")
    for j in jobs:
        j.join()
        
    # go from mean and variances of subsets to toal mean and variance
    meanmaps = np.array(meanmaps)
    varmaps = np.array(varmaps)
    counts = np.array(counts)
    for i in xrange(counts.size):
        meanmaps[i,:]*=counts[i]
        varmaps[i,:]*=counts[i]
    totalmean = np.sum(meanmaps,axis=0)/np.sum(counts)
    totalvar = np.sum(varmaps,axis=0)/np.sum(counts)

    #save mean and variance
    meantag = '{2:s}_MEANr{0:d}-{1:d}'.format(realmin,realmax,lvbasestr)
    vartag = '{2:s}_VARr{0:d}-{1:d}'.format(realmin,realmax,lvbasestr)
    meanfile = get_filename_testcase(datadir = lvmapdir, mapbase = mapbase, number =0, stattype = 'lvmap', maskname = maskname,extratag = meantag)
    varfile = get_filename_testcase(datadir = lvmapdir, mapbase = mapbase, number =0, stattype = 'lvmap', maskname = maskname,extratag = vartag)
    print("Writing mean LV map to", meanfile)
    print(totalmean.shape)
    hp.write_map(meanfile,totalmean)
    print("Writing variance LV map to", varfile)
    hp.write_map(varfile,totalvar)
    
    return meanfile,varfile
        
    
# Taken from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def extractLVmap_forlist(realmin,realmax,\
                         datadir= 'output/lcdm-map-testspace/', \
                         lvmapdir = '',mapbase='map', maskfile='', maskname='',\
                         disksizedeg = 8, NSIDELV=16, overwrite = False,\
                         diskpixlist=[],NSIDEIN=None, queue= None):
    """
    Given range of realizations and info about input map filenames,
    goes through and makes local variance maps, storing them in lvmapdir.

    datadir - where the input maps are
    lvmapdir - where to put LV maps; if empty, matches datadir
    diskpixlist - array of arrays; computed if emtpy, but can be passed
            to save time

    if queue != put (meanmap,varmap,Nmap) into queue;
    otherwise just return that tuple

    if not overwrite, only make maps if files for them don't already exist
    """
    print('In extractLVmap_forlist, rlzn=',realmin,' - ',realmax)
    if not lvmapdir:
        lvmapdir = datadir
    rlzns = np.arange(realmin,realmax+1)

    print('  getting input map filenames')
    inmaplist = [get_filename_testcase(datadir = datadir, mapbase = mapbase, number =i, stattype = 'map') for i in rlzns]

    lvbasestr ='LVd{0:02d}n{1:02d}'.format(int(disksizedeg),NSIDELV)
    print('getting output map filenames')
    outmaplist = [get_filename_testcase(datadir = lvmapdir, mapbase = mapbase, number =i, stattype = 'lvmap', maskname = maskname,extratag = lvbasestr) for i in rlzns]
    print('  reading in mask')
    if maskname:
        inmask = hp.read_map(maskfile , verbose = False)
    else:
        inmask = None
    if not len(diskpixlist):
        #print "Getting diskpixlist for rlzns",realmin,' - ',realmax
        if NSIDEIN is None:
            firstmap = hp.read_map(get_filename_testcase(datadir = datadir, mapbase = mapbase, number =realmin, stattype = 'map'), verbose = False)
            NSIDEIN = hp.get_nside(firstmap)
        diskpixlist = get_diskpixlist(NSIDEIN,NSIDELV,disksizedeg,inmask)

    Nmap = rlzns.size
    outmapdat = np.zeros((Nmap,len(diskpixlist)))
    print("  Getting LV map data for rlzns",realmin,' - ',realmax)
    t0 = time.time()
    for i in xrange(Nmap):
        if overwrite or not os.path.isfile(outmaplist[i]):
            if i%100==0:
                print("   ...extracting map",i)
            outmapdat[i,:] = extractLVmap(inmaplist[i],diskpixlist, outmaplist[i], inmask)
        else:
            if i%100==0:
                print("   ...reading map",i)
            outmapdat[i,:] = hp.read_map(outmaplist[i], verbose = False)
    t1 = time.time()
    print("  ...took {0} sec".format(t1-t0))
    print('  getting mean and variance for rlzns',realmin,' - ',realmax)
    #lvmask = getLVmap_mask(inmask,disksizedeg,NSIDELV,diskpixlist)
    meanmap = hp.ma(np.mean(outmapdat,axis=0))
    #meanmap.mask = np.logical_not(lvmask)
    #hp.mollview(meanmap,title = 'mean map')
    #plt.show()
    varmap = hp.ma(np.var(outmapdat,axis=0))
    #varmap.mask = np.logical_not(lvmask)
    #hp.mollview(varmap,title = 'variance map')
    #plt.show()
    if queue is None:
        print('  no queue, returning')
        return  meanmap,varmap,Nmap
    else:
        print('  putting mean and variance maps in queue')
        queue.put((meanmap,varmap,Nmap))
    return

# Taken from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def extractLVmap(inmapf,diskpixlist=[],outmapf='',mask = None, NSIDELV = 16, disksizedeg = 8):
    """
    inmapf - string filename of input map
    mask - healpy array of mask to use (of same NSIDE as inmap)
           used here jsut for dipole subtraction, if diskpixlist given
    diskpixlist - list of all unmasked pixels in disks around NSIDELV pix
           if diskpixlist passed, NSIDELV and disksizedeg not used
    
    if outfile given, save the map there
    """
    
    inmap = hp.read_map(inmapf, verbose = False)
    #subtract monopole and dipole
    if (mask is not None):
        inmap = hp.ma(inmap)
        inmap.mask = np.logical_not(mask)

    inmap = hp.remove_dipole(inmap, verbose=False)

    if not len(diskpixlist):
        NSIDEIN = hp.get_nside(inmap)
        diskpixlist = get_diskpixlist(NSIDEIN,NSIDELV,disksizedeg,mask)
    NPIXLV = len(diskpixlist)

    plt.show()
    outmap = np.zeros(NPIXLV)
    for i in xrange(NPIXLV):
        if len(diskpixlist[i]):
            outmap[i] = np.var(inmap[diskpixlist[i]])
    if outmapf:
        print('    Saving ',outmapf)
        hp.write_map(outmapf,outmap)
    return outmap
    
# Taken from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def get_diskpixlist(NSIDEIN,NSIDELV,disksizedeg,mask=None):
    """
    returns array of arrays; 
    each entry in outer array corresponds to a pixel in an healpy map of 
    resolution NSIDELV, and contains an array of the unmasked
    pixels in a map of resolution NSIDEIN within a disk of radius
    disksizedeg centered on that NSIDELV pixel.

    inmask - expects NSIDEIN map with 1 = unmasked
    """
    NpixLV = hp.nside2npix(NSIDELV)
    NpixIN = hp.nside2npix(NSIDEIN)
    if mask is not None:
        unmasked =np.where(mask)[0] #indices of unmasked pixels
    else:
        unmasked = np.arange(NpixIN)
    #get list of unmasked pixels in disk centered on each lower res pixel
    diskpixlist = []
    for p in xrange(NpixLV):
        alldisk = hp.query_disc(nside=NSIDEIN, vec=hp.pix2vec(NSIDELV, p), radius=np.deg2rad(disksizedeg))
        unmaskeddisk = np.intersect1d(alldisk,unmasked,assume_unique=True)
        diskpixlist.append(unmaskeddisk)
            
    return diskpixlist
    
    
# Taken from https://github.com/jessmuir/cmbanomcov_muir-adhikari-huterer
def getLVmap_mask(inmask,disksizedeg = 8, NSIDELV=16,diskpixlist=[]):
    """
    Return mask for LV masks, where for a pixel to be unmasked, more than
    10% of the disk centered on it must be unmasked in original map.
    """
    NpixLV = hp.nside2npix(NSIDELV)
    NSIDEIN = hp.get_nside(inmask)
    if not len(diskpixlist):
        #list of unmasked pixels in disk corresponding to each NSIDELV disk
        diskpixlist = get_diskpixlist(NSIDEIN,NSIDELV,disksizedeg,inmask)
    #1 means unmasked in inmask, 0 means masked
    # if at least 1 percent of pixels are unmasked, leave unmasked
    fracunmasked = np.array([float(diskpixlist[i].size)/float(hp.query_disc(nside=NSIDEIN, vec=hp.pix2vec(NSIDELV, i), radius=np.deg2rad(disksizedeg)).size) for i in xrange(NpixLV)])
    #ratio of pixels in each disk with and without mask
    
    lvmask = (fracunmasked > 0.1).astype(bool)

    return lvmask
