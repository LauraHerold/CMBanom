import CMBanom

# Parameters
nside = 128
N_start = 0
N_maps = 100000
cl_fn = "../data/real/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
outdir = "/tank/NoBackup/lherold/maps_100k/"

CMBanom.gen_maps_from_cls(cldatfile=cl_fn, outdir=outdir, Nside=nside, N_start=N_start, N_maps=N_maps, lmax=200, regen=True, returnoutf=False)
