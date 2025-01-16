#!/bin/bash                                                                                                                                                                     
NMIN=0
NMAX=1000
OUTDIR="/tank/NoBackup/lherold/sims"
# Full sky
MASKFILE="None"
# Common mask
##MASKFILE="../data/masks/common-Mask-Int_cutoff0.9_Nside128.fits"
##MASKLABEL="commask"
# Std. mask
##MASKFILE="../data/masks/stdv_mask_1percent.fits_v2.fits"
##MASKLABEL="stdmask"

if [[ "$MASKFILE" == "None" ]]; then
    for N in `seq $NMIN $NMAX`; do
        MAPFILE="${OUTDIR}/map__${N}.fits"
        CORFILE="${OUTDIR}/corr_fullsky__${N}.txt"
        CLFILE="${OUTDIR}/cl_fullsky__${N}.txt"
        spice -mapfile $MAPFILE -corfile $CORFILE -clfile $CLFILE
    done
else
    for N in `seq $NMIN $NMAX`; do
        MAPFILE="${OUTDIR}/map__${N}.fits"
        CORFILE="${OUTDIR}/corr_${MASKLABEL}__${N}.txt"
        CLFILE="${OUTDIR}/cl_${MASKLABEL}__${N}.txt"
        spice -mapfile $MAPFILE -maskfile $MASKFILE -corfile $CORFILE -clfile $CLFILE
    done
fi
