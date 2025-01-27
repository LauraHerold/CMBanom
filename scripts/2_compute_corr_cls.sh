#!/bin/bash                                                                                                                                                                     
NMIN=10000
NMAX=20000
OUTDIR="/tank/NoBackup/lherold"
# Full sky
#MASKFILE="None"
# Common mask
MASKFILE="../data/masks/common-Mask-Int_cutoff0.9_Nside128.fits"
MASKLABEL="commask"
# Std. mask
##MASKFILE="../data/masks/stdv_mask_1percent.fits_v3.fits"
##MASKLABEL="stdmask"

if [[ "$MASKFILE" == "None" ]]; then
    for N in `seq $NMIN $NMAX`; do
        MAPFILE="${OUTDIR}/maps_100k/map__${N}.fits"
        CORFILE="${OUTDIR}/corrs_fullsky_100k/corr_fullsky__${N}.txt"
        CLFILE="${OUTDIR}/cls_fullsky_100k/cl_fullsky__${N}.txt"
        spice -mapfile $MAPFILE -corfile $CORFILE -clfile $CLFILE
    done
else
    for N in `seq $NMIN $NMAX`; do
        MAPFILE="${OUTDIR}/maps_100k/map__${N}.fits"
        CORFILE="${OUTDIR}/corrs_${MASKLABEL}_100k/corr_${MASKLABEL}__${N}.txt"
        CLFILE="${OUTDIR}/cls_${MASKLABEL}_100k/cl_${MASKLABEL}__${N}.txt"
        spice -mapfile $MAPFILE -maskfile $MASKFILE -corfile $CORFILE -clfile $CLFILE
    done
fi
