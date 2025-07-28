#!/bin/bash                                                                                                                                                                     
NMIN=0
NMAX=100000
SIMS_100K=True
#OUTDIR="/tank/NoBackup/lherold"
OUTDIR="../data/sims/sims_1k"

# Full sky
#MASKFILE="None"

# Common mask
#MASKFILE="../data/masks/com_mask_cutoff_0.9_nside_128.fits"
#MASKLABEL="commask"

# Std. mask
MASKFILE="../data/masks/1percent_mask_v9.fits"
MASKLABEL="stdmask"

if [[ "$MASKFILE" == "None" ]]; then
    for N in `seq $NMIN $NMAX`; do
	if [[ "$SIMS_100K" == "True" ]]; then
	    MAPFILE="${OUTDIR}/map__${N}.fits"
            CORFILE="${OUTDIR}/corr_fullsky__${N}.txt"
            CLFILE="${OUTDIR}/cl_fullsky__${N}.txt"
	else
	    MAPFILE="${OUTDIR}/maps_100k/map__${N}.fits"
	    CORFILE="${OUTDIR}/corrs_fullsky_100k/corr_fullsky__${N}.txt"
	    CLFILE="${OUTDIR}/cls_fullsky_100k/cl_fullsky__${N}.txt"
	fi
        spice -mapfile $MAPFILE -corfile $CORFILE -clfile $CLFILE
    done
else
    for N in `seq $NMIN $NMAX`; do
	if [[ "$SIMS_100K" == "True" ]]; then 
            MAPFILE="${OUTDIR}/map__${N}.fits"
            CORFILE="${OUTDIR}/corr_${MASKLABEL}__${N}.txt"
            CLFILE="${OUTDIR}/cl_${MASKLABEL}__${N}.txt"
	else
	    MAPFILE="${OUTDIR}/maps_100k/map__${N}.fits"
            CORFILE="${OUTDIR}/corrs_${MASKLABEL}_100k/corr_${MASKLABEL}__${N}.txt"
            CLFILE="${OUTDIR}/cls_${MASKLABEL}_100k/cl_${MASKLABEL}__${N}.txt"
	fi
        spice -mapfile $MAPFILE -maskfile $MASKFILE -corfile $CORFILE -clfile $CLFILE
    done
fi
