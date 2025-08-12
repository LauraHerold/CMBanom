#!/bin/bash                                                                                                                                                                     
NMIN=0
NMAX=10000

# Full 100k sims
#SIMS_TYPE="full" 
#OUTDIR="/tank/NoBackup/lherold"

# Sims 1k
#SIMS_TYPE="1k" 
#OUTDIR="../data/sims/sims_1k"

# Hayley's fake-cleaned sims
SIMS_TYPE="Hayley"
MAPDIR="/tank/NoBackup/hnofi/sim_maps/LCDM/cleaned94GHz/94GHz_LCDM"
OUTDIR="/tank/NoBackup/lherold/cleaned_sims_test/94GHz_LCDM"

# Sky cut
#MASKFILE="None"
MASKFILE="../data/masks/1percent_mask_v9.fits"
MASKLABEL="stdmask"
#MASKFILE="../data/masks/com_mask_cutoff_0.9_nside_128.fits"
#MASKLABEL="commask"

if [[ "$SIMS_TYPE" == "full" ]]; then
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
fi

if [[ "$SIMS_TYPE" == "1k" ]]; then
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
fi

if [[ "$SIMS_TYPE" == "Hayley" ]]; then
    if [[ "$MASKFILE" == "None" ]]; then
        for N in $(seq $NMIN $NMAX); do
	    N0=$(printf "%05d" $N)
            MAPFILE="${MAPDIR}_${N0}.fits"
            CORFILE="${OUTDIR}_corr_fullsky__${N}.txt"
            CLFILE="${OUTDIR}_cl_fullsky__${N}.txt"
	    spice -mapfile $MAPFILE -corfile $CORFILE -clfile $CLFILE
	done
    else
	for N in $(seq $NMIN $NMAX); do
            N0=$(printf "%05d" $N)
            MAPFILE="${MAPDIR}_${N0}.fits"
            CORFILE="${OUTDIR}_corr_${MASKLABEL}__${N}.txt"
            CLFILE="${OUTDIR}_cl_${MASKLABEL}__${N}.txt"
	    spice -mapfile $MAPFILE -maskfile $MASKFILE -corfile $CORFILE -clfile $CLFILE
	done
    fi
fi
