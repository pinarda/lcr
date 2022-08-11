#!/bin/tcsh
#PBS -A NTDD0005
#PBS -N testb
#PBS -q regular
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M apinard@ucar.edu
#PBS -l select=1:ncpus=1

module load conda
conda activate ldcpy_env

setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

python ./compute_batch.py -o '/glade/scratch/apinard/sz3/Z050_calcs.csv' -j './batch_scripts/modified_2_sz_dssim_scripts/Z050.json' -v -ld



 

