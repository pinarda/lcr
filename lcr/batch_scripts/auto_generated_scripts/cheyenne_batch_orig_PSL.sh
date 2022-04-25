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

python ../compute_batch.py -oo '/glade/scratch/apinard/orig_calcs_PSL.csv' -j 'orig_calcs_PSL.json' -v -ld





