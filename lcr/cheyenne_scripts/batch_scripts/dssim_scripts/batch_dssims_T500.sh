#!/bin/tcsh
#PBS -A NTDD0005
#PBS -N testb
#PBS -q regular
#PBS -l walltime=0:45:00
#PBS -j oe
#PBS -M apinard@ucar.edu
#PBS -l select=1:ncpus=1

module load conda
conda activate ldcpy_env

setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

python /glade/u/home/apinard/lcr/lcr/data_gathering/optimal_compression.py --var 'T500'