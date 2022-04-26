#!/bin/tcsh
#PBS -A NTDD0005
#PBS -N testb
#PBS -q regular
#PBS -l walltime=0:20:00
#PBS -j oe
#PBS -M abaker@ucar.edu
#PBS -l select=1:ncpus=1

module load conda
conda activate my-npl-ldc

setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

python ./optimal_compression.py --var 'so4_a2_SRF'