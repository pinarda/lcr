#!/bin/tcsh
#PBS -A NTDD0005
#PBS -N testb
#PBS -q regular
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M apinard@ucar.edu
#PBS -l select=1:ncpus=1

module load conda
conda activate my-npl-ml

setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin

python ../data_gathering/compute_batch.py -o '/glade/scratch/apinard/ICWMR.csv' -j 'ICWMR.json' -tt 1 -ld