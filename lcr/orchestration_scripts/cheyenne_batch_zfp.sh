#!/bin/tcsh
#PBS -A NTDD0004
#PBS -N testb
#PBS -q regular
#PBS -l walltime=0:20:00
#PBS -j oe
#PBS -M abaker@ucar.edu
#PBS -l select=1:ncpus=1

module load conda
conda activate npl

setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin

python ./compute_batch.py -o '/glade/scratch/abaker/testzfp.csv' -j 'sample-zfp.json' -tt 1 -ld



    

