#!/bin/tcsh
#PBS -A NTDD0004
#PBS -N testb
#PBS -q regular
#PBS -l walltime=0:20:00
#PBS -j oe
#PBS -M abaker@ucar.edu
#PBS -l select=1:ncpus=1:mem=109GB

module load conda
conda activate my-npl-ldc

setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

python ./compute_batch.py -o '/glade/scratch/abaker/chey_bg_daily.csv' -j 'bg_daily.json' -tt 10



 

