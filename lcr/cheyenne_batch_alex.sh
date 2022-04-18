#!/bin/tcsh
#PBS -A NTDD0005
#PBS -N testb
#PBS -q regular
#PBS -l walltime=0:20:00
#PBS -j oe
#PBS -M apinard@ucar.edu
#PBS -l select=1:ncpus=1:mem=109GB

module load conda
conda activate ldcpy_env

setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

python ./compute_batch.py -o '/glade/scratch/apinard/chey_calcs_daily.csv' -j 'alex.json' -tt 10 -ts 10 -v -ld "/glade/u/home/apinard/newldcpy/ldcpy"



 

