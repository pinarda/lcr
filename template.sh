#!/bin/tcsh
#PBS -A NTDD0005
#PBS -N test_rscript
#PBS -q regular
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -M apinard@ucar.edu
#PBS -l select=1:ncpus=1

module load R/3.5.2
Rscript HelloWorld.R