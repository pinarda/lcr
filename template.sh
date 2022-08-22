#!/bin/tcsh
#PBS -A PROJECTCODE
#PBS -N test_rscript
#PBS -q regular
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -M YOUREMAIL@ucar.edu
#PBS -l select=1:ncpus=1

module load R/3.5.2
Rscript HelloWorld.R