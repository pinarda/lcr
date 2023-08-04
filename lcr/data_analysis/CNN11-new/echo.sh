#!/bin/tcsh
### Job Name
#PBS -N CNN11_TEMPLATE
### Charging account
#PBS -A NTDD0005
### Request a resource chunk with a GPU
#PBS -l select=1:ngpus=1
### Specify that the GPUs will be V100s
#PBS -l gpu_type=v100
### Allow job to run up to 12 hours
#PBS -l walltime=12:00:00
### Route the job to the casper queue
#PBS -q casper
### Join output and error streams into single file
#PBS -j oe


setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin
cd ~/lcr/lcr/data_analysis/CNN11-new
rm -f echosave/*
git pull

conda activate my-npl-ml
python main.py --onlydata

conda activate echo
echo-run hyperparameters.yml model_config.yml

conda activate my-npl-ml
python main.py