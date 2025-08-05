#!/bin/bash

# Path to the directory containing config files
config_dir="." # Change this if your config files are in a different directory

# Specify the single config file to process
config_file="$config_dir/multi_config_1.json"

# Check if the specified config file exists
if [ -f "$config_file" ]; then
    # Extract the config file name without extension for the job script name
    config_name=$(basename "$config_file" .json)
    job_script="${config_name}.sh"

    # Create a new job script for the specified config file
    cat <<EOF > "$job_script"
#!/bin/tcsh
### Job Name
#PBS -N CNN11_${config_name}_1000
### Charging account
#PBS -A NTDD0005
### Request a resource chunk with a GPU
#PBS -l select=1:ngpus=1
### Specify that the GPUs will be V100s
#PBS -l gpu_type=a100
### Allow job to run up to 12 hours
#PBS -l walltime=12:00:00
### Route the job to the casper queue
#PBS -q casper
### Join output and error streams into single file
#PBS -j oe

conda activate my-npl-2023a

setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin
cd /glade/derecho/scratch/apinard/lcr3/lcr/modeling

python newmain.py -c $config_file
EOF

    # Submit the job script
    qsub "$job_script"
    echo "Submitted job for $config_file using script $job_script"
else
    echo "Config file $config_file not found!"
fi
