#!/bin/bash

# Run from the directory containing this launcher and the config files.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir" || exit 1
config_dir="."

# Loop over the single-variable and multi-variable config files
for config_file in "$config_dir"/rotated_config_*.json "$config_dir"/multi_config_*.json; do
    # Extract the config file name without extension for the job script name
    config_name=$(basename "$config_file" .json)
    job_script="${config_name}.sh"

    # Use the matching modeling entry point for each config family
    if [[ "$config_name" == multi_config_* ]]; then
        main_script="newmain_derecho.py"
    else
        main_script="newmain.py"
    fi

    # Create a new job script for each config file
    cat <<EOF > "$job_script"
#!/bin/tcsh
### Job Name
#PBS -N CNN11_${config_name}_1000
### Charging account
#PBS -A NTDD0005
### Request a resource chunk with a GPU
#PBS -l select=1:ngpus=1:mem=384GB
### Specify the GPU type
#PBS -l gpu_type=a100
### Allow job to run up to 12 hours
#PBS -l walltime=12:00:00
### Route the job to the casper queue
#PBS -q casper
### Join output and error streams into single file
#PBS -j oe

source /etc/csh.cshrc
module load conda
conda activate my-npl-2023a

setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin
cd /glade/derecho/scratch/apinard/lcr/lcr/paper_submit
mkdir -p data/trees

python $main_script -c $config_file
EOF

    # Submit the job script
    qsub "$job_script"
    echo "Submitted job for $config_file using script $job_script"
done
