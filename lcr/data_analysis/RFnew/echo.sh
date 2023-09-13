#!/bin/tcsh
### Job Name
#PBS -N RFmain
### Charging account
#PBS -A NTDD0005
### Request a resource chunk with a GPU
#PBS -l select=1:ncpus=1
### Allow job to run up to 12 hours
#PBS -l walltime=12:00:00
### Route the job to the casper queue
#PBS -q casper
### Join output and error streams into single file
#PBS -j oe

module load conda
setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin
cd ~/lcr/lcr/data_analysis/RFnew
rm -f echosave/*
git pull
set models = ("rf")
set features = ("ns_con_var")
#set features = ("ns_con_var" "ew_con_var" "w_e_first_differences" "n_s_first_differences" "fftratio" "magnitude_range")
set ids = ()

foreach model ($models)
  if ($model == "cnn") then
    conda activate my-npl-ml
    python main.py --onlydata -j RF_TEMPLATE.json -m "${model}"

    conda activate echo
    alias postcmd 'set start_time=`date +%s`'
    alias precmd 'set end_time=`date +%s`; @ cmd_time= $end_time - $start_time; echo took $cmd_time seconds'

    # time the run and pipe the output to a file
    time echo-run hyperparameters.yml model_config.yml > echosave/echo-run.out

    # time the run
    conda activate my-npl-ml
    python main.py -j RF_TEMPLATE.json -m "${model}" --testset TESTSET
  endif
  if ($model == "rf") then
    conda activate my-npl-ml
    foreach feature ($features)
      # save all ids for line 48
      set newid = `printf "tcsh -c 'setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin && module load conda && conda activate my-npl-ml && python main.py -j RF_TEMPLATE.json -m "${model}" -f "${feature}"'" | qsub -A NTDD0005 -N feature -q casper -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
      # remove the period and everything after it
      set newid = `echo $newid | sed 's/\..*//'`
      set ids = ( $ids $newid )
    end

    # I need a long string consisting of all the ids in the ids array, separated by commas, in tcsh
    set joblist = `printf '%s,' "$ids[-]"`
    # remove any trailing commas (may be more than one)
    set joblist = `echo $joblist | sed 's/,*$//'`

    printf '%s,' ${joblist}

    `printf "tcsh -c 'setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin && module load conda && conda activate my-npl-ml && python main.py -j RF_TEMPLATE.json -m "${model}" --testset TESTSET -l ${features}'" | qsub -A NTDD0005 -N final -q casper -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ngpus=1:mem=40GB -l gpu_type=v100`
  endif
end
