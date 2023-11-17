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

#module load conda
setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin
cd ~/lcr2/lcr/lcr/data_analysis/RFnew
rm -f echosave/*
git pull
set models = ("cnn" "rf")
set features = ("ns_con_var" "ew_con_var" "w_e_first_differences" "n_s_first_differences" "fftratio")
#set features = ("ns_con_var" "ew_con_var" "w_e_first_differences" "n_s_first_differences" "fftratio" "magnitude_range")
set ids = ()

foreach model ($models)
  if ($model == "cnn") then
    conda activate my-npl-ml
    python main.py --onlydata True -j RF_TEMPLATE.json -m "${model}"

    conda activate echo
    alias postcmd 'set start_time=`date +%s`'
    alias precmd 'set end_time=`date +%s`; @ cmd_time= $end_time - $start_time; echo took $cmd_time seconds'

    # time the run and pipe the output to a file
    time echo-run hyperparameters.yml model_config.yml > echosave/echo-run.out

    # time the run
    conda activate my-npl-ml
    python main.py -j RF_TEMPLATE.json -m "${model}" --testset TESTSET -r METRIC
  endif
  if ($model == "rf") then
    conda activate my-npl-ml
    foreach feature ($features)
      # save all ids for line 48
#      set newid = `printf "tcsh -c 'setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin && module load conda && conda activate my-npl-ml && python main.py -j RF_TEMPLATE.json -d JOBID -x none -m "${model}" -r METRIC -x TRANSFORM -f "${feature}"'" | qsub -A NTDD0005 -N ${feature} -q casper -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
      set newid = `printf "tcsh -c 'setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin && conda activate my-npl-ml && python main.py -j RF_TEMPLATE.json -d JOBID -x none -m "${model}" -r METRIC -x TRANSFORM -f "${feature}"'" | qsub -A NTDD0005 -N ${feature} -q casper -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`

      # remove the period and everything after it
      set newid = `echo $newid | sed 's/\..*//'`
      printf '%s,' ${newid}
      set ids = ( $ids $newid )
    end

    # I need a long string consisting of all the ids in the ids array, separated by commas, in tcsh
    #set joblist = `printf '%s,' "$ids[-]"`
    # remove any trailing commas (may be more than one)
    #set joblist = `echo $joblist | sed 's/,*$//'`

    #printf '%s,' ${joblist}

    # look through the following string and replace MODEL with the model name, and features with the features
#    set str = 'setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin && module load conda && conda activate my-npl-ml && python main.py -j RF_TEMPLATE.json -m "${model}" --testset TESTSET -l ${features}'
#    set str = `echo $str | sed "s/MODEL/${model}/"`
#    set str = `echo $str | sed "s/FEATURES/${features}/"`
#    printf '%s\n' ${str}
    set features_csv = `echo $features | sed 's/ /,/g'`

    echo "tcsh -c 'setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin && conda activate my-npl-ml && python main.py -j RF_TEMPLATE.json -x none -d JOBID -m "${model}" -r METRIC --testset TESTSET -x TRANSFORM -l "${features_csv}"'" | qsub -W depend=afterok:${newid} -A NTDD0005 -N final -q casper -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ngpus=1:mem=40GB -l gpu_type=v100
  endif
end
