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

setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin
cd ~/lcr2/lcr/lcr/data_analysis/RFnew
rm -f echosave/*
git pull
conda activate mynpl2023a
set ids = ()

rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*
rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*
rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5

set model = MODEL

set features = ("ns_con_var" "ew_con_var" "w_e_first_differences" "n_s_first_differences")
#set model = "rf"
#
if ($model == "rf") then
  foreach feature ($features)
    set newid = `printf "tcsh -c 'setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin && conda activate mynpl2023a && cd ~/lcr2/lcr/lcr/data_analysis/RFnew && python main3.py -j run_casper_TEMPLATE.json -d JOBID -m rf -r dssim -f "${feature}" --onlydata True --runonlydata True --labelsonly True'" | qsub -A NTDD0005 -N ${feature} -q casper -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`

    set newid = `echo $newid | sed 's/\..*//'`
    printf '%s,' ${newid}
    set ids = ( $ids $newid )
  end

  set features_csv = `echo $features | sed 's/ /,/g'`
  echo "tcsh -c 'setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin && conda activate mynpl2023a && cd ~/lcr2/lcr/lcr/data_analysis/RFnew && python main3.py -j run_casper_TEMPLATE.json -d JOBID -m rf -r dssim --testset TESTSET -l "${features_csv}" --onlydata False --runonlydata False --labelsonly True'" | qsub -W depend=afterok:${newid} -A NTDD0005 -N final -q casper -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ngpus=1:mem=40GB -l gpu_type=v100
else
  set newid = `printf "tcsh -c 'setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin && conda activate mynpl2023a && cd ~/lcr2/lcr/lcr/data_analysis/RFnew && python main3.py -j run_casper_TEMPLATE.json -d JOBID -m cnn -r dssim --onlydata True --runonlydata True --labelsonly True'" | qsub -A NTDD0005 -N ${feature} -q casper -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
  echo "tcsh -c 'setenv HDF5_PLUGIN_PATH /glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin && conda activate mynpl2023a && cd ~/lcr2/lcr/lcr/data_analysis/RFnew && python main3.py -j run_casper_TEMPLATE.json -d JOBID -m cnn -r dssim --testset TESTSET --onlydata False --runonlydata False --labelsonly True'" | qsub -W depend=afterok:${newid} -A NTDD0005 -N final -q casper -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ngpus=1:mem=40GB -l gpu_type=v100

#  python main3.py --onlydata True --runonlydata True --metric dssim -j run_casper_TEMPLATE.json --labelsonly True --model cnn
#  python main3.py --onlydata False --runonlydata False --metric dssim -j run_casper_TEMPLATE.json --labelsonly True --model cnn
endif





#
##
##rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*.npy
##rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*.npy
##rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5
#python main2.py --onlydata True --runonlydata True --metric spre -j RF_template_spre.json --labelsonly True
#python main2.py --onlydata False --runonlydata False --metric spre -j RF_template_spre.json --labelsonly True
#
##rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*.npy
##rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*.npy
##rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5
#python main2.py --onlydata True --runonlydata True --metric pcc -j RF_template_pcc.json --labelsonly True
#python main2.py --onlydata False --runonlydata False --metric pcc -j RF_template_pcc.json --labelsonly True
#
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5
#python main2.py --onlydata True --runonlydata True --metric ks -j RF_template_ks.json --labelsonly True
#python main2.py --onlydata False --runonlydata False --metric ks -j RF_template_ks.json --labelsonly True