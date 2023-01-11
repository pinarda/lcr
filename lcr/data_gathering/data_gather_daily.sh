#!/bin/tcsh
#PBS -A NTDD0005
#PBS -N testb
#PBS -q regular
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M apinard@ucar.edu
#PBS -l select=1:ncpus=1

## KNOWN ISSUES: This script does not wait for all compute_batch.py jobs to finish before moving on to the next step.
## The same may hold true for optimal_compression.py. Compression script needs to be manually edited rather than using command
## line options in this file. Filesizes are not computed in this script.

# all variable lists
#set arrMonth= (ABSORB ANRAIN ANSNOW AODABS AODDUST1 AODDUST2 AODDUST3 AODVIS AQRAIN AQSNOW AREI AREL AWNC AWNI bc_a1_SRF BURDENBC BURDENDUST BURDENPOM BURDENSEASALT BURDENSO4 BURDENSOA CCN3 CDNUMC CLDHGH CLDICE CLDLIQ CLDLOW CLDMED CLDTOT CLOUD CO2 CO2_FFF CO2_LND CO2_OCN DCQ dst_a1_SRF dst_a3_SRF DTCOND DTV EXTINCT FICE FLDS FLNS FLNSC FLNT FLNTC FLUT FLUTC FREQI FREQL FREQR FREQS FSDS FSDSC FSNS FSNSC FSNT FSNTC FSNTOA FSNTOAC ICEFRAC ICIMR ICLDIWP ICLDTWP ICWMR IWC LANDFRAC LHFLX LWCF NUMICE NUMLIQ OCNFRAC OMEGA OMEGAT PBLH PHIS pom_a1_SRF PRECC PRECL PRECSC PRECSL PS PSL Q QFLX QRL QRS RELHUM SFCO2 SFCO2_FFF SFCO2_LND SFCO2_OCN SHFLX SNOWHICE SNOWHLND so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF soa_a2_SRF SOLIN SRFRAD SWCF T TAUX TAUY TGCLDIWP TGCLDLWP TMCO2 TMCO2_FFF TMCO2_LND TMCO2_OCN TMQ TOT_CLD_VISTAU TREFHT TREFHTMN TREFHTMX TROP_P TROP_T TS TSMN TSMX U10 U UQ UU V VD01 VQ VT VU VV WGUSTD WSPDSRFMX WSUB Z3)
#set arrDay= (bc_a1_SRF dst_a1_SRF dst_a3_SRF FLNS FLNSC FLUT FSNS FSNSC FSNTOA ICEFRAC LHFLX pom_a1_SRF PRECL PRECSC PRECSL PRECT PRECTMX PSL Q200 Q500 Q850 QBOT SHFLX so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF soa_a2_SRF T010 T200 T500 T850 TAUX TAUY TMQ TREFHT TREFHTMN TREFHTMX TS U010 U200 U500 U850 VBOT WSPDSRFAV Z050 Z500)

# Instructions:
# To change the daily or monthly variables to be processed, change the variable list $arrDay and $arrMonth below
# To change the time slices to be processed, change the starting time slices $time and the total number of time slices to be processed in the compute_batch commands
# To change the metrics to use in determining optimal compression level, change the metrics (option -p) to optimal_compression.py
# Ensure that ${prefix}_calcs.json and ${prefix}_diff.json are in this directory and set them up appropriately
# NOTE: Compressing only currently works for zfp compression. If the runtype is "compress" - modify the indir and ourdir variables in grab_all_TS.sh and lines 51-55.
# example : "./data_gather_daily.sh rerun random" | qsub
conda activate my-npl-ml
source  /glade/work/haiyingx/netcdf-c-4.8.1/use_nccopy.sh

# directory and filename prefix
set prefix = AllCAMdaily4
# "new" or "rerun" or "compress"
set runtype = "new"
# "fixed" or "random"
set testset = "calcs"
set alg = "br"

set arrDay= (bc_a1_SRF dst_a1_SRF dst_a3_SRF FLNS FLNSC FLUT FSNS FSNSC FSNTOA ICEFRAC LHFLX pom_a1_SRF PRECL PRECSC PRECSL PRECT PRECTMX PSL Q200 Q500 Q850 QBOT SHFLX so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF soa_a2_SRF T010 T200 T500 T850 TAUX TAUY TMQ TREFHT TREFHTMN TREFHTMX TS U010 U200 U500 U850 UBOT V200 V500 V850 VBOT WSPDSRFAV Z050 Z500)
#set arrDay= (ICEFRAC)
#set arrDay = (FLUT LHFLX PRECT TAUX TS Z500)

#git pull

rm testb*

if ($runtype == "compress") then
    ./grab_all_TS.sh
endif

# for this use -tt 1095 on line 44
# FULL DS
#set time = (0 365 730 1095 1460 1825 2190 2555 2920 3285 3650 4015 4380 4745 5110 5475 5840 6205 6570 6935 7300 7665 8030 8395 8760 9125 9490 9855 10220 10585 10950 11315 11680 12045 12410 12775 13140 13505 13870 14235 14600 14965 15330 15695 16060 16425 16790 17155 17520 17885 18250 18615 18980 19345 19710 20075 20440 20805 21170 21535 21900 22265 22630 22995 23360 23725 24090 24455 24820 25185 25550 25915 26280 26645 27010)
# 730 TIME SLICES
#set time = (0 365)
#2190 3285 4380 5475 6570 7665 8760 9855 10950 12045 13140 14235 15330 16425 17520 18615 19710 20805 21900 22995 24090 25185 26280)
set time = (0)

if ($runtype == "new" || $runtype == "compress") then
  #rm -rf ../../data/${prefix}_calcs/*
  mkdir ../../data/${prefix}_calcs
  mkdir ../../data/${prefix}_calcs/reports
  #set arrComp = (zfp_p_6 zfp_p_8 zfp_p_10 zfp_p_12 zfp_p_14 zfp_p_16 zfp_p_18 zfp_p_20 zfp_p_22 zfp_p_24 zfp_p_26)
  #set arrComp = (sz3_ROn0.1 sz3_ROn0.05 sz3_ROn0.01 sz3_ROn0.005 sz3_ROn0.001 sz3_ROn0.0005 sz3_ROn0.0001 sz3_ROn5e-05 sz3_ROn1e-05 sz3_ROn5e-06 sz3_ROn1e-06)
  set arrComp = (br_2 br_4 br_6 br_8 br_10 br_12 br_14 br_16 br_18 br_20 br_22)

  #currently requires custom json files in the current directory
  #temporary comment down to next echo line
  foreach x ($arrDay)
    foreach z ($time)
        set id = `printf "tcsh -c 'cat ${prefix}_calcs.json | sed "s/MATCHVAR/$x/" > ${prefix}_calcs_${x}.json && conda activate my-npl-ml && set prefix = ${prefix} && python ~/lcr/lcr/data_gathering/compute_batch.py -oo ~/lcr/data/${prefix}_calcs/${prefix}_daily_calcs.csv -j ${prefix}_calcs_${x}.json -ld -ts ${z} -tt 365 -v'" | qsub -A NTDD0005 -N testb -q regular -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
    end
    foreach y ($arrComp)
      echo $x
  #   set id = `printf "tcsh -c 'conda activate my-npl-ml && set prefix = ${prefix} && python ~/lcr/lcr/data_gathering/compute_batch.py -oo ~/lcr/data/\${prefix}_calcs/\${prefix}_daily_calcs_${x}.csv -j \${prefix}_calcs.json -ld -tt 10 -v'" | qsub -A NTDD0005 -N testb -q regular -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
  #   set id2 = `printf "tcsh -c 'conda activate my-npl-ml && set prefix = ${x} && python ~/lcr/lcr/data_gathering/compute_batch.py -o ~/lcr/data/\${prefix}_calcs/\${prefix}_daily_metrics_${x}.csv -j \${prefix}_diff.json -ld -tt 10 -v'" | qsub -A NTDD0005 -N testb -q regular -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
      foreach z ($time)
        set id2 = `printf "tcsh -c 'cat ${prefix}_diff.json | sed "s/MATCHME/$y/" | sed "s/MATCHVAR/$x/" > ${prefix}_diff_${y}_${x}.json && conda activate my-npl-ml && set prefix = ${prefix} && python ~/lcr/lcr/data_gathering/compute_batch.py -o ~/lcr/data/${prefix}_calcs/${prefix}_daily_metrics_${z}.csv -j ${prefix}_diff_${y}_${x}.json -ld -ts ${z} -tt 365  -v'" | qsub -A NTDD0005 -N testb -q regular -l walltime=5:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
      end
    end
  end
cat
  foreach x ($arrMonth)
    foreach z ($time)
        set id = `printf "tcsh -c 'conda activate my-npl-ml && set prefix = ${prefix} && python ~/lcr/lcr/data_gathering/compute_batch.py -oo ~/lcr/data/${prefix}_calcs/${prefix}_monthly_calcs.csv -j ${prefix}_calcs.json -ld -ts ${z} -v'" | qsub -A NTDD0005 -N testb -q regular -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
    end
    foreach y ($arrComp)
      echo $x
  #   set id = `printf "tcsh -c 'conda activate my-npl-ml && set prefix = ${prefix} && python ~/lcr/lcr/data_gathering/compute_batch.py -oo ~/lcr/data/\${prefix}_calcs/\${prefix}_daily_calcs_${x}.csv -j \${prefix}_calcs.json -ld -tt 10 -v'" | qsub -A NTDD0005 -N testb -q regular -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
  #   set id2 = `printf "tcsh -c 'conda activate my-npl-ml && set prefix = ${x} && python ~/lcr/lcr/data_gathering/compute_batch.py -o ~/lcr/data/\${prefix}_calcs/\${prefix}_daily_metrics_${x}.csv -j \${prefix}_diff.json -ld -tt 10 -v'" | qsub -A NTDD0005 -N testb -q regular -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
      foreach z ($time)
        set id2 = `printf "tcsh -c 'cat ${prefix}_diff.json | sed "s/MATCHME/$y/" > ${prefix}_diff_$y.json && conda activate my-npl-ml && set prefix = ${prefix} && python ~/lcr/lcr/data_gathering/compute_batch.py -o ~/lcr/data/${prefix}_calcs/${prefix}_monthly_metrics.csv -j ${prefix}_diff_$y.json -ld -ts ${z} -v'" | qsub -A NTDD0005 -N testb -q regular -l walltime=2:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
      end
    end
  end

  echo $id
  set split = ($id:as/./ /)
  set split2 = ($id2:as/./ /)

  sleep 60
  echo `qstat $split[1]`
  echo `qstat $split2[1]`

  set notnow = `date +%s`
  while (1)
    set now = `date +%s`
    @ diff = $now - $notnow
    echo $diff
    set out = `qstat $split[1]`
    set out2 = `qstat $split2[1]`
    if ("$out" == "" && "$out2" == "") then
      break
    endif
    sleep 10
  end
endif

if ($runtype == "calcs") then
  rm -f ../../data/${prefix}_calcs/${prefix}_daily_calcs.csv
  foreach x ($arrDay)
    foreach z ($time)
        set id = `printf "tcsh -c 'cat ${prefix}_calcs.json | sed "s/MATCHVAR/$x/" > ${prefix}_calcs_${x}.json && conda activate my-npl-ml && set prefix = ${prefix} && python ~/lcr/lcr/data_gathering/compute_batch.py -oo ~/lcr/data/${prefix}_calcs/${prefix}_daily_calcs.csv -j ${prefix}_calcs_${x}.json -ld -ts ${z} -tt 365 -v'" | qsub -A NTDD0005 -N testb -q regular -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
    end
  end

  echo $id
  set split = ($id:as/./ /)

  sleep 60
  echo `qstat $split[1]`

  set notnow = `date +%s`
  while (1)
    set now = `date +%s`
    @ diff = $now - $notnow
    echo $diff
    set out = `qstat $split[1]`
    if ("$out" == "") then
      break
    endif
    sleep 10
  end
endif

echo "calculations and metrics performed, starting to determine optimal compression parameters for each algorithm"

rm -f ../../data/${prefix}_calcs/${prefix}_daily_labels.csv
rm -f ../../data/${prefix}_calcs/${prefix}_daily_optim.csv
rm -f ../../data/${prefix}_calcs/${prefix}_daily_optim_algs.csv
rm -f ../../data/${prefix}_calcs/${prefix}_daily_df.csv
rm -f ../../data/${prefix}_calcs/reports/*
rm -f ../../data/${prefix}_calcs/${prefix}_monthly_labels.csv
rm -f ../../data/${prefix}_calcs/${prefix}_monthly_optim.csv
rm -f ../../data/${prefix}_calcs/${prefix}_monthly_optim_algs.csv
rm -f ../../data/${prefix}_calcs/${prefix}_monthly_df.csv

# local single file
# python ~/git/lcr/lcr/data_gathering/compute_batch.py -oo ~/git/lcr/data/${prefix}_calcs/${prefix}_daily_calcs.csv -j ${prefix}_calcs.json -ld
# python ~/git/lcr/lcr/data_gathering/compute_batch.py -o ~/git/lcr/data/${prefix}_calcs/${prefix}_daily_metrics.csv -j ${prefix}_diff.json -ld

foreach x ($arrDay)
  foreach z ($time)
#  python optimal_compression.py -l ../../data/${prefix}_calcs/${prefix}_daily_optim.csv -f daily -v $x -z ../../data/daily/AllCAMmonthly_filesizes.csv -m ../../data/${prefix}_calcs/${prefix}_daily_metrics.csv -a zfp -m dssim ks spatial max_spatial pcc
    set idlast = `printf "tcsh -c 'conda activate my-npl-ml && set prefix = ${prefix} && python optimal_compression.py -l ../../data/${prefix}_calcs/${prefix}_daily_optim.csv -f daily -v $x -z ../../data/${prefix}_calcs/${prefix}_filesizes.csv -m ../../data/${prefix}_calcs/${prefix}_daily_metrics_${z}.csv -a ${alg} -p dssim -t ${z} -d 0.9995'" | qsub -A NTDD0005 -N testb -q regular -l walltime=4:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
  end
end

foreach x ($arrMonth)
#  python optimal_compression.py -l ../../data/${prefix}_calcs/${prefix}_daily_optim.csv -f daily -v $x -z ../../data/daily/AllCAMmonthly_filesizes.csv -m ../../data/${prefix}_calcs/${prefix}_daily_metrics.csv -a zfp -m dssim ks spatial max_spatial pcc
  foreach z ($time)
    set idlast = `printf "tcsh -c 'conda activate my-npl-ml && set prefix = ${prefix} &&  python optimal_compression.py -l ../../data/${prefix}_calcs/${prefix}_monthly_optim.csv -f monthly -v $x -z ../../data/${prefix}_calcs/${prefix}_filesizes.csv -m ../../data/${prefix}_calcs/${prefix}_monthly_metrics_${z}.csv -a ${alg} -p dssim -t ${z} -d 0.9995'"  | qsub -A NTDD0005 -N testb -q regular -l walltime=4:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
  end
end

# Wait for optimal compression calculations to complete
echo $idlast
set split3 = ($idlast:as/./ /)

sleep 60
echo `qstat $split3[1]`

set notnow = `date +%s`
while (1)
  set now = `date +%s`
  @ diff = $now - $notnow
  echo $diff
  set out3 = `qstat $split3[1]`
  if ("$out3" == "") then
    break
  endif
  sleep 10
end

echo "optimal settings determined, comparing across algorithms"

python compare_algorithms.py -a ${alg} -v TS -l ../../data/${prefix}_calcs/${prefix}_daily_optim.csv -o ../../data/${prefix}_calcs/${prefix}_daily_optim_algs.csv
python compare_algorithms.py -a ${alg} -v TS -l ../../data/${prefix}_calcs/${prefix}_monthly_optim.csv -o ../../data/${prefix}_calcs/${prefix}_monthly_optim_algs.csv

echo "algorithms compared, creating labels"

foreach x ($arrDay)
  python histograms_of_optimal_levels.py -l ../../data/${prefix}_calcs/${prefix}_daily_optim_algs.csv -o ../../data/${prefix}_calcs/${prefix}_daily_labels.csv -v $x
end
foreach x ($arrMonth)
  python histograms_of_optimal_levels.py -l ../../data/${prefix}_calcs/${prefix}_monthly_optim_algs.csv -o ../../data/${prefix}_calcs/${prefix}_monthly_labels.csv -v $x
end

echo "labels created, creating dataframe by merging labels and calculations"

python create_dataframe.py -l ../../data/${prefix}_calcs/${prefix}_daily_labels.csv -c ../../data/${prefix}_calcs/${prefix}_daily_calcs.csv -o ../../data/${prefix}_calcs/${prefix}_daily_df.csv
python create_dataframe.py -l ../../data/${prefix}_calcs/${prefix}_monthly_labels.csv -c ../../data/${prefix}_calcs/${prefix}_monthly_calcs.csv -o ../../data/${prefix}_calcs/${prefix}_monthly_df.csv

echo "dataframe created, performing feature extraction"

python ../data_analysis/feature_selector.py -p 1 -l ../../data/${prefix}_calcs/${prefix}_daily_df.csv -o ../../data/${prefix}_calcs/${prefix}_feature_list_daily.pkl
python ../data_analysis/feature_selector.py -p 1 -l ../../data/${prefix}_calcs/${prefix}_monthly_df.csv -o ../../data/${prefix}_calcs/${prefix}_feature_list_monthly.pkl

echo "features selected, running models"

if ($testset == "random") then
  python ../data_analysis/models.py -d ../../data/${prefix}_calcs/${prefix}_daily_df.csv -e rf -t 1 -r ../../data/${prefix}_calcs/reports/ -f ../../data/${prefix}_calcs/${prefix}_feature_list_daily.pkl
else
  # select models to run: ada, rf, nn, svm, lda, qda, agg
  python ../data_analysis/models.py -d ../../data/${prefix}_calcs/${prefix}_daily_df.csv -m ../../data/${prefix}_calcs/${prefix}_monthly_df.csv -e rf -t 0 -r ../../data/${prefix}_calcs/reports/ -f ../../data/${prefix}_calcs/${prefix}_feature_list_daily.pkl
endif



#add a step here for formatting the tables and reports before converting to latex and saving

echo "models completed, formatting report"
foreach p (rf)
  python tably.py ../../data/${prefix}_calcs/reports/${p}_report_5.csv -o ../../data/${prefix}_calcs/reports/${p}_report.tex
end