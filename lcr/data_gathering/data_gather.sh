#!/bin/tcsh
#PBS -A NTDD0005
#PBS -N testb
#PBS -q regular
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M apinard@ucar.edu
#PBS -l select=1:ncpus=1

# all variable lists
#set arrMonth= (ABSORB ANRAIN ANSNOW AODABS AODDUST1 AODDUST2 AODDUST3 AODVIS AQRAIN AQSNOW AREI AREL AWNC AWNI bc_a1_SRF BURDENBC BURDENDUST BURDENPOM BURDENSEASALT BURDENSO4 BURDENSOA CCN3 CDNUMC CLDHGH CLDICE CLDLIQ CLDLOW CLDMED CLDTOT CLOUD CO2 CO2_FFF CO2_LND CO2_OCN DCQ dst_a1_SRF dst_a3_SRF DTCOND DTV EXTINCT FICE FLDS FLNS FLNSC FLNT FLNTC FLUT FLUTC FREQI FREQL FREQR FREQS FSDS FSDSC FSNS FSNSC FSNT FSNTC FSNTOA FSNTOAC ICEFRAC ICIMR ICLDIWP ICLDTWP ICWMR IWC LANDFRAC LHFLX LWCF NUMICE NUMLIQ OCNFRAC OMEGA OMEGAT PBLH PHIS pom_a1_SRF PRECC PRECL PRECSC PRECSL PS PSL Q QFLX QRL QRS RELHUM SFCO2 SFCO2_FFF SFCO2_LND SFCO2_OCN SHFLX SNOWHICE SNOWHLND so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF soa_a2_SRF SOLIN SRFRAD SWCF T TAUX TAUY TGCLDIWP TGCLDLWP TMCO2 TMCO2_FFF TMCO2_LND TMCO2_OCN TMQ TOT_CLD_VISTAU TREFHT TREFHTMN TREFHTMX TROP_P TROP_T TS TSMN TSMX U10 U UQ UU V VD01 VQ VT VU VV WGUSTD WSPDSRFMX WSUB Z3)
#set arrDay= (bc_a1_SRF dst_a1_SRF dst_a3_SRF FLNS FLNSC FLUT FSNS FSNSC FSNTOA ICEFRAC LHFLX pom_a1_SRF PRECL PRECSC PRECSL PRECT PRECTMX PSL Q200 Q500 Q850 QBOT SHFLX so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF soa_a2_SRF T010 T200 T500 T850 TAUX TAUY TMQ TREFHT TREFHTMN TREFHTMX TS U010 U200 U500 U850 VBOT WSPDSRFAV Z050 Z500)

# Instructions:
# To change the daily or monthly variables to be processed, change the variable list arrDay and arrMonth below on lines 24 and 25
# To change the time slices to be processed, change the starting time slices in line 27 and the total number of time slices to be processed on lines 33 and 40
# To change the metrics to use in determining optimal compression level, change the metrics (option -p) on line 74 and 100
# example : "./data_gather.sh rerun random" | qsub
conda activate my-npl-ml
set prefix = test3D
set runtype = "new"
set testset = "random"

set arrDay= ()
set arrMonth= (T)

rm -rf ../../data/${prefix}_calcs/*

if ($runtype == "new") then
  mkdir ../../data/${prefix}_calcs
  mkdir ../../data/${prefix}_calcs/reports
  set arrComp = (zfp_p_10 zfp_p_12 zfp_p_14 zfp_p_16 zfp_p_18 zfp_p_20 zfp_p_22 zfp_p_24 zfp_p_26)
  # for this use -tt 1095 on line 44
  #set time = (0 1095 2190 3285 4380 5475 6570 7665 8760 9855 10950 12045 13140 14235 15330 16425 17520 18615 19710 20805 21900 22995 24090 25185 26280)
  set time = (0)

  #currently requires custom json files in the current directory
  #temporary comment down to next echo line
  foreach x ($arrDay)
    foreach z ($time)
        set id = `printf "tcsh -c 'conda activate my-npl-ml && set prefix = ${prefix} && python ~/lcr/lcr/data_gathering/compute_batch.py -oo ~/lcr/data/${prefix}_calcs/${prefix}_daily_calcs.csv -j ${prefix}_calcs.json -ld -ts ${z} -tt 1095 -v'" | qsub -A NTDD0005 -N testb -q regular -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
    end
    foreach y ($arrComp)
      echo $x
  #   set id = `printf "tcsh -c 'conda activate my-npl-ml && set prefix = ${prefix} && python ~/lcr/lcr/data_gathering/compute_batch.py -oo ~/lcr/data/\${prefix}_calcs/\${prefix}_daily_calcs_${x}.csv -j \${prefix}_calcs.json -ld -tt 10 -v'" | qsub -A NTDD0005 -N testb -q regular -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
  #   set id2 = `printf "tcsh -c 'conda activate my-npl-ml && set prefix = ${x} && python ~/lcr/lcr/data_gathering/compute_batch.py -o ~/lcr/data/\${prefix}_calcs/\${prefix}_daily_metrics_${x}.csv -j \${prefix}_diff.json -ld -tt 10 -v'" | qsub -A NTDD0005 -N testb -q regular -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
      foreach z ($time)
        set id2 = `printf "tcsh -c 'cat ${prefix}_diff.json | sed "s/MATCHME/$y/" > ${prefix}_diff_$y.json && conda activate my-npl-ml && set prefix = ${prefix} && python ~/lcr/lcr/data_gathering/compute_batch.py -o ~/lcr/data/${prefix}_calcs/${prefix}_daily_metrics.csv -j ${prefix}_diff_$y.json -ld -ts ${z} -v'" | qsub -A NTDD0005 -N testb -q regular -l walltime=2:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
      end
    end
  end

  foreach x ($arrMonth)
    foreach z ($time)
        set id = `printf "tcsh -c 'conda activate my-npl-ml && set prefix = ${prefix} && python ~/lcr/lcr/data_gathering/compute_batch.py -oo ~/lcr/data/${prefix}_calcs/${prefix}_monthly_calcs.csv -j ${prefix}_calcs.json -ld -ts ${z} -tt 1095 -v'" | qsub -A NTDD0005 -N testb -q regular -l walltime=12:00:00 -j oe -M apinard@ucar.edu -l select=1:ncpus=1`
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
#  python optimal_compression.py -l ../../data/${prefix}_calcs/${prefix}_daily_optim.csv -f daily -v $x -z ../../data/daily/daily_filesizes.csv -m ../../data/${prefix}_calcs/${prefix}_daily_metrics.csv -a zfp -m dssim ks spatial max_spatial pcc
  python optimal_compression.py -l ../../data/${prefix}_calcs/${prefix}_daily_optim.csv -f daily -v $x -z ../../data/daily/daily_filesizes.csv -m ../../data/${prefix}_calcs/${prefix}_daily_metrics.csv -a zfp -p dssim
end
foreach x ($arrMonth)
#  python optimal_compression.py -l ../../data/${prefix}_calcs/${prefix}_daily_optim.csv -f daily -v $x -z ../../data/daily/daily_filesizes.csv -m ../../data/${prefix}_calcs/${prefix}_daily_metrics.csv -a zfp -m dssim ks spatial max_spatial pcc
  python optimal_compression.py -l ../../data/${prefix}_calcs/${prefix}_monthly_optim.csv -f monthly -v $x -z ../../data/monthly/monthly_filesizes.csv -m ../../data/${prefix}_calcs/${prefix}_monthly_metrics.csv -a zfp -p dssim
end

echo "optimal settings determined, comparing across algorithms"

python compare_algorithms.py -a zfp -v TS -l ../../data/${prefix}_calcs/${prefix}_daily_optim.csv -o ../../data/${prefix}_calcs/${prefix}_daily_optim_algs.csv
python compare_algorithms.py -a zfp -v TS -l ../../data/${prefix}_calcs/${prefix}_monthly_optim.csv -o ../../data/${prefix}_calcs/${prefix}_monthly_optim_algs.csv

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

echo "dataframe created, running models"

if ($testset == "random") then
  python ../data_analysis/models.py -m ../../data/${prefix}_calcs/${prefix}_monthly_df.csv -e rf nn -t 1 -r ../../data/${prefix}_calcs/reports/
else
  python ../data_analysis/models.py -d ../../data/${prefix}_calcs/${prefix}_daily_df.csv -m ../../data/${prefix}_calcs/${prefix}_monthly_df.csv -e rf nn -t 0 -r ../../data/${prefix}_calcs/reports/
endif



#add a step here for formatting the tables and reports before converting to latex and saving

echo "models completed, formatting report"
foreach p (rf nn)
  python tably.py ../../data/${prefix}_calcs/reports/${p}_report.csv -o ../../data/${prefix}_calcs/reports/${p}_report.tex
end