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

set prefix=TS

mkdir ../../data/$prefix
mkdir ../../data/$prefix_calcs
#mkdir ../../data/$prefix_calcs/reports

set arrDay= (TS)

#remote mutiple files
foreach x ($arrDay)
#  echo $x
  python ~/lcr/lcr/data_gathering/compute_batch.py -oo ~/lcr/data/$prefix/$prefix_daily_calcs_${x}.csv -j $prefix_calcs.json -ld -tt 100
  python ~/lcr/lcr/data_gathering/compute_batch.py -o ~/lcr/data/$prefix/$prefix_daily_metrics_${x}.csv -j $prefix_diff.json -ld -tt 100
end

# local single file
# python ~/git/lcr/lcr/data_gathering/compute_batch.py -oo ~/git/lcr/data/$prefix_calcs/$prefix_daily_calcs.csv -j $prefix_calcs.json -ld
# python ~/git/lcr/lcr/data_gathering/compute_batch.py -o ~/git/lcr/data/$prefix_calcs/$prefix_daily_metrics.csv -j $prefix_diff.json -ld

foreach x ($arrDay)
  python optimal_compression.py -l ../../data/$prefix_calcs/$prefix_daily_optim.csv -f daily -v $x -z ../../data/daily/daily_filesizes.csv -m ../../data/$prefix_calcs/$prefix_daily_metrics.csv -a zfp
end

python compare_algorithms.py -a zfp -v TS -l ../../data/$prefix_calcs/$prefix_daily_optim.csv -o ../../data/$prefix_calcs/$prefix_daily_optim_algs.csv

foreach x ($arrDay)
  python histograms_of_optimal_levels.py -l ../../data/$prefix_calcs/$prefix_daily_optim_algs.csv -o ../../data/$prefix_calcs/$prefix_daily_labels.csv -v $x
end

python create_dataframe.py -l ../../data/$prefix_calcs/$prefix_daily_labels.csv -c ../../data/$prefix_calcs/$prefix_daily_calcs.csv -o ../../data/$prefix_calcs/$prefix_daily_df.csv
python ../data_analysis/models.py -d ../../data/$prefix_calcs/$prefix_daily_df.csv -e rf nn