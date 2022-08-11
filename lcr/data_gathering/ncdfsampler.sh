#!/bin/tcsh

module load nco
# set arrname = (PRECSL TAUY soa_a2_SRF TAUX CDNUMC TGCLDLWP PRECL soa_a1_SRF SHFLX dst_a3_SRF pom_a1_SRF bc_a1_SRF TGCLDIWP dst_a1_SRF LHFLX QFLX LWCF so4_a2_SRF WGUSTD so4_a3_SRF CLDMED so4_a1_SRF PBLH CLDHGH BURDENSEASALT SFCO2 FLNS BURDENDUST U10 SWCF CLDTOT WSPDSRFMX CLDLOW BURDENPOM BURDENBC AODDUST3 BURDENSO4 BURDENSOA AODVIS TMQ FSNS AODABS AODDUST1 FLNSC FSDS SRFRAD FSNT FSNTOA FLDS FLNT FLUT PRECC FSNSC TROP_P FLNTC FLUTC FSNTC FSNTOAC SFCO2_OCN TREFHTMN TSMN TSMX TREFHTMX TMCO2 TS TMCO2_FFF TREFHT PS TMCO2_LND TROP_T SFCO2_FFF PSL PHIS SOLIN SFCO2_LND OCNFRAC SNOWHLND SNOWHICE PRECSC ICEFRAC LANDFRAC)
set arrMonth= (ABSORB ANRAIN ANSNOW AODABS AODDUST1 AODDUST2 AODDUST3 AODVIS AQRAIN AQSNOW AREI AREL AWNC AWNI bc_a1_SRF BURDENBC BURDENDUST BURDENPOM BURDENSEASALT BURDENSO4 BURDENSOA CCN3 CDNUMC CLDHGH CLDICE CLDLIQ CLDLOW CLDMED CLDTOT CLOUD CO2 CO2_FFF CO2_LND CO2_OCN DCQ dst_a1_SRF dst_a3_SRF DTCOND DTV EXTINCT FICE FLDS FLNS FLNSC FLNT FLNTC FLUT FLUTC FREQI FREQL FREQR FREQS FSDS FSDSC FSNS FSNSC FSNT FSNTC FSNTOA FSNTOAC ICEFRAC ICIMR ICLDIWP ICLDTWP ICWMR IWC LANDFRAC LHFLX LWCF NUMICE NUMLIQ OCNFRAC OMEGA OMEGAT PBLH PHIS pom_a1_SRF PRECC PRECL PRECSC PRECSL PS PSL Q QFLX QRL QRS RELHUM SFCO2 SFCO2_FFF SFCO2_LND SFCO2_OCN SHFLX SNOWHICE SNOWHLND so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF soa_a2_SRF SOLIN SRFRAD SWCF T TAUX TAUY TGCLDIWP TGCLDLWP TMCO2 TMCO2_FFF TMCO2_LND TMCO2_OCN TMQ TOT_CLD_VISTAU TREFHT TREFHTMN TREFHTMX TROP_P TROP_T TS TSMN TSMX U10 U UQ UU V VD01 VQ VT VU VV WGUSTD WSPDSRFMX WSUB Z3)
set arrDay= (bc_a1_SRF dst_a1_SRF dst_a3_SRF FLNS FLNSC FLUT FSNS FSNSC FSNTOA ICEFRAC LHFLX pom_a1_SRF PRECL PRECSC PRECSL PRECT PRECTMX PSL Q200 Q500 Q850 QBOT SHFLX so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF soa_a2_SRF T010 T200 T500 T850 TAUX TAUY TMQ TREFHT TREFHTMN TREFHTMX TS U010 U200 U500 U850 VBOT WSPDSRFAV Z050 Z500)
set compressDirs= (10 12 14 16 18 20 22 24 26)



foreach x ($arrDay)
  ncks -O -v $x,lat,lon -d time,1,365,365 /glade/p/cisl/asap/abaker/pepsi/ens_31/orig/daily/b.e11.BRCP85C5CNBDRD.f09_g16.031.cam.h1.${x}.20060101-20801231.nc /glade/u/home/apinard/lcr/data/minidata/minidata_daily_${x}.nc
  foreach y ($compressDirs)
    ncks -O -v $x,lat,lon -d time,1,365,365 /glade/p/cisl/asap/CAM_lossy_test_data_31/research/daily_zfp_hdf5/zfp_p_${y}/b.e11.BRCP85C5CNBDRD.f09_g16.031.cam.h1.${x}.20060101-20071231.nc /glade/u/home/apinard/lcr/data/minidata/compressed/zfp/zfp_p_${y}/minidata_daily_${x}.nc
  end
end

#foreach x ($arrMonth)
#  ncks -O -v $x,lat,lon -d time,1,365,91 /glade/p/cisl/asap/abaker/pepsi/ens_31/orig/monthly/b.e11.BRCP85C5CNBDRD.f09_g16.031.cam.h0.${x}.200601-208012.nc /glade/scratch/apinard/minidata/minidata_monthly_${x}.nc
#end