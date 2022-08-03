#!/bin/tcsh

#./clean.sh

# known issues so far:
# 1. the script does not work for the AODDUST2.json file - divide by zero issue
# 2. the script does not work for the CLDICE.json (3D) file
# 3. the script is very slow for the 3D vars see ABSORB, Z3 etc.
#set arrname = (OMEGAT OMEGA VU VQ V VT VV UQ U UU QRL CCN3 ABSORB EXTINCT RELHUM QRS Q DCQ DTCOND CLOUD Z3 VD01 CLDICE DTV T ICLDTWP NUMICE TOT_CLD_VISTAU CLDLIQ IWC ICLDIWP ICIMR CO2 ICWMR WSUB CO2_FFF NUMLIQ FICE ANSNOW CO2_OCN CO2_LND AQSNOW AWNI AREI FREQS FREQI ANRAIN AWNC AQRAIN AREL FREQL FREQR)
set arrALL= (ABSORB ANRAIN ANSNOW AODABS AODDUST1 AODDUST2 AODDUST3 AODVIS AQRAIN AQSNOW AREI AREL AWNC AWNI bc_a1_SRF BURDENBC BURDENDUST BURDENPOM BURDENSEASALT BURDENSO4 BURDENSOA CCN3 CDNUMC CLDHGH CLDICE CLDLIQ CLDLOW CLDMED CLDTOT CLOUD CO2 CO2_FFF CO2_LND CO2_OCN DCQ dst_a1_SRF dst_a3_SRF DTCOND DTV EXTINCT FICE FLDS FLNS FLNSC FLNT FLNTC FLUT FLUTC FREQI FREQL FREQR FREQS FSDS FSDSC FSNS FSNSC FSNT FSNTC FSNTOA FSNTOAC ICEFRAC ICIMR ICLDIWP ICLDTWP ICWMR IWC LANDFRAC LHFLX LWCF NUMICE NUMLIQ OCNFRAC OMEGA OMEGAT PBLH PHIS pom_a1_SRF PRECC PRECL PRECSC PRECSL PS PSL Q QFLX QRL QRS RELHUM SFCO2 SFCO2_FFF SFCO2_LND SFCO2_OCN SHFLX SNOWHICE SNOWHLND so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF soa_a2_SRF SOLIN SRFRAD SWCF T TAUX TAUY TGCLDIWP TGCLDLWP TMCO2 TMCO2_FFF TMCO2_LND TMCO2_OCN TMQ TOT_CLD_VISTAU TREFHT TREFHTMN TREFHTMX TROP_P TROP_T TS TSMN TSMX U10 U UQ UU V VD01 VQ VT VU VV WGUSTD WSPDSRFMX WSUB Z3)
#set arrname = (PRECSL TAUY soa_a2_SRF TAUX CDNUMC TGCLDLWP PRECL soa_a1_SRF SHFLX dst_a3_SRF pom_a1_SRF bc_a1_SRF TGCLDIWP dst_a1_SRF LHFLX QFLX LWCF so4_a2_SRF WGUSTD so4_a3_SRF CLDMED so4_a1_SRF PBLH CLDHGH BURDENSEASALT SFCO2 FLNS BURDENDUST U10 SWCF CLDTOT WSPDSRFMX CLDLOW BURDENPOM BURDENBC AODDUST3 BURDENSO4 BURDENSOA AODVIS TMQ FSNS AODABS AODDUST1 FLNSC FSDS SRFRAD FSNT FSNTOA FLDS FLNT FLUT PRECC FSNSC TROP_P FLNTC FLUTC FSNTC FSNTOAC SFCO2_OCN TREFHTMN TSMN TSMX TREFHTMX TMCO2 TS TMCO2_FFF TREFHT PS TMCO2_LND TROP_T SFCO2_FFF PSL PHIS SOLIN SFCO2_LND OCNFRAC SNOWHLND SNOWHICE PRECSC ICEFRAC LANDFRAC)
foreach x ($arrALL)
#  qsub zfp/$x.sh
#  qsub sz/$x.sh
#  qsub br/$x.sh
  qsub orig/$x.sh
end

sleep 15
ls