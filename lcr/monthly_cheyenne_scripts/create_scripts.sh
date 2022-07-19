#!/bin/tcsh

set arrname = (ABSORB ANRAIN ANSNOW AODABS AODDUST1 AODDUST2 AODDUST3 AODVIS AQRAIN AQSNOW AREI AREL AWNC AWNI bc_a1_SRF BURDENBC BURDENDUST BURDENPOM BURDENSEASALT BURDENSO4 BURDENSOA CCN3 CDNUMC CLDHGH CLDICE CLDLIQ CLDLOW CLDMED CLDTOT CLOUD CO2 CO2_FFF CO2_LND CO2_OCN DCQ dst_a1_SRF dst_a3_SRF DTCOND DTV EXTINCT FICE FLDS FLNS FLNSC FLNT FLNTC FLUT FLUTC FREQI FREQL FREQR FREQS FSDS FSDSC FSNS FSNSC FSNT FSNTC FSNTOA FSNTOAC ICEFRAC ICIMR ICLDIWP ICLDTWP ICWMR IWC LANDFRAC LHFLX LWCF NUMICE NUMLIQ OCNFRAC OMEGA OMEGAT PBLH PHIS pom_a1_SRF PRECC PRECL PRECSC PRECSL PS PSL Q QFLX QRL QRS RELHUM SFCO2 SFCO2_FFF SFCO2_LND SFCO2_OCN SHFLX SNOWHICE SNOWHLND so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF soa_a2_SRF SOLIN SRFRAD SWCF T TAUX TAUY TGCLDIWP TGCLDLWP TMCO2 TMCO2_FFF TMCO2_LND TMCO2_OCN TMQ TOT_CLD_VISTAU TREFHT TREFHTMN TREFHTMX TROP_P TROP_T TS TSMN TSMX U10 U UQ UU V VD01 VQ VT VU VV WGUSTD WSPDSRFMX WSUB Z3)
foreach x ($arrname)
  cat batch_script_zfp_template.sh | sed "s/MATCHME.csv/monthly$x.csv/" | sed "s/MATCHME.json/$x.json/" > zfp/$x.sh
  cat sample_zfp_diff_config.json | sed "s/MATCHME/$x/" > zfp/$x.json
  cat batch_script_sz_template.sh | sed "s/MATCHME.csv/monthly$x.csv/" | sed "s/MATCHME.json/$x.json/" > sz/$x.sh
  cat sample_sz_diff_config.json | sed "s/MATCHME/$x/" > sz/$x.json
  cat batch_script_br_template.sh | sed "s/MATCHME.csv/monthly$x.csv/" | sed "s/MATCHME.json/$x.json/" > br/$x.sh
  cat sample_br_diff_config.json | sed "s/MATCHME/$x/" > br/$x.json
  cat batch_script_orig_template.sh | sed "s/MATCHME.csv/monthly$x.csv/" | sed "s/MATCHME.json/$x.json/" > orig/$x.sh
  cat sample_orig_calcs_config.json | sed "s/MATCHME/$x/" > orig/$x.json
  cat batch_script_comp_template.sh | sed "s/MATCHME/$x/" > comp/$x.sh
end