#!/bin/tcsh

set arrname = (bc_a1_SRF dst_a1_SRF dst_a3_SRF FLNS FLNSC FLUT FSNS FSNSC FSNTOA ICEFRAC LHFLX pom_a1_SRF PRECL PRECSC PRECSL PRECT PRECTMX PSL Q200 Q500 Q850 QBOT SHFLX so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF soa_a2_SRF T010 T200 T500 T850 TAUX TAUY TMQ TREFHT TREFHTMN TREFHTMX TS U010 U200 U500 U850 VBOT WSPDSRFAV Z050 Z500)
foreach x ($arrname)
  cat batch_script_zfp_template.sh | sed "s/MATCHME.csv/$x.csv/" | sed "s/MATCHME.json/$x.json/" > zfp/$x.sh
  cat sample_zfp_diff_config.json | sed "s/MATCHME/$x/" > zfp/$x.json
  cat batch_script_sz_template.sh | sed "s/MATCHME.csv/$x.csv/" | sed "s/MATCHME.json/$x.json/" > sz/$x.sh
  cat sample_sz_diff_config.json | sed "s/MATCHME/$x/" > sz/$x.json
  cat batch_script_br_template.sh | sed "s/MATCHME.csv/$x.csv/" | sed "s/MATCHME.json/$x.json/" > br/$x.sh
  cat sample_br_diff_config.json | sed "s/MATCHME/$x/" > br/$x.json
  cat batch_script_orig_template.sh | sed "s/MATCHME.csv/$x.csv/" | sed "s/MATCHME.json/$x.json/" > orig/$x.sh
  cat sample_orig_calcs_config.json | sed "s/MATCHME/$x/" > orig/$x.json
  cat batch_script_comp_template.sh | sed "s/MATCHME/$x/" > comp/$x.sh
end