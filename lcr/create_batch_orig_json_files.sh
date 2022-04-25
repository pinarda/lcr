#!/bin/tcsh

set arrname = (bc_a1_SRF dst_a1_SRF dst_a3_SRF FLNS FLNSC FLUT FSNS FSNSC FSNTOA ICEFRAC LHFLX pom_a1_SRF PRECL PRECSC PRECSL PRECT PRECTMX PSL Q200 Q500 Q850 QBOT SHFLX so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF soa_a2_SRF T010 T200 T500 T850 TAUX TAUY TMQ TREFHT TREFHTMN TREFHTMX TS U010 U200 U500 U850 VBOT WSPDSRFAV Z050 Z500)
rm batch_scripts/auto_generated_scripts/*
foreach x ($arrname)
  cat batch_scripts/cheyenne_batch_orig.sh | sed "s/orig_calcs.csv' -j 'orig_calcs.json'/orig_calcs_$x.csv' -j 'orig_calcs_$x.json'/" > batch_scripts/auto_generated_scripts/cheyenne_batch_orig_$x.sh
  cat batch_scripts/orig_calcs.json | sed 's/pom_a1_SRF/'$x'/' > batch_scripts/auto_generated_scripts/orig_calcs_$x.json
end