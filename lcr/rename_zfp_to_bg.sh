#!/bin/tcsh

set arrname = (bc_a1_SRF dst_a1_SRF dst_a3_SRF FLNS FLNSC FLUT FSNS FSNSC FSNTOA ICEFRAC LHFLX pom_a1_SRF PRECL PRECSC PRECSL PRECT PRECTMX PSL Q200 Q500 Q850 QBOT SHFLX so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF soa_a2_SRF T010 T200 T500 T850 TAUX TAUY TMQ TREFHT TREFHTMN TREFHTMX TS U010 U200 U500 U850 VBOT WSPDSRFAV Z050 Z500)
foreach x ($arrname)
  cat batch_scripts/$x.json | sed 's/\["zfp_p_8", "zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"\]/["bg_2", "bg_3", "bg_4", "bg_5", "bg_6", "bg_7"]/' > batch_scripts/bg_dssim_scripts/$x.json
  cat batch_scripts/cheyenne_batch_$x.sh | sed "s/-j '$x.json'/-j '.\/batch_scripts\/bg_dssim_scripts\/$x.json'/" > batch_scripts/bg_dssim_scripts/cheyenne_batch_$x.sh
end