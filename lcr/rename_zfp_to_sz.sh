#!/bin/tcsh

set arrname = (bc_a1_SRF dst_a1_SRF dst_a3_SRF FLNS FLNSC FLUT FSNS FSNSC FSNTOA ICEFRAC LHFLX pom_a1_SRF PRECL PRECSC PRECSL PRECT PRECTMX PSL Q200 Q500 Q850 QBOT SHFLX so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF soa_a2_SRF T010 T200 T500 T850 TAUX TAUY TMQ TREFHT TREFHTMN TREFHTMX TS U010 U200 U500 U850 VBOT WSPDSRFAV Z050 Z500)
foreach x ($arrname)
  cat batch_scripts/$x.json | sed 's/\["zfp_p_8", "zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"\]/["sz3_ROn0.0001", "sz3_ROn0.001", "sz3_ROn0.01", "sz3_ROn0.1", "sz3_ROn1e-06", "sz3_ROn5e-06", "sz3_ROn0.0005", "sz3_ROn0.005", "sz3_ROn0.05", "sz3_ROn1e-05", "sz3_ROn5e-05"]/' | sed 's/daily_zfp_hdf5/daily_sz3/' > batch_scripts/sz_dssim_scripts/$x.json
  cat batch_scripts/cheyenne_batch_$x.sh | sed "s/-j '.\/batch_scripts\/.json'/-j '.\/batch_scripts\/sz_dssim_scripts\/$x.json'/" | sed "s/'\/glade\/scratch\/apinard\/$x\_calcs.csv'/'\/glade\/scratch\/apinard\/sz3\/$x\_calcs.csv'/" > batch_scripts/sz_dssim_scripts/cheyenne_batch_$x.sh
end