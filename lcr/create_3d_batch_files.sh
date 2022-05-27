#!/bin/tcsh

set arrname = (CCN3 CLOUD FLNS FLNT FSNS FSNT LHFLX PRECC PRECL PS QFLX SHFLX TMQ TS U)
foreach x ($arrname)
  cat batch_scripts/$x.json | sed 's/\["zfp_p_8", "zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"\]/\["zfp_p_8", "zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"\]/' | sed 's/daily_zfp_hdf5/zfp_hdf5/' | sed 's/research\/daily_orig/research\/orig_nocomp/' > batch_scripts/3d_dssim_scripts/$x.json
  cat batch_scripts/cheyenne_batch_$x.sh | sed "s/-j '.\/batch_scripts\/$x.json'/-j '.\/batch_scripts\/3d_dssim_scripts\/$x.json' -ts 0 -tt 30/" | sed "s/'\/glade\/scratch\/apinard\/$x\_calcs.csv'/'\/glade\/scratch\/apinard\/3D\/$x\_calcs.csv'/" > batch_scripts/3d_dssim_scripts/cheyenne_batch_$x.sh
end