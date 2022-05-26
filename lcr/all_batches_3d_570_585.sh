#!/bin/tcsh

set arrname = (CCN3 CLOUD FLNS FLNT FSNS FSNT LHFLX PRECC PRECL PS QFLX SHFLX TMQ TS U)
foreach x ($arrname)
qsub batch_scripts/3d_dssim_scripts_570_585/cheyenne_batch_$x.sh
end