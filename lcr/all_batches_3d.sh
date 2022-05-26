#!/bin/tcsh

set arrname = (FLUT LHFLX PRECT TAUX TS Z500)
set arrname = (CCN3 CLOUD FLNS FLNT FSNS FSNT LHFLX PRECC PRECL PS QFLX SHFLX TMQ TS U)
foreach x ($arrname)
qsub batch_scripts/3d_dssim_scripts/cheyenne_batch_$x.sh
end