#!/bin/tcsh

set arrname = (bc_a1_SRF)
foreach x ($arrname)
qsub batch_scripts/dssim_scripts/cheyenne_batch_$x.sh
end