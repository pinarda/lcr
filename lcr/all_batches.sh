#!/bin/tcsh

set arrname = (bc_a1_SRF)
foreach x ($arrname)
qsub cheyenne_batch_$x.sh
end