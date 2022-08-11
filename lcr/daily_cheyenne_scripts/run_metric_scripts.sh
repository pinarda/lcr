#!/bin/tcsh

# ./clean.sh

# known issues so far:
set arrname = (bc_a1_SRF dst_a1_SRF dst_a3_SRF FLNS FLNSC FLUT FSNS FSNSC FSNTOA ICEFRAC LHFLX pom_a1_SRF PRECL PRECSC PRECSL PRECT PRECTMX PSL Q200 Q500 Q850 QBOT SHFLX so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF soa_a2_SRF T010 T200 T500 T850 TAUX TAUY TMQ TREFHT TREFHTMN TREFHTMX TS U010 U200 U500 U850 VBOT WSPDSRFAV Z050 Z500)
foreach x ($arrname)
#  qsub zfp/$x.sh
#  qsub sz/$x.sh
#  qsub br/$x.sh
  qsub orig/$x.sh
end

sleep 15
ls