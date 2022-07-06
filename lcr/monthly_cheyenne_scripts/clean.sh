#!/bin/bash

rm -f testb*
rm -f /glade/scratch/apinard/*.csv

output=$(qstat -u apinard | tail -n +4 | cut -d '.' -f 1)
foreach x ($output)
  qdel $x.chadmin1.ib0.cheyenne.ucar.edu
end