#!/bin/bash

rm -f testb*
rm -f /glade/scratch/apinard/*.csv

output=$(qstat -u apinard | tail -n +4 | cut -d '.' -f 1)
arr=($output)
for i in "${arr[@]}"; do qdel $i.chadmin1.ib0.cheyenne.ucar.edu; done