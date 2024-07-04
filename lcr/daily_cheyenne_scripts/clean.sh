#!/bin/bash

output=$(qstat -u apinard | tail -n +4 | cut -d '.' -f 1)
arr=($output)
for i in "${arr[@]}"; do qdel $i; done

#rm -f ~/lcr/lcr/monthly_cheyenne_scripts/testb*
#rm -f /glade/scratch/apinard/*.csv