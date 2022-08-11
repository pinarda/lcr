#!/bin/bash

scp apinard@cheyenne.ucar.edu:/glade/scratch/apinard/*.csv ../../data/monthly/
cd ../../data/monthly/

awk '(NR == 1) || (FNR > 1)'  [bsz]*.csv > monthly_dssims.csv