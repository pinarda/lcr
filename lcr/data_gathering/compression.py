#! /usr/bin/env python
import os

# use  nco/5.0.5 or greater!
indir = "/glade/p/cisl/asap/CAM_lossy_test_data_31/research/daily_orig"
outdir = "/glade/scratch/apinard/daily_br"

print(indir)
print(outdir)

for filename in os.listdir(indir):
    # get variable name
    loc1 = filename.find('.h')
    loc2 = filename.find('.', loc1 + 1)
    loc3 = filename.find('.', loc2 + 1)
    vname = filename[loc2 + 1:loc3]
    print(vname)
    if filename.endswith(".nc"):
        for i in range(1, 23):
            bgdir = "bg_" + str(i)
            infile = indir + "/" + filename
            outfile = outdir + "/" + bgdir + "/" + filename
            bg_command = "ncks --ppc " + vname + "=" + str(i) + " --baa 8 " + infile + " " + outfile
            print(bg_command)

            d_cmd = "mkdir /glade/scratch/apinard/daily_br/" + bgdir

            os.system(d_cmd)
            os.system(bg_command)