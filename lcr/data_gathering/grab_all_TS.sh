#! /usr/bin/env python
import os

#indir = "/glade/scratch/abaker/comp/sm_orig"
# Full dataset
#indir = "/glade/p/cisl/asap/abaker/pepsi/ens_31/orig/daily"
# 2006-2010 subset daily
indir = "/glade/p/cisl/asap/CAM_lossy_test_data_31/research/daily_orig/"
# 2006-2010 subset monthly
indir = "/glade/p/cisl/asap/CAM_lossy_test_data_31/research/monthly_orig/"

outdir = "/glade/scratch/apinard/zfp"

print("indir = ",indir)
print("outdir = ", outdir)

#later we may want to use NetDCF4py to resd in # of dims (and lev vs. ilev)
vars_3d = ["CLOUD", "CCN3", "U"]

#make sure environment is properly set
#zfp, precision mode -F ,varname>,32013,2,<precision>

hdf_pp = os.environ["HDF5_PLUGIN_PATH"]
print("hdf5 = ", hdf_pp)
path_pp = os.environ["PATH"]
print("path = ", path_pp)
ld_path_pp = os.environ["LD_LIBRARY_PATH"]
print("ld_lib_path = ", ld_path_pp)

for filename in os.listdir(indir):
    #get variable name
    loc1 = filename.find('.h')
    loc2 = filename.find('.', loc1+1)
    loc3 = filename.find('.', loc2+1)
    vname = filename[loc2+1:loc3]
    print(vname)

    chunk_2d = " -c "+ vname + ":1,192,288 "
    chunk_3d = " -c "+ vname + ":1,30,192,288 "

    if vname in vars_3d:
        chunk = chunk_3d
    else:
        chunk = chunk_2d

# whole time series
#    if filename.endswith("TS.20060101-20801231.nc"):
# daily
#    if filename.endswith(".20060101-20071231.nc"):
#monthly
    if filename.endswith("T.200601-203512.nc"):
        for i in range(10,27,2):
            zfpdir = "zfp_p_" + str(i)
            infile = indir + "/" + filename
            outfile = outdir + "/" + zfpdir + "/" + filename

            zfp_command = "nccopy -F " + vname + ",32013,2," + str(i) + chunk + infile + " " + outfile
            print(zfp_command)

            os.system(zfp_command)

