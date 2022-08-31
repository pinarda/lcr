#TEST SET IGNORE ME

import time
# Make sure you are using the cmip6-2019.10 kernel

# Add ldcpy root to system path (MODIFY FOR YOUR LDCPY CODE LOCATION)
import sys

sys.path.insert(0, '/glade/u/home/apinard/newldcpy/ldcpy')
import ldcpy
# silence warnings
import warnings

warnings.filterwarnings("ignore")
import os
hdf_pp = os.environ["HDF5_PLUGIN_PATH"]
env_list = ['export HDF5_PLUGIN_PATH='+hdf_pp]

monthly_variables = ["TS"]
#daily_variables = ["FLUT", "LHFLX", "PRECT", "TAUX", "TS", "Z500"]

cols_monthly = {}
cols_daily = {}
sets = {}
levels = {}
data_path = "/glade/p/cisl/asap/CAM_lossy_test_data_31/research/"


# for variable in daily_variables:
#     print(variable)
#     levels[variable] = [f"bg_2_{variable}",
#                         f"bg_3_{variable}",
#                         f"bg_4_{variable}", f"bg_5_{variable}",
#                         f"bg_6_{variable}", f"bg_7_{variable}",]
#     sets[variable] = [f"{data_path}/orig_nocomp/b.e11.BRCP85C5CNBDRD.f09_g16.031.cam.h1.{variable}.20060101-20071231.nc",
#                       f"{data_path}/bg/bg_2/b.e11.BRCP85C5CNBDRD.f09_g16.031.cam.h1.{variable}.20060101-20071231.nc",
#                       f"{data_path}/bg/bg_3/b.e11.BRCP85C5CNBDRD.f09_g16.031.cam.h1.{variable}.20060101-20071231.nc",
#                       f"{data_path}/bg/bg_4/b.e11.BRCP85C5CNBDRD.f09_g16.031.cam.h1.{variable}.20060101-20071231.nc",
#                       f"{data_path}/bg/bg_5/b.e11.BRCP85C5CNBDRD.f09_g16.031.cam.h1.{variable}.20060101-20071231.nc",
#                       f"{data_path}/bg/bg_6/b.e11.BRCP85C5CNBDRD.f09_g16.031.cam.h1.{variable}.20060101-20071231.nc",
#                       f"{data_path}/bg/bg_7/b.e11.BRCP85C5CNBDRD.f09_g16.031.cam.h1.{variable}.20060101-20071231.nc"]
    # cols_daily[variable] = ldcpy.open_datasets("cam-fv", [f"{variable}"], sets[variable], [f"orig_{variable}"] + levels[variable], chunks={"time":700})

for variable in monthly_variables:
    print(variable)
    levels[variable] = [f"zfp_p_10_{variable}"]
    sets[variable] = [f"/glade/p/cisl/asap/abaker/pepsi/ens_31/orig/daily/b.e11.BRCP85C5CNBDRD.f09_g16.031.cam.h1.{variable}.20060101-20801231.nc",
                      f"/glade/scratch/apinard/zfp/zfp_p_10/b.e11.BRCP85C5CNBDRD.f09_g16.031.cam.h1.{variable}.20060101-20801231.nc"]
    cols_monthly[variable] = ldcpy.open_datasets("cam-fv", [f"{variable}"], sets[variable], [f"orig_{variable}"] + levels[variable], chunks={"time":700})


agg_dims = ['lat', 'lon']

varname="TS"
set1="orig_TS"
set2="zfp_p_10_TS"
t=0
data_type="cam-fv"
calcs =["ssim_fp"]


ds = cols_monthly[varname].to_array().squeeze()
print(ds)
print(varname)
print(calcs)
print(set1)
print(set2)
print(t)
print(data_type)

diff_metrics = ldcpy.Diffcalcs(
    ds.sel(collection=set1).isel(time=t),
    ds.sel(collection=set2).isel(time=t),
    data_type,
    agg_dims,
    spre_tol=0.001
)

calc_dict = {}
for calc in calcs:
    start = time.time()
    #print(calc)
    #temp = diff_metrics.get_diff_calc(calc).compute()
    #calc_dict[calc] = temp.item(0)
    calc_dict[calc] = diff_metrics.get_diff_calc(calc)
    t = time.time() - start
    print(calc)
    print(t)