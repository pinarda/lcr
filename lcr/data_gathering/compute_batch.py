import os,sys
import argparse
import json
import csv
import time

def main(argv):
 # Get command line stuff and store in a dictionary
    args = parseArguments()

    outfile = args.outfile
    origoutfile = args.origoutfile
    jsonfile = args.json
    verbose = args.verbose
    ts = args.tstart
    tt = args.ttotal
    ldc = args.ldcpydev

    data_type = "cam-fv"

    if verbose:
        print("Starting compute_batch.py")
        print("   output file = ", outfile)
        print("   orig output file = ", origoutfile)
        print("   json file = ", jsonfile)
        print("   verbose = ", verbose)
        print("   tstart = ", ts)
        print("   ttotal = ", tt)
        print("   ldcpydev = ", ldc)

    #read from jsonfile
    if jsonfile:
        var_list, label_list, calcs, orig_calcs, filename_pre, filename_post, orig_path, comp_path, comp_dirs, ldcpy_dev_path = read_jsonlist(jsonfile)
    else:
        var_list = ['TS', 'LHFLX']
        label_list = ["bg_2", "bg_3", "bg_4", "bg_5", "bg_6", "bg_7"]
        comp_dirs = ["bg_2", "bg_3", "bg_4", "bg_5", "bg_6", "bg_7"]
        filename_pre = "b.e11.BRCP85C5CNBDRD.f09_g16.031.cam.h1."
        filename_post = ".20060101-20071231.nc"
        calcs = ["ssim_fp"]
        orig_calcs = ["mean"]
        comp_path = "/glade/p/cisl/asap/CAM_lossy_test_data_31/research/bg/"
        orig_path = "/glade/p/cisl/asap/CAM_lossy_test_data_31/orig/"
        ldcpy_dev_path = ""


    print("   ldcpydevpath = ", ldcpy_dev_path)

    #TODO: add checks : e.g.,  that varlist and label list are the same length

    if ldc:
        sys.path.insert(0, ldcpy_dev_path)
    import ldcpy


    #populate dictionaries
    cols = {}
    labels = {}    
    files = {}
    for var in var_list:
        if verbose:
            print(var)

        #append var name to labels
        labels[var] = ['orig_' + var]
        for k in label_list:
            labels[var].append(k + "_" + var)

        #construct full pathnames for the files
        var_filename = filename_pre + var + filename_post    
        #the control one is first
        files[var] = [orig_path + var_filename]
        for k in comp_dirs:
            t = comp_path + k + "/" + var_filename
            files[var].append(t)
            
        if verbose:    
            print("files = ", files[var])
            print("labels = ", labels[var])

        #now create collection for this variable    
        #add dask later
        list_var = [var]
        print("data_type = ", data_type)
        print("list_var = ", list_var)
        print("files = ", files[var])
        print("labels = ", labels[var])
        cols[var] = ldcpy.open_datasets(data_type, list_var, files[var], labels[var], weights=False, chunks={"time":50})

    #create/open csv file and add computations 

    if verbose:
        print("Output file = ", outfile)
        print("Orig output file = ", origoutfile)

    #loop through calculations
    for var in var_list:
        if verbose:
            print("var =", var)
            
        #get time range to iterate on
        ntime = cols[var].dims["time"]
        if ts > ntime:
            print("Specified start time slice is greater than total number of slices for variable", var, ". Skipping")
            continue
        if tt < 0:
            tend = ntime
        else:
            tend = ts + tt
            if tend > ntime:
                tend = ntime
        if ts < 0:
            ts = 0

        if verbose:
            print("   ts = ", ts)
            print("   tend = ", tend)
            
        orig = labels[var][0]
        comp_list = labels[var][1:]
        for c in comp_list:

            #write with each time group            
            file_exists = os.path.isfile(outfile)
            with open(outfile, 'a', newline='') as csvfile:
                fieldnames = [
                    'set',
                    'time'
                ] + calcs
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()

                for t in range(ts, tend):
                    #print(var, orig, c, t)
                    start = time.time()
                    calc_dict = simple_diff_calcs(cols[var], var, calcs,  orig, c , t, data_type)
                    row = {
                        'set': c,
                        'time': t
                    }
                    row.update(calc_dict)
                    writer.writerow(row)
                    t = time.time() - start
                    print(t)



        orig_file_exists = os.path.isfile(origoutfile)
        with open(origoutfile, 'a+', newline='') as origcsvfile:
            fieldnames = [
                             'set',
                             'time'
                         ] + orig_calcs
            writer = csv.DictWriter(origcsvfile, fieldnames=fieldnames)
            if not orig_file_exists:
                writer.writeheader()

            for t in range(ts, tend):
                row = {
                    'set': orig,
                    'time': t
                }
                orig_calc_dict = simple_orig_calcs(cols[var], var, orig_calcs, orig, t, data_type)
                row.update(orig_calc_dict)
                writer.writerow(row)

        del cols[var]

def read_jsonlist(metajson):
    var_list = []
    label_list = []
    calcs = []
    orig_calcs = []
    filename_pre = ""
    filename_post = ""
    orig_path = ""
    comp_path = ""
    comp_dirs = ""
    ldcpy_dev_path = ""

    print("Reading jsonfile", metajson, " ...")
    if not os.path.exists(metajson):
        print("\n")
        print("*************************************************************************************")
        print("Warning: Specified json file does not exist: ", metajson)
        print("*************************************************************************************")
        print("\n")
    else:
        fd = open(metajson)
        metainfo = json.load(fd)
        if 'VarList' in metainfo:
            var_list = metainfo['VarList']
        if 'VarLabel' in metainfo:
            label_list = metainfo['VarLabel']
        if "DiffCalcs" in metainfo:
            calcs = metainfo["DiffCalcs"]
        if "OrigCalcs" in metainfo:
            orig_calcs = metainfo["OrigCalcs"]
        if "FilenamePre" in metainfo:
            filename_pre = metainfo["FilenamePre"]
        if "FilenamePost" in metainfo:
            filename_post = metainfo["FilenamePost"]
        if "OrigPath" in metainfo:
            orig_path = metainfo["OrigPath"]
        if "CompPath" in metainfo:
            comp_path = metainfo["CompPath"]
        if "CompDirs" in metainfo:
            comp_dirs = metainfo["CompDirs"]
        if "OptLdcpyDevPath" in metainfo:
            ldcpy_dev_path = metainfo["OptLdcpyDevPath"]

    return var_list, label_list, calcs, orig_calcs, filename_pre, filename_post, orig_path, comp_path, comp_dirs, ldcpy_dev_path



def simple_diff_calcs(
    ds,
    varname,
    calcs,
    set1,
    set2,
    t,
    data_type
):
    """                                                                                                                                
    This function takes an input list of metrics to compute on a                                                                               
    dataset and returns them
                     
    ds : xarray.Dataset       
        An xarray dataset containing multiple netCDF files concatenated across a 'collection' dimension                                
    varname : str                                                                                                       
        The variable of interest in the dataset                                                                
    calcs :        
        The array of calculations to perform on the dataset
    set1 : str                                     
        The collection label of the "control" data                                 
    set2 : str                                                   
        The collection label of the (1st) data to compare   
    """

    ds = ds.to_array().squeeze()
    import ldcpy

    agg_dims = ['lat', 'lon']
    
    diff_metrics = ldcpy.Diffcalcs(
        ds.sel(collection=set1).isel(time=t),
        ds.sel(collection=set2).isel(time=t),
        data_type,
        agg_dims,
        spre_tol=0.001
    )

    calc_dict = {}
    for calc in calcs:
        #print(calc)
        #temp = diff_metrics.get_diff_calc(calc).compute()
        #calc_dict[calc] = temp.item(0)
        calc_dict[calc] = diff_metrics.get_diff_calc(calc)

    return calc_dict


def simple_orig_calcs(
        ds,
        varname,
        calcs,
        set1,
        time,
        data_type
):
    """
    This function takes an input list of metrics to compute on a
    dataset and returns them

    ds : xarray.Dataset
        An xarray dataset containing multiple netCDF files concatenated across a 'collection' dimension
    varname : str
        The variable of interest in the dataset
    calcs :
        The array of calculations to perform on the dataset
    set1 : str
        The collection label of the "control" data
    set2 : str
        The collection label of the (1st) data to compare
    """

    import ldcpy

    agg_dims = ['lat', 'lon']

    clcs = ldcpy.Datasetcalcs(
        ds[varname].sel(collection=set1).isel(time=time),
        data_type,
        [],
        weighted=False
    )

    # HERE, compute 2D FFT first
    #my_data_fft2 = diff_metrics.get_calc("fft2")
    #fft2_calcs = ldcpy.Datasetcalcs(my_data_fft2, "cam-fv", agg_dims, weighted=False)


    calc_dict = {}
    for calc in calcs:
        # print(calc)
        if calc in ["entropy", "range", "lat_autocorr", "lon_autocorr", "percent_unique", "most_repeated", "most_repeated_percent"]:
            calc_dict[calc] = clcs.get_single_calc(calc)
        else:
            calc_dict[calc] = clcs.get_calc(calc)
        temp = calc_dict[calc].compute()
        #temp = float(fft2_calcs.get_calc(calc).compute())
        #temp = float(clcs.get_calc(calc).compute())
        calc_dict[calc] = temp.item(0)
        #calc_dict[calc] = temp

    return calc_dict


def parseArguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json", help="json input file that describes the calculation to perform.", type=str, default="./sample_zfp_diff_config.json")
    parser.add_argument("-o", "--outfile", help="csv file to store output (if file exists, then data will append).", type=str, default="./sample.csv")
    parser.add_argument("-oo", "--origoutfile", help="csv file to store original data output (if file exists, then data will append).", type=str, default="./origsample.csv")

    parser.add_argument("-ts", "--tstart", help="Starting time slice.", type=int, default=0)
    parser.add_argument("-tt", "--ttotal", help="Number of time slices to process  (-1 means all slices from the start).", type=int, default=-1)

    parser.add_argument("-v", "--verbose", help="Output extra info as the computation runs.", action="store_true")
    parser.add_argument("-ld", "--ldcpydev", help="Use the development version of ldcpy specified in the json file",action="store_true")

    args = parser.parse_args()

    return args

       
if __name__ == "__main__":
    main(sys.argv[1:])
