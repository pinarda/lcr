import csv
import ldcpy
import os

def batch_calcs(
    full_ds,
    varname,
    location,
    calcs,
    set1,
    set2=None,
    time=0,
    lev=0
):
    """
    This function takes an input list of metrics to compute on a
    dataset and returns them in the form of a .csv

    Parameters
    ==========
    ds : xarray.Dataset
        An xarray dataset containing multiple netCDF files concatenated across a 'collection' dimension
    varname : str
        The variable of interest in the dataset
    location : str
        The location on disk to store the dataset
    calcs :
        The array of calculations to perform on the dataset
    set1 : str
        The collection label of the "control" data
    set2 : str
        The collection label of the (1st) data to compare
    time : int, optional
        The time index used t (default = 0)
    lev : int, optional
        The level index of interest in a 3D dataset (default 0)
    Returns
    =======
    out : Number of failing metrics
    """

    # can optionally grab a subset of the data
    ds = ldcpy.subset_data(full_ds)

    # count the number of failuress
    num_fail = 0

    print(
        'Computing calculations for {} data (set1), time {}'.format(
            set1, time
        ),
        ':',
    )

    if set2 != None:
        diff_metrics = ldcpy.Diffcalcs(
            ds[varname].sel(collection=set1).isel(time=time),
            ds[varname].sel(collection=set2).isel(time=time),
            ['lat', 'lon'],
        )

    metrics = ldcpy.Datasetcalcs(ds[varname].sel(collection=set1).isel(time=time), "cam-fv", ['lat', 'lon'])

    calc_dict = {}
    for calc in calcs:
        temp = metrics.get_calc(calc).compute()
        calc_dict[calc] = temp.item(0)


    file_exists = os.path.isfile(location)
    with open(location, 'a', newline='') as csvfile:
        fieldnames = [
            'set',
            'time'
        ] + calcs
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        row = {
                'set': set1,
                'time': time
            }
        row.update(calc_dict)
        writer.writerow(row)

    return num_fail