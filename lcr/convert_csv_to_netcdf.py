"""
Convert a .csv file with "lat", "lon", and value columns (mean.methane)
to a netCDF file
"""
import csv
import re
import numpy as np
import xarray as xr

def csv_to_netcdf(f, value_col, lats, lons, days, filename):

    data_array = xr.DataArray(None, coords=[lats, lons, days], dims=["lat", "lon", "time"])
    with open(f, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lon = float(row["lon"])
            lat = float(row["lat"])
            time = int(row["day"])
            value = row[value_col]
            if value == "NA":
                data_array.loc[dict(lon=lon, lat=lat, time=time)] = np.nan
            else:
                data_array.loc[dict(lon=lon, lat=lat, time=time)] = float(value)

    # This is needed to make cf_xarray work
    data_array.lon.assign_attrs(long_name="longitude", units="degrees_north")
    data_array.lat.assign_attrs(long_name="latitude", units="degrees_east")
    data_array.time.assign_attrs(long_name="time", units="time_bnds")

    ds = data_array.rename("CH4").to_dataset(promote_attrs=True)
    ds["CH4"].lon.attrs["units"] = "degrees_east"
    ds["CH4"].lat.attrs["units"] = "degrees_north"
    ds["CH4"].time.attrs["units"] = "time_bnds"
    return ds.to_netcdf(path=f"../data/methane/{filename}.nc")

if __name__ == "__main__":
    csvfilename = "/Users/alex/Downloads/Daily/2021/Permian/Permian_res_0.5_2021_12.csv"
    values = "mean.methane"
    filename = "Permian_res_0.5_2021_12"

    # have to pass in the ranges right now, could be simplified by reading these from the csv though.
    lon_range = {min:-106.75, max:-100.25}
    lat_range = {min:30.25, max:34.75}
    day_range = {min:1, max:30}
    step = 0.5

    lon_length = lon_range[max] - lon_range[min]
    lat_length = lat_range[max] - lat_range[min]

    lon_seq = np.linspace(0, lon_length, int(lon_length/step) + 1) + lon_range[min]
    lat_seq = np.linspace(0, lat_length, int(lat_length/step) + 1) + lat_range[min]
    day_seq = range(1, 31)

    netcdf_data = csv_to_netcdf(csvfilename, values, lat_seq, lon_seq, day_seq, filename)
