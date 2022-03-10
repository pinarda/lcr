import csv
import ldcpy
import os

def save_metrics(
    full_ds,
    varname,
    set1,
    set2,
    time=0,
    color='coolwarm',
    lev=0,
    ks_tol=0.05,
    pcc_tol=0.99999,
    spre_tol=5.0,
    ssim_tol=0.9995,
    location='names.csv',
):
    """
    Check the K-S, Pearson Correlation, and Spatial Relative Error metrics from:
    A. H. Baker, H. Xu, D. M. Hammerling, S. Li, and J. Clyne,
    “Toward a Multi-method Approach: Lossy Data Compression for
    Climate Simulation Data”, in J.M. Kunkel et al. (Eds.): ISC
    High Performance Workshops 2017, Lecture Notes in Computer
    Science 10524, pp. 30–42, 2017 (doi:10.1007/978-3-319-67630-2_3).
    Check the SSIM metric from:
    A.H. Baker, D.M. Hammerling, and T.L. Turton. “Evaluating image
    quality measures to assess the impact of lossy data compression
    applied to climate simulation data”, Computer Graphics Forum 38(3),
    June 2019, pp. 517-528 (doi:10.1111/cgf.13707).
    Default tolerances for the tests are:
    ------------------------
    K-S: fail if p-value < .05 (significance level)
    Pearson correlation coefficient:  fail if coefficient < .99999
    Spatial relative error: fail if > 5% of grid points fail relative error
    SSIM: fail if SSIM < .99995
    Parameters
    ==========
    ds : xarray.Dataset
        An xarray dataset containing multiple netCDF files concatenated across a 'collection' dimension
    varname : str
        The variable of interest in the dataset
    set1 : str
        The collection label of the "control" data
    set2 : str
        The collection label of the (1st) data to compare
    time : int, optional
        The time index used t (default = 0)
    ks_tol : float, optional
        The p-value threshold (significance level) for the K-S test (default = .05)
    pcc_tol: float, optional
        The default Pearson corrolation coefficient (default  = .99999)
    spre_tol: float, optional
        The percentage threshold for failing grid points in the spatial relative error test (default = 5.0).
    ssim_tol: float, optional
         The threshold for the ssim test (default = .999950
    time : lev, optional
        The level index of interest in a 3D dataset (default 0)
    Returns
    =======
    out : Number of failing metrics
    """

    ds = subset_data(full_ds)

    # count the number of failuress
    num_fail = 0

    print(
        'Evaluating 4 metrics for {} data (set1) and {} data (set2), time {}'.format(
            set1, set2, time
        ),
        ':',
    )

    diff_metrics = Diffcalcs(
        ds[varname].sel(collection=set1).isel(time=time),
        ds[varname].sel(collection=set2).isel(time=time),
        ['lat', 'lon'],
    )

    # reg_metrics = Datasetcalcs(
    #    ds[varname].sel(collection=set1).isel(time=time)
    #    - ds[varname].sel(collection=set2).isel(time=time),
    #    ['lat', 'lon'],
    # )
    # max_abs = reg_metrics.get_calc('max_abs').data.compute()

    # max_rel_error = diff_metrics.get_diff_calc('max_spatial_rel_error')

    # Pearson less than pcc_tol means fail
    # pcc = diff_metrics.get_diff_metric('pearson_correlation_coefficient').data.compute()

    # K-S p-value less than ks_tol means fail (can reject null hypo)
    # ks = diff_metrics.get_diff_metric('ks_p_value')

    # Spatial rel error fails if more than spre_tol
    # spre = diff_metrics.get_diff_metric('spatial_rel_error')

    # SSIM less than of ssim_tol is failing
    ssim_val = diff_metrics.get_diff_calc('ssim', color)

    ssim_fp_val = diff_metrics.get_diff_calc('ssim_fp')

    ssim_fp_old_val = diff_metrics.get_diff_calc('ssim_fp_old')

    file_exists = os.path.isfile(location)
    with open(location, 'a', newline='') as csvfile:
        fieldnames = [
            'set',
            'time',
            # 'max_abs',
            # 'max_rel_error',
            #            'pcc',
            #            'ks_p_value',
            #            'spatial_rel_error',
            'ssim',
            'ssim_fp',
            'ssim_fp_old',
            #            'pcc_pass',
            #            'ks_p_value_pass',
            #            'spatial_rel_error_pass',
            # 'ssim_pass',
            # 'ssim_fp_pass',
            # 'ssim_fp_old_pass',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                'set': set2,
                'time': time,
                # 'max_abs': max_abs,
                # 'max_rel_error': max_rel_error,
                #                'pcc': pcc,
                #                'ks_p_value': ks,
                #                'spatial_rel_error': spre,
                'ssim': ssim_val,
                'ssim_fp': ssim_fp_val,
                'ssim_fp_old': ssim_fp_old_val,
                #                'pcc_pass': pcc >= pcc_tol,
                #                'ks_p_value_pass': ks >= ks_tol,
                #                'spatial_rel_error_pass': spre <= spre_tol,
                # 'ssim_pass': ssim_val >= ssim_tol,
                # 'ssim_fp_pass': ssim_fp_val >= ssim_tol,
                # 'ssim_fp_old_pass': ssim_fp_old_val >= ssim_tol,
            }
        )

    return num_fail