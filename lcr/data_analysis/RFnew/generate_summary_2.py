import json
import os
import re
import itertools

# Define combinations of variables
newvars = [
    ["bc_a1_SRF"], ["dst_a1_SRF"], ["dst_a3_SRF"], ["FLNS"], ["FLNSC"], ["FLUT"], ["FSNS"],
    ["FSNSC"], ["FSNTOA"], ["ICEFRAC"], ["LHFLX"], ["pom_a1_SRF"], ["PRECL"], ["PRECSC"],
    ["PRECSL"], ["PRECT"], ["PSL"], ["Q200"], ["Q500"], ["Q850"], ["QBOT"], ["SHFLX"],
    ["so4_a1_SRF"], ["so4_a2_SRF"], ["so4_a3_SRF"], ["soa_a1_SRF"], ["soa_a2_SRF"],
    ["T010"], ["T200"], ["T500"], ["T850"], ["TAUX"], ["TAUY"], ["TMQ"], ["TREFHT"],
    ["TREFHTMN"], ["TREFHTMX"], ["TS"], ["U010"], ["U200"], ["U500"], ["U850"], ["UBOT"],
    ["V200"], ["V500"], ["V850"], ["VBOT"], ["WSPDSRFAV"], ["Z050"], ["Z500"],
    ["TS", "PRECT"]  # Example of a combination of two variables
]
# newvars = [
#     ["dst_a1_SRF"], ["dst_a3_SRF"], ["pom_a1_SRF"], ["so4_a1_SRF"],
#     ["so4_a2_SRF"], ["so4_a3_SRF"], ["soa_a1_SRF"], ["soa_a2_SRF"],
# ]

# newvars = [
#     ["TS"]
# ]

# Define the times, models, metrics, transforms, etc.
newcomps = [["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"]]
newtimes = [1000, 2500]
# newtimes = [10]
newtestset = ["10_90_wholeslice"]
metrics = ["dssim"]
metric = [["dssim"]]
transforms = ["none"]
cutdatasets = [False]
models = ["cnn"]

# Read in the template JSON file
with open('RF_template.json', 'r') as f:
    config = json.load(f)

# Generate all combinations of the parameters
combinations = list(
    itertools.product(newvars, newtimes, models, newcomps, newtestset, metrics, metric, transforms, cutdatasets))

# Generate new configurations and batch files
for idx, (var, time, model, comp, testset, metric_single, metric_list, transform, cutdataset) in enumerate(
        combinations):
    newconfig = config.copy()
    newconfig['VarList'] = var
    newconfig['CompDirs'] = comp
    newconfig['Times'] = [time]
    newconfig['CutDataset'] = cutdataset
    newconfig['Metric'] = metric_list

    # Generate a name for the job based on the variable(s), model, and time
    var_name = "_".join(var)
    newname = f"{var_name}_{model.upper()}_{time}_{testset}"
    json_filename = f'run_casper_{newname}.json'

    # Write the new JSON configuration file
    with open(json_filename, 'w') as f:
        json.dump(newconfig, f)

    # Read the batch file template
    with open('run_casper.sh', 'r') as f:
        batch = f.read()

    # Replace placeholders in the batch file
    batch = re.sub('TEMPLATE', newname, batch)
    batch = re.sub('TESTSET', f'"{testset}"', batch)
    batch = re.sub('JOBID', str(100 + idx), batch)
    batch = re.sub('METRIC', metric_single, batch)
    batch = re.sub('TRANSFORM', transform, batch)
    batch = re.sub('MODEL', model, batch)

    batch_filename = f'run_casper_{newname}.sh'

    # Write the new batch file
    with open(batch_filename, 'w') as f:
        f.write(batch)

    # Submit the batch file to the queue using qsub
    os.system(f'qsub -q regular -v PLP="{json_filename}" {batch_filename}')