# this batch file needs to create a new json file for model configuration
# then create a new batch file for each model configuration
# then submit the batch file to the queue

# first, take CNN11.json, and use a regex to replace the following fields:
# VarList, CompDirs, and Times (all of which are lists)


import json
import re
import os

# read in the json file
with open('CNN11_template.json', 'r') as f:
    config = json.load(f)

# get the list of variables
VarList = config['VarList']
# get the list of component directories
CompDirs = config['CompDirs']
# get the list of times
Times = config['Times']

newnames = ["dz_bigtest_1var",
            "dz_bigtest_2var",
            "dz_bigtest_5var",
            "dz_bigtest_10var",
            "dz_bigtest_15var",
            "dz_bigtest_2var_alt",
            "dz_bigtest_5var_alt",
            "dz_bigtest_10var_alt",
            "dz_bigtest_15var_alt",
            "dz_bigtest_allvar"]
newvars = [["Z500"],
           ["Z500", "dst_a1_SRF"],
           ["Z500", "dst_a1_SRF", "dst_a3_SRF", "FLNS", "FLNSC"],
           ["Z500", "dst_a1_SRF", "dst_a3_SRF", "FLNS", "FLNSC", "FLUT", "FSNS", "FSNSC", "FSNTOA", "ICEFRAC"],
           ["Z500", "dst_a1_SRF", "dst_a3_SRF", "FLNS", "FLNSC", "FLUT", "FSNS", "FSNSC", "FSNTOA", "ICEFRAC", "LHFLX", "pom_a1_SRF", "PRECL", "PRECSC", "PRECSL", "PRECT", "PRECTMX", "PSL", "Q200", "Q500"],
           ["Z500", "Q850"],
           ["Z500", "Q850", "QBOT", "SHFLX", "so4_a1_SRF", "so4_a2_SRF"],
           ["Z500", "Q850", "QBOT", "SHFLX", "so4_a1_SRF", "so4_a2_SRF", "so4_a3_SRF", "soa_a1_SRF", "soa_a2_SRF", "T010", "T200", "T500"],
           ["Z500", "Q850", "QBOT", "SHFLX", "so4_a1_SRF", "so4_a2_SRF", "so4_a3_SRF", "soa_a1_SRF", "soa_a2_SRF", "T010", "T200", "T500", "T850", "TAUX", "TAUY", "TMQ", "TREFHT", "TREFHTMN", "TREFHTMX", "TS", "U010"],
           ["Z500", "dst_a1_SRF", "dst_a3_SRF", "FLNS", "FLNSC", "FLUT", "FSNS", "FSNSC", "FSNTOA", "ICEFRAC", "LHFLX", "pom_a1_SRF", "PRECL", "PRECSC", "PRECSL", "PRECT", "PRECTMX", "PSL", "Q200", "Q500", "Q850", "QBOT", "SHFLX", "so4_a1_SRF", "so4_a2_SRF", "so4_a3_SRF", "soa_a1_SRF", "soa_a2_SRF", "T010", "T200", "T500", "T850", "TAUX", "TAUY", "TMQ", "TREFHT", "TREFHTMN", "TREFHTMX", "TS", "U010", "U200", "U500", "U850", "UBOT", "V200", "V500", "V850", "VBOT", "WSPDSRFAV", "Z050", "Z500"]
           ]
newcomps = [["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
            ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
            ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
            ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
            ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
            ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
            ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
            ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
            ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
            ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"]]
newtimes = [[2, 6, 11, 16],
            [6],
            [6],
            [6],
            [6],
            [6],
            [6],
            [6],
            [6],
            [2, 3, 4]]
newtestset = ["10pct",
              "1var",
              "1var",
              "1var",
              "1var",
              "1var",
              "1var",
              "1var",
              "1var",
              "1var"]

# we need to write a new json file for each model configuration
# so we need to loop over variables, component directories, and times simultaneously
for i in range(len(newvars)):
    newconfig = config.copy()
    newconfig['VarList'] = newvars[i]
    newconfig['CompDirs'] = newcomps[i]
    newconfig['Times'] = newtimes[i]
    # write the new json file
    with open('CNN11_' + newnames[i] + '.json', 'w') as f:
        json.dump(newconfig, f)

# now we need to create a new batch file for each model configuration
# all we have to do here is replace the string "TEMPLATE" in CNN11_template.sh
# with the corresponding element in newnames
for i in range(len(newnames)):
    with open('CNN11_template.sh', 'r') as f:
        batch = f.read()
    batch = re.sub('TEMPLATE', newnames[i], batch)
    # replace the test set with the new test set enclosed in quotes
    batch = re.sub('TESTSET', '"' + newtestset[i] + '"', batch)


    with open('CNN11_' + newnames[i] + '.sh', 'w') as f:
        f.write(batch)

# now we need to submit the batch file to the queue using qsub
for i in range(len(newnames)):
    # run a command to submit the batch file to the queue using the -v flag and setting the environment variable PLP
    # to the name of the json file
    os.system('qsub -v PLP=' + '"' + 'CNN11_' + newnames[i] + '.json' + '"' + ' CNN11_' + newnames[i] + '.sh')



    # os.system(f'qsub -v PLP="CNN11_{newnames[i]}.json" CNN11_{newnames[i]}.sh')
