# this batch file needs to create a new json file for model configuration
# then create a new batch file for each model configuration
# then submit the batch file to the queue

# first, take CNN11.json, and use a regex to replace the following fields:
# VarList, CompDirs, and Times (all of which are lists)


import json
import re
import os

# read in the json file
with open('RF_template.json', 'r') as f:
    config = json.load(f)

# get the list of variables
VarList = config['VarList']
# get the list of component directories
CompDirs = config['CompDirs']
# get the list of times
Times = config['Times']

newnames = ["4z_bigtest_1var",
            "4z_bigtest_1var_TS",
            "4z_bigtest_1var_PRECT",
            "4z_bigtest_1var_ICEFRAC",
            "4z_bigtest_1var_FLUT",
            "4z_bigtest_2var",
            "4z_bigtest_5var",
            "4z_bigtest_10var",
            "4z_bigtest_15var",
            "4z_bigtest_2var_alt",
            "4z_bigtest_5var_alt",
            "4z_bigtest_10var_alt",
            "PRECTTS"]
newvars = [["Z500"],
           ["TS"],
           ["PRECL"],
           ["ICEFRAC"],
           ["FLUT"],
           ["Z500", "dst_a1_SRF"],
           ["Z500", "dst_a1_SRF", "dst_a3_SRF", "FLNS", "FLNSC"],
           ["Z500", "dst_a1_SRF", "dst_a3_SRF", "FLNS", "FLNSC", "FLUT", "FSNS", "FSNSC", "FSNTOA", "ICEFRAC"],
           ["Z500", "dst_a1_SRF", "dst_a3_SRF", "FLNS", "FLNSC", "FLUT", "FSNS", "FSNSC", "FSNTOA", "ICEFRAC", "LHFLX", "pom_a1_SRF", "PRECL", "PRECSC", "PRECSL", "PRECT", "PRECTMX", "PSL", "Q200", "Q500"],
           ["Z500", "Q850"],
           ["Z500", "Q850", "QBOT", "SHFLX", "so4_a1_SRF", "so4_a2_SRF"],
           ["Z500", "Q850", "QBOT", "SHFLX", "so4_a1_SRF", "so4_a2_SRF", "so4_a3_SRF", "soa_a1_SRF", "soa_a2_SRF", "T010", "T200", "T500"],
           ["PRECT", "TS"]
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
            ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
            ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
            ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
            ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"]]
newtimes = [[6, 11, 16],
            [6, 11, 16],
            [6, 11, 16],
            [6, 11, 16],
            [6, 11, 16],
            [6],
            [6],
            [6],
            [6],
            [6],
            [6],
            [6],
            [6]]
newtestset = ["60_25_wholeslice",
              "60_25_wholeslice",
              "60_25_wholeslice",
              "60_25_wholeslice",
              "60_25_wholeslice",
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
    with open('RF_' + newnames[i] + '.json', 'w') as f:
        json.dump(newconfig, f)

# now we need to create a new batch file for each model configuration
# all we have to do here is replace the string "TEMPLATE" in CNN11_template.sh
# with the corresponding element in newnames
for i in range(len(newnames)):
    with open('echo.sh', 'r') as f:
        batch = f.read()
    batch = re.sub('TEMPLATE', newnames[i], batch)
    # replace the test set with the new test set enclosed in quotes
    batch = re.sub('TESTSET', '"' + newtestset[i] + '"', batch)


    with open('RF_' + newnames[i] + '.sh', 'w') as f:
        f.write(batch)

# now we need to submit the batch file to the queue using qsub
for i in range(len(newnames)):
    # run a command to submit the batch file to the queue using the -v flag and setting the environment variable PLP
    # to the name of the json file
    os.system('qsub -v PLP=' + '"' + 'RF_' + newnames[i] + '.json' + '"' + ' RF_' + newnames[i] + '.sh')



    # os.system(f'qsub -v PLP="CNN11_{newnames[i]}.json" CNN11_{newnames[i]}.sh')
