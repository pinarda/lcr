# this batch file needs to create a new json file for model configuration
# then create a new batch file for each model configuration
# then submit the batch file to the queue

# first, take CNN11.json, and use a regex to replace the following fields:
# VarList, CompDirs, and Times (all of which are lists)


import json
import re
import os

if __name__ == '__main__':
    # read in the json file
    with open('RF_template.json', 'r') as f:
        config = json.load(f)

    # get the list of variables
    VarList = config['VarList']
    # get the list of component directories
    CompDirs = config['CompDirs']
    # get the list of times
    Times = config['Times']

    newnames = ["TS1_jfl",
                "TS2_jfl",
                "TS3_jfl",
                "TS4_jfl",
                "TS5_jfl",
                "TS6_jfl",
                "TS7_jfl",]
    newvars = [#["TS", "T500", "TSMX", "TSMN"],
                ["TS"],
                ["TS"],
                ["TS"],
                ["TS"],
                ["TS"],
                ["TS"],
                ["TS", "T010", "T200", "T500", "T850", "TREFHTMN", "TREFHT"]]
    newcomps = [["zfp_p_10"],
                ["zfp_p_10"],
                ["zfp_p_10"],
                ["zfp_p_10"],
                ["zfp_p_10"],
                ["zfp_p_10"],
                ["zfp_p_10"]]
    newtimes = [[6, 11, 16],
                [6, 11, 16],
                [6, 11, 16],
                [6],
                [6],
                [6],
                [6]]
    newtestset = ["60_25_wholeslice",
                  "60_25_wholeslice",
                  "60_25_wholeslice",
                  "60_25_wholeslice",
                  "60_25_wholeslice",
                  "60_25_wholeslice",
                  "1var"]
    jobids = [1,
              2,
              3,
              4,
              5,
              6,
              7]
    metrics = ["dssim",
               "mse",
               "logdssim",
               "dssim",
               "mse",
               "logdssim",
               "dssim"]
    transforms = ["quantile",
                    "quantile",
                    "quantile",
                    "none",
                    "none",
                    "none",
                    "quantile"]


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

        # replace the job id with the new job id
        batch = re.sub('JOBID', str(jobids[i]), batch)
        # replace the metric with the new metric
        batch = re.sub('METRIC', metrics[i], batch)
        # replace the transform with the new transform
        batch = re.sub('TRANSFORM', transforms[i], batch)


        with open('RF_' + newnames[i] + '.sh', 'w') as f:
            f.write(batch)

    # now we need to submit the batch file to the queue using qsub
    for i in range(len(newnames)):
        # run a command to submit the batch file to the queue using the -v flag and setting the environment variable PLP
        # to the name of the json file
        os.system('qsub -v PLP=' + '"' + 'RF_' + newnames[i] + '.json' + '"' + ' RF_' + newnames[i] + '.sh')



        # os.system(f'qsub -v PLP="CNN11_{newnames[i]}.json" CNN11_{newnames[i]}.sh')
