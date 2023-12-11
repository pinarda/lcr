# this batch file needs to create a new json file for model configuration
# then create a new batch file for each model configuration
# then submit the batch file to the queue

# first, take CNN11.json, and use a regex to replace the following fields:
# VarList, CompDirs, and Times (all of which are lists)


import json
import re
import os

TEST = True

if __name__ == '__main__':
    # read in the json file
    with open('RF_template5.json', 'r') as f:
        config = json.load(f)

    # get the list of variables
    VarList = config['VarList']
    # get the list of component directories
    CompDirs = config['CompDirs']
    # get the list of times
    Times = config['Times']


    newnames = ["unwindowed-quant",
                "unwindowed-noquant",
                "windowed-quant",
                "windowed-noquant"]

    # append PRECT1 ... PRECT8 to newnames
    if not TEST:
        for varname in ["PRECT", "PSL", "FLUT", "Z500"]:
            for i in range(1, 5):
                stri = str(i)
                newnames.append(f"{varname}_{stri}_jflj")
    newvars = [#["TS", "T500", "TSMX", "TSMN"],
                ["TS"],
                ["TS"],
                ["TS"],
                ["TS"]]
    # append PRECT to newvars
    if not TEST:
        for varname in ["PRECT", "PSL", "FLUT", "Z500"]:
            for i in range(1, 5):
                newvars.append([varname])
    newcomps = [["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
                ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
                ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
                ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"]]
    # repeat the above for PRECT
    if not TEST:
        for varname in ["PRECT", "PSL", "FLUT", "Z500"]:
            for i in range(1, 5):
                if i % 2 == 0:
                    newcomps.append(["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"])
                else:
                    newcomps.append(["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"])
    newtimes = [[100],
                [100],
                [100],
                [100]]
    # repeat the above for PRECT
    if not TEST:
        for varname in ["PRECT", "PSL", "FLUT", "Z500"]:
            for i in range(1, 5):
                if i % 2 == 0:
                    newtimes.append([100])
                else:
                    newtimes.append([100])
    newtestset = ["10_90_wholeslice",
                  "10_90_wholeslice",
                  "10_90_wholeslice",
                  "10_90_wholeslice"]
    # repeat the above for PRECT
    if not TEST:
        for varname in ["PRECT", "PSL", "FLUT", "Z500"]:
            for i in range(1, 5):
                newtestset.append("10_90_wholeslice")
    jobids = [1,
              2,
              3,
              4]
    # repeat the above for PRECT
    if not TEST:
        for varname in ["PRECT", "PSL", "FLUT", "Z500"]:
            for i in range(1, 5):
                jobids.append(i)

    metrics = ["dssim",
               "dssim",
               "dssim",
               "dssim"]
    # repeat the above for PRECT
    if not TEST:
        for varname in ["PRECT", "PSL", "FLUT", "Z500"]:
            for i in range(1, 5):
                metrics.append("dssim")
    transforms = ["quantile",
                    "none",
                    "quantile",
                    "none"]
    # repeat the above for PRECT
    if not TEST:
        for varname in ["PRECT", "PSL", "FLUT", "Z500"]:
            for i in range(1, 5):
                if i < 3:
                    transforms.append("quantile")
                else:
                    transforms.append("none")
    cutdatasets = [
        False,
        False,
        True,
        True
    ]
    if not TEST:
        for varname in ["PRECT", "PSL", "FLUT", "Z500"]:
            # repeat the above for PRECT, should be two True, two False, repeated twice
            for i in range(1, 5):
                cutdatasets.append(cutdatasets[i-1])





    # we need to write a new json file for each model configuration
    # so we need to loop over variables, component directories, and times simultaneously
    for i in range(len(newvars)):
        newconfig = config.copy()
        newconfig['VarList'] = newvars[i]
        newconfig['CompDirs'] = newcomps[i]
        newconfig['Times'] = newtimes[i]
        newconfig['CutDataset'] = cutdatasets[i]
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
