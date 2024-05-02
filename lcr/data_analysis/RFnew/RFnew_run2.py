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

    newnames = ["TODAYSPLOTS111",
                "TODAYSPLOTS211",
                "TODAYSPLOTS311",
                "TODAYSPLOTS411",
                "TODAYSPLOTS511",
                "TODAYSPLOTS611",
                "TODAYSPLOTS711",
                "TODAYSPLOTS811",
                "TODAYSPLOTS111",
                "TODAYSPLOTS211",
                "TODAYSPLOTS311",
                "TODAYSPLOTS411",
                "TODAYSPLOTS511",
                "TODAYSPLOTS611",
                "TODAYSPLOTS711",
                "TODAYSPLOTS811"
                ]
    newvars = [["PRECT"],
               ["TS"],
               ["Z050"],
               ["LHFLX"],
               ["ICEFRAC"],
               ["FLUT"],
               ["PSL"],
               ["TS"],
               ["PRECT"],
               ["TS"],
               ["Z050"],
               ["LHFLX"],
               ["ICEFRAC"],
               ["FLUT"],
               ["PSL"],
               ["TS"]
               ]
    # newcomps = [["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"]]
    newcomps = [["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18"],
                ["zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
                ["zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
                ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18"],
                ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18"],
                ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18"],
                ["zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
                ["zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
                ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18"],
                ["zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
                ["zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
                ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18"],
                ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18"],
                ["zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18"],
                ["zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"],
                ["zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"]
                ]
    newtimes = [[200],
                [200],
                [200],
                [200],
                [200],
                [200],
                [200],
                [200],
                [400],
                [400],
                [400],
                [400],
                [400],
                [400],
                [400],
                [400]
                ]
    newtestset = ["50_50_wholeslice",
                  "50_50_wholeslice",
                  "50_50_wholeslice",
                  "50_50_wholeslice",
                  "50_50_wholeslice",
                  "50_50_wholeslice",
                  "50_50_wholeslice",
                  "50_50_wholeslice",
                  "50_50_wholeslice",
                  "50_50_wholeslice",
                  "50_50_wholeslice",
                  "50_50_wholeslice",
                  "50_50_wholeslice",
                  "50_50_wholeslice",
                  "50_50_wholeslice",
                  "50_50_wholeslice"
                  ]
    jobids = [1,
              2,
              3,
              4,
              5,
              6,
              7,
              8,
              9,
              10,
              11,
              12,
              13,
              14,
              15]
    metrics = ["dssim",
               "dssim",
               "dssim",
               "dssim",
               "dssim",
               "dssim",
               "dssim",
               "dssim",
               "dssim",
               "dssim",
               "dssim",
               "dssim",
               "dssim",
               "dssim",
               "dssim",
               "dssim"
               ]
    metric = [["dssim"],
              ["dssim"],
              ["dssim"],
              ["dssim"],
              ["dssim"],
              ["dssim"],
              ["dssim"],
              ["dssim"],
              ["dssim"],
              ["dssim"],
              ["dssim"],
              ["dssim"],
              ["dssim"],
              ["dssim"],
              ["dssim"],
              ["dssim"]
              ]
    transforms = [ "none",
                   "none",
                   "none",
                   "none",
                   "none",
                   "none",
                   "none",
                   "none",
                   "none",
                   "none",
                   "none",
                   "none",
                   "none",
                   "none",
                   "none",
                   "none"
                   ]
    cutdatasets = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True
    ]


    # we need to write a new json file for each model configuration
    # so we need to loop over variables, component directories, and times simultaneously
    for i in range(len(newvars)):
        newconfig = config.copy()
        newconfig['VarList'] = newvars[i]
        newconfig['CompDirs'] = newcomps[i]
        newconfig['Times'] = newtimes[i]
        newconfig['CutDataset'] = cutdatasets[i]
        newconfig['Metric'] = metric[i]
        # write the new json file
        with open('run_casper_' + newnames[i] + '.json', 'w') as f:
            json.dump(newconfig, f)

    # now we need to create a new batch file for each model configuration
    # all we have to do here is replace the string "TEMPLATE" in CNN11_template.sh
    # with the corresponding element in newnames
    for i in range(len(newnames)):
        with open('run_casper.sh', 'r') as f:
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


        with open('run_casper_' + newnames[i] + '.sh', 'w') as f:
            f.write(batch)

    # now we need to submit the batch file to the queue using qsub
    for i in range(len(newnames)):
        # run a command to submit the batch file to the queue using the -v flag and setting the environment variable PLP
        # to the name of the json file
        os.system('qsub -v PLP=' + '"' + 'run_casper_' + newnames[i] + '.json' + '"' + ' run_casper_' + newnames[i] + '.sh')



        # os.system(f'qsub -v PLP="CNN11_{newnames[i]}.json" CNN11_{newnames[i]}.sh')
