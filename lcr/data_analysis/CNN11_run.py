# this batch file needs to create a new json file for model configuration
# then create a new batch file for each model configuration
# then submit the batch file to the queue

# first, take CNN11.json, and use a regex to replace the following fields:
# VarList, CompDirs, and Times (all of which are lists)


import json
import re
import os
import sys

# read in the json file
with open('CNN11_template.json', 'r') as f:
    config = json.load(f)

# get the list of variables
VarList = config['VarList']
# get the list of component directories
CompDirs = config['CompDirs']
# get the list of times
Times = config['Times']

newnames = ["prect_mytest"]
newvars = [["PRECT"]]
newcomps = [["zfp_p_10", "zfp_p_16", "zfp_p_22"]]
newtimes = [[2, 6]]
newtestset = "10pct"

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
    batch = re.sub('TESTSET', f"\"{newtestset}\"", batch)
    with open('CNN11_' + newnames[i] + '.sh', 'w') as f:
        f.write(batch)

# now we need to submit the batch file to the queue using qsub
for i in range(len(newnames)):
    os.system(f'qsub -v PLP="CNN11_{newnames[i]}.json" CNN11_{newnames[i]}.sh')
