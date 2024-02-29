import argparse
import ast
import os
import json
os.environ["HDF5_PLUGIN_PATH"]

def read_parameters_from_json(metajson):
    comp = ""
    lev = []
    save = ""
    vlist = []
    pre = ""
    post = ""
    opath = ""
    cpath = ""
    cdirs = []
    ldcpypath = ""
    times = []
    storage = ""
    navg = 0
    stride=1
    metric = "dssim"
    cutdataset = True

    print("Reading jsonfile", metajson, " ...")
    if not os.path.exists(metajson):
        print("\n")
        print("*************************************************************************************")
        print("Warning: Specified json file does not exist: ", metajson)
        print("*************************************************************************************")
        print("\n")
    else:
        fd = open(metajson)
        metainfo = json.load(fd)
        if 'SaveDir' in metainfo:
            save = metainfo['SaveDir']
        if "VarList" in metainfo:
            vlist = metainfo['VarList']
        if "FilenamePre" in metainfo:
            pre = metainfo['FilenamePre']
        if "FilenamePost" in metainfo:
            post = metainfo['FilenamePost']
        if "OrigPath" in metainfo:
            opath = metainfo['OrigPath']
        if "CompPath" in metainfo:
            cpath = metainfo['CompPath']
        if "CompDirs" in metainfo:
            cdirs = metainfo['CompDirs']
        if "OptLdcpyDevPath" in metainfo:
            ldcpypath = metainfo['OptLdcpyDevPath']
        if "Times" in metainfo:
            times = metainfo['Times']
        if "StorageLoc" in metainfo:
            storage = metainfo['StorageLoc']
        if "Navg" in metainfo:
            navg = metainfo['Navg']
        if "Stride" in metainfo:
            stride = metainfo['Stride']
        if "Metric" in metainfo:
            metric = metainfo['Metric']
        if "CutDataset" in metainfo:
            cutdataset = metainfo['CutDataset']
        if "SubDirs" in metainfo:
            subdirs = metainfo['SubDirs']

    print("Save directory: ", save)
    print("Variable list: ", vlist)
    print("Filename prefix: ", pre)
    print("Filename postfix: ", post)
    print("Original data path: ", opath)
    print("Compressed data path: ", cpath)
    print("Compressed data directories: ", cdirs)
    print("Optimized Ldcpy path: ", ldcpypath)
    print("Times: ", times)
    print("Storage location: ", storage)
    print("Navg: ", navg)
    print("Stride: ", stride)
    print("Metric: ", metric)
    print("CutDataset: ", cutdataset)
    print("SubDirs: ", subdirs)



    return save, vlist, pre, post, opath, cpath, cdirs, ldcpypath, times, storage, navg, stride, metric, cutdataset, subdirs

def list_of_strings(arg):
    return arg.split(',')

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json", help="json configuration file", type=str, default="RF_local_dssim.json")
    parser.add_argument("-t", "--testset", help="test set type", type=str, default="50_50_wholeslice")
    parser.add_argument("-o", "--onlydata", help="whether to fit the model or only generate training and test data", type=ast.literal_eval, default=False)
    parser.add_argument("-m", "--model", help="model type", type=str, default="cnn")
    parser.add_argument("-f", "--feature", help="select a feature to save", type=str, default="mean")
    parser.add_argument("-l", "--listfeatures", help="features to use for fitting", type=list_of_strings, default="mean")
    parser.add_argument("-x", "--transform", help="data transform", type=str, default="quantile")
    parser.add_argument("-d", "--jobid", help="jobid", type=int, default=1)
    parser.add_argument("-r", "--metric", help="metric (default dssim)", type=str, default="dssim")
    parser.add_argument("-c", "--cutdataset", help="whether to cut the dataset into windows", type=bool, default=0)
    parser.add_argument("-a", '--runonlydata', help='Run only data?', type=ast.literal_eval, default=False)
    parser.add_argument("-y", '--labelsonly', help='Run only labels?', type=ast.literal_eval, default=True)

    print(f"Only Data Status: {parser.parse_args().onlydata}")
    # let's add a -v option as well to add debug messages
    args = parser.parse_args()

    return args