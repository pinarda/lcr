import argparse
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

    return save, vlist, pre, post, opath, cpath, cdirs, ldcpypath, times, storage, navg, stride

def list_of_strings(arg):
    return arg.split(',')

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json", help="json configuration file", type=str, default="RF_local.json")
    parser.add_argument("-t", "--testset", help="test set type", type=str, default="60_25_wholeslice")
    parser.add_argument("-o", "--onlydata", help="whether to fit the model or only generate training and test data", type=bool, default=False)
    parser.add_argument("-m", "--model", help="model type", type=str, default="cnn")
    parser.add_argument("-f", "--feature", help="select a feature to save", type=str, default=None)
    parser.add_argument("-l", "--listfeatures", help="features to use for fitting", type=list_of_strings, default=None)
    parser.add_argument("-x", "--transform", help="data transform", type=str, default="quantile")
    parser.add_argument("-d", "--jobid", help="jobid", type=int, default=0)
    args = parser.parse_args()

    return args