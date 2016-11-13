#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runs COCO to generate a set of Inputs and Outputs
   Corresponding to Dim and to Fun_ID
   Used to train NN
"""
import sys # in case we want to control what to run via command line args
import time
import numpy as np
import fgeneric
import bbobbenchmarks
import os

argv = sys.argv[1:]

datapath = 'DataGen'

dataLength = 100 if len(argv) < 1 else int(argv[0])  #Number of sets of data
dim = [2,3,5] #List of Dimensions to save

fileLocation = "FunctionData"

if not os.path.exists(fileLocation):
    os.makedirs(fileLocation)

def DataLog(fun_id, dim):

    filename = fileLocation + "/DATA-FunID-%d-DIM-%d" % (fun_id, dim)
    file_ = open(filename, 'w')
    for _ in range(dataLength):
        xpop = 10. * np.random.rand(dim) - 5
        line = ""
        for num in xpop:
            line = line + str(num) + ", "
        line = line + str(f.evalfun(xpop)) + "\n"
        file_.write(line)

def DataLogAllFun(dimNum):
    for fid in range(1,25):
        f.setfun(*bbobbenchmarks.instantiate(fid, 1))
        DataLog(fid, dimNum)

f = fgeneric.LoggingFunction(datapath)
#f.setfun(*bbobbenchmarks.instantiate(fun_id, 0))
for dimNum in dim:
    DataLogAllFun(dimNum)
