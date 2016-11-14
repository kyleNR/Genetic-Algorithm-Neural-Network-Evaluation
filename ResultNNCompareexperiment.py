#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runs an entire experiment for benchmarking PURE_RANDOM_SEARCH on a testbed.

CAPITALIZATION indicates code adaptations to be made.
This script as well as files bbobbenchmarks.py and fgeneric.py need to be
in the current working directory.

Under unix-like systems:
    nohup nice python exampleexperiment.py [data_path [dimensions [functions [instances]]]] > output.txt &

"""
import sys # in case we want to control what to run via command line args
import time
import numpy as np
import fgeneric
import bbobbenchmarks
import NeuralNetworkControl as NNC
import os

argv = sys.argv[1:] # shortcut for input arguments

generations = 200 if len(argv) < 1 else int(argv[0])
population_size = 50 if len(argv) < 2 else int(argv[1])
mutate_chance = 0.25 if len(argv) < 3 else float(argv[2])
elitism = True if len(argv) < 4 else argv[3] == "True"

datapath = 'NNCompare'
fileLocation = "NNCOMPAREDATA"
if not os.path.exists(fileLocation):
    os.makedirs(fileLocation)

dimensions = [2,3,5]
#function_ids = [1,2,3]
function_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
instances = range(1, 6) + range(41, 51)

opts = dict(algid='NN Comparison using Monte Carlo Search',
            comments='Generations: %d  Population Size: %d  Mutation Chance: %0.2f  Elitism: %r  ' % (generations, population_size, mutate_chance, elitism))
maxfunevals = '10 * dim' # 10*dim is a short test-experiment taking a few minutes
# INCREMENT maxfunevals SUCCESSIVELY to larger value(s)
minfunevals = 'dim + 2'  # PUT MINIMAL sensible number of EVALUATIONS before to restart
maxrestarts = 100      # SET to zero if algorithm is entirely deterministic


def run_optimizer(fun, dim, maxfunevals, fun_id, file_, nn, ftarget=-np.Inf):
    """start the optimizer, allowing for some preparation.
    This implementation is an empty template to be filled

    """
    # prepare
    x_start = 8. * np.random.rand(dim) - 4

    # call, REPLACE with optimizer to be tested
    COMPARENN(fun, x_start, maxfunevals, ftarget, fun_id, file_, nn)

def COMPARENN(fun, x, maxfunevals, ftarget, fun_id, file_, nn):
    """samples new points uniformly randomly in [-5,5]^dim and evaluates
    them on fun until maxfunevals or ftarget is reached, or until
    1e8 * dim function evaluations are conducted.

    """
    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)
    fbest = np.inf
    total_err = 0.
    compareCount = 0

    for _ in range(0, int(np.ceil(maxfunevals / popsize))):
        xpop = 10. * np.random.rand(popsize, dim) - 5.
        fvalues = fun(xpop)
        nnvalues = nnfun(xpop, nn)
        idx = np.argsort(fvalues)
        for i in range(len(fvalues)):
            compareCount = compareCount + 1
            err = (fvalues[i] - nnvalues[i])
            if err < 0:
                err = err * -1
            line = str(dim) + "," + str(fun_id) + "," + str(err[0]) + "\n"
            file_.write(line)
            total_err = total_err + err
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
        if fbest < ftarget:  # task achieved
            break
    avgErr = (total_err / compareCount)
    return avgErr

def nnfun(xpop, nn):
    nnvalues = []
    for pop in xpop:
        nnvalues.append(nn.forward(pop))
    return nnvalues

t0 = time.time()
np.random.seed(int(t0))

f = fgeneric.LoggingFunction(datapath, **opts)
for dim in dimensions:  # small dimensions first, for CPU reasons
    for fun_id in function_ids:
        nn = NNC.PickNN(dim, fun_id)
        nn.train(generations, population_size, mutate_chance, elitism)
        for iinstance in instances:
            filename = fileLocation + "/DATA-FunID-%d-DIM-%d-INSTANCE-%d" % (fun_id, dim, iinstance)
            file_ = open(filename, 'w')

            f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=iinstance))

            # independent restarts until maxfunevals or ftarget is reached
            for restarts in xrange(maxrestarts + 1):
                if restarts > 0:
                    f.restart('independent restart')  # additional info
                run_optimizer(f.evalfun, dim,  eval(maxfunevals) - f.evaluations,
                              fun_id, file_, nn, f.ftarget)
                if (f.fbest < f.ftarget
                    or f.evaluations + eval(minfunevals) > eval(maxfunevals)):
                    break

            file_.close()

            f.finalizerun()

            print('  f%d in %d-D, instance %d: FEs=%d with %d restarts, '
                  'fbest-ftarget=%.4e, elapsed time [h]: %.2f'
                  % (fun_id, dim, iinstance, f.evaluations, restarts,
                     f.fbest - f.ftarget, (time.time()-t0)/60./60.))

        print '      date and time: %s' % (time.asctime())
    print '---- dimension %d-D done ----' % dim
