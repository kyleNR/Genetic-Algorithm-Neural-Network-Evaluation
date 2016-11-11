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

argv = sys.argv[1:] # shortcut for input arguments

datapath = 'NNCompare' if len(argv) < 1 else argv[0]

dimensions = [2,4,8] if len(argv) < 2 else eval(argv[1])
function_ids = [1]
# function_ids = bbobbenchmarks.noisyIDs if len(argv) < 3 else eval(argv[2])
#instances = range(1, 6) if len(argv) < 4 else eval(argv[3])
instances = range(1, 6) + range(41, 51) if len(argv) < 4 else eval(argv[3])

opts = dict(algid='NN Against Monte Carlo',
            comments='')
maxfunevals = '10 * dim' # 10*dim is a short test-experiment taking a few minutes
# INCREMENT maxfunevals SUCCESSIVELY to larger value(s)
minfunevals = 'dim + 2'  # PUT MINIMAL sensible number of EVALUATIONS before to restart
maxrestarts = 10000      # SET to zero if algorithm is entirely deterministic


def run_optimizer(fun, dim, maxfunevals, fun_id, nn, ftarget=-np.Inf):
    """start the optimizer, allowing for some preparation.
    This implementation is an empty template to be filled

    """
    # prepare
    x_start = 8. * np.random.rand(dim) - 4

    # call, REPLACE with optimizer to be tested
    COMPARENN(fun, x_start, maxfunevals, ftarget, nn)

def COMPARENN(fun, x, maxfunevals, ftarget, nn):
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
        nnidx = np.argsort(nnvalues)
        for i in range(len(fvalues)):
            compareCount = compareCount + 1
            err = (fvalues[i] - nnvalues[i])
            if err < 0:
                err = err * -1
            total_err = total_err + err
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
        if fbest < ftarget:  # task achieved
            break
    print "Total Error Between COCO and NN: ",
    print total_err,
    print "Average Error: ",
    print (total_err / compareCount)
    return xbest

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
        nn.train(200, 50, 0.25, True)
        for iinstance in instances:
            f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=iinstance))

            # independent restarts until maxfunevals or ftarget is reached
            for restarts in xrange(maxrestarts + 1):
                if restarts > 0:
                    f.restart('independent restart')  # additional info
                run_optimizer(f.evalfun, dim,  eval(maxfunevals) - f.evaluations,
                              fun_id, nn, f.ftarget)
                if (f.fbest < f.ftarget
                    or f.evaluations + eval(minfunevals) > eval(maxfunevals)):
                    break

            f.finalizerun()

            print('  f%d in %d-D, instance %d: FEs=%d with %d restarts, '
                  'fbest-ftarget=%.4e, elapsed time [h]: %.2f'
                  % (fun_id, dim, iinstance, f.evaluations, restarts,
                     f.fbest - f.ftarget, (time.time()-t0)/60./60.))

        print '      date and time: %s' % (time.asctime())
    print '---- dimension %d-D done ----' % dim
