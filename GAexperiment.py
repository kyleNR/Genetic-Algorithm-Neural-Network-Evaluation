#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runs an entire experiment for benchmarking GENETIC ALGORITHM FUNCTION on a testbed.

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
import random

argv = sys.argv[1:] # shortcut for input arguments

datapath = 'Logs/GA' if len(argv) < 1 else argv[0]

#dimensions = [2, 3, 5, 10, 20, 40] if len(argv) < 2 else eval(argv[1])
dimensions = [2] if len(argv) < 2 else eval(argv[1])
#function_ids = bbobbenchmarks.nfreeIDs if len(argv) < 3 else eval(argv[2])
function_ids = [1] if len(argv) < 3 else eval(argv[2])
#instances = range(1, 6) + range(41, 51) if len(argv) < 4 else eval(argv[3])
instances = range(1, 6) + range(41, 51)

maxfunevals = '10 * dim' # 10*dim is a short test-experiment taking a few minutes
# INCREMENT maxfunevals SUCCESSIVELY to larger value(s)
minfunevals = 'dim + 2'  # PUT MINIMAL sensible number of EVALUATIONS before to restart
maxrestarts = 10000	  # SET to zero if algorithm is entirely deterministic

elitism = True
generations = 1000
population_size = 50
mutate_chance = 0.25

opts = dict(algid='Genetic Algorithm',
comments='Elitism: %r  Generations: %d  Population Size: %d  Mutation Chance: %0.2f' % (elitism, generations, population_size, mutate_chance))


def run_optimizer(fun, dim, maxfunevals, ftarget=-np.Inf):
	"""start the optimizer, allowing for some preparation.
	This implementation is an empty template to be filled

	"""
	# prepare
	x_start = 1. * np.random.rand(population_size,dim) - 0.5

	# call, REPLACE with optimizer to be tested
	GA_OPTIMISER(fun, x_start, maxfunevals, ftarget)

def GA_OPTIMISER(fun, x, maxfunevals, ftarget):
	"""samples new points uniformly randomly in [-5,5]^dim and evaluates
	them on fun until maxfunevals or ftarget is reached, or until
	1e8 * dim function evaluations are conducted.
	"""
	dim = len(x)
	maxfunevals = min(1e8 * dim, maxfunevals)
	popsize = min(maxfunevals, 200)
	fbest = np.inf
	pop = x

	for gen in range(generations):
		pop = Evolve(pop, fun, ftarget)
		fvalues = PopFitness(pop, fun, ftarget)
		idx = np.argsort(fvalues)
		if fbest > fvalues[idx[0]]:
			fbest = fvalues[idx[0]]
			xbest = pop[idx[0]]
		if fbest < 1e-8:  # task achieved
			print "WITHIN 1e(-8) OF TARGET"
			break
	#print "BEST VALUE: %.8f\t TARGET: %.8f" % (fbest, ftarget)
	return xbest


def Fitness(chromosome, fun, ftarget):
	difference = fun(chromosome) - ftarget
	if difference < 0:
		difference = difference *  -1
	return difference

def PopFitness(pop, fun, ftarget):
	fitnesspop = []
	for chromosome in pop:
		fitnesspop.append(Fitness(chromosome,fun,ftarget))
	return fitnesspop

def Evolve(pop, fun, ftarget):
	#fvalues = fun(pop)
	newpop = []
	fvalues = PopFitness(pop, fun, ftarget)
	idx = np.argsort(fvalues)
	#print fvalues
	#print idx
	popbest = pop[idx[0]]
	print "BEST VALUE: %.8f\t TARGET: %.8f" % (fun(popbest), ftarget)
	if elitism:
		newpop.append(popbest)

	while len(newpop) != len(pop):
		parents = np.random.choice(population_size,4)
		if Fitness(pop[parents[0]],fun,ftarget) <= Fitness(pop[parents[1]],fun,ftarget):
			parent1 = pop[parents[0]]
		else:
			parent1 = pop[parents[1]]

		if Fitness(pop[parents[2]],fun,ftarget) <= Fitness(pop[parents[3]],fun,ftarget):
			parent2 = pop[parents[2]]
		else:
			parent2 = pop[parents[3]]

		child1 = Crossover(parent1,parent2)
		if random.random() > mutate_chance:
			newpop.append(Mutate(child1))
	return newpop

def Crossover(chromosome1, chromosome2):
	if dim == 1:
		return chromosome1
	index = random.randint(1,dim -1)
	result = np.append(chromosome1[:index],chromosome2[index:])
	return result

def Mutate(chromosome):
	for num in range(0,random.randint(1,len(chromosome))):
		index = random.randint(0,len(chromosome)-1)
		if random.random() >= 0.5:
			chromosome[index] = chromosome[index] + random.random()
		else:
			chromosome[index] = chromosome[index] - random.random()
	return chromosome


t0 = time.time()
np.random.seed(int(t0))

f = fgeneric.LoggingFunction(datapath, **opts)
for dim in dimensions:  # small dimensions first, for CPU reasons
	for fun_id in function_ids:
		for iinstance in instances:
			f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=iinstance))

			# independent restarts until maxfunevals or ftarget is reached
			for restarts in xrange(maxrestarts + 1):
				if restarts > 0:
					f.restart('independent restart')  # additional info
				run_optimizer(f.evalfun, dim,  eval(maxfunevals) - f.evaluations,
							  f.ftarget)
				if (f.fbest < f.ftarget
					or f.evaluations + eval(minfunevals) > eval(maxfunevals)):
					break

			f.finalizerun()

			print('  f%d in %d-D, instance %d: FEs=%d with %d restarts, '
				  'fbest-ftarget=%.4e, elapsed time [h]: %.2f'
				  % (fun_id, dim, iinstance, f.evaluations, restarts,
					 f.fbest - f.ftarget, (time.time()-t0)/60./60.))

		print '	  date and time: %s' % (time.asctime())
	print '---- dimension %d-D done ----' % dim
