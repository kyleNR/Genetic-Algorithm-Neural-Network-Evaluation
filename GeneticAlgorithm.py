#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

class GA:
    # Config Options
    def __init__(self,
                elitism=True,
                generations=30,
                population_size=50,
                mutate_chance=0.25,
                weights,
                functionInput,
                targetOutput,
                function):
        self.elitism = elitism
        self.generations = generations
        self.population_size = population_size
        self.mutate_chance = mutate_chance
        self.weights = weights
        self.functionInput = functionInput
        self.targetOutput = targetOutput
        self.function = function

    def InitialGeneration(self):
        return np.random.rand(self.population_size, self.weights)

    def Fitness(self, chromosome):
        difference = self.function(functionInput, chromosome) / self.targetOutput
        if difference < 0:
            difference = difference * -1
        return difference

    def PopFitness(self, pop):
    	fitnesspop = []
    	for chromosome in pop:
    		fitnesspop.append(Fitness(chromosome))
    	return fitnesspop

    def Mutate(self, chromosome):
        for num in range(0,random.randint(1,len(chromosome))):
    		index = random.randint(0,len(chromosome)-1)
    		if random.random() >= 0.5:
    			chromosome[index] = chromosome[index] + random.random()
    		else:
    			chromosome[index] = chromosome[index] - random.random()
    	return chromosome

    def Crossover(self, chromosome1, chromosome2):
        if self.weights == 1:
            return chromosome1
        index = random.randint(1, self.weights -1)
        result = np.append(chromosome1[:index], chromosome2[index:])
        return result

    def TournamentSelection(self, pop):
        parents = np.random.choice(self.population_size,4)
        if Fitness(pop[parents[0]]) >= Fitness(pop[parents[1]]):
            parent1 = pop[parents[0]]
        else:
            parent1 = pop[parents[1]]

        if Fitness(pop[parents[2]]) >= Fitness(pop[parents[3]]):
            parent2 = pop[parents[2]]
        else:
            parent2 = pop[parents[3]]

        child = Crossover(parent1,parent2)
        if random.random() > self.mutate_chance:
            child = Mutate(child)
        return child

    def Evolve(self, pop):
        newpop = []
        popbest = pop[(np.argsort(pop)[0])]
        if elitism:
            newpop.append(popbest)
        while len(newpop) != len(pop):
            newpop.append(TournamentSelection(pop))
        return newpop

    def NewGeneration(self, pop):
        fbest = np.inf
        for gen in self.generations:
            pop = Evolve(pop)
            fvalues = PopFitness(pop)
    		idx = np.argsort(fvalues)
    		if fbest > fvalues[idx[0]]:
    			fbest = fvalues[idx[0]]
    			bestChromosome = pop[idx[0]]
    		if fbest < 1e-8:  # task achieved
    			break
        return bestChromosome

    def Run():
        pop = InitialGeneration()
        return NewGeneration(pop)


def NewGA(elitism, generations, population_size, mutate_chance, weightNum, functionInput, targetOutput, function):
    ga = GA(elitism, generations, population_size, mutate_chance, weightNum, functionInput, targetOutput, function)
    return ga.Run()

def NewGA(weightNum, functionInput, targetOutput, function):
    ga = GA(weightNum, functionInput, targetOutput, function)
    return ga.Run()
