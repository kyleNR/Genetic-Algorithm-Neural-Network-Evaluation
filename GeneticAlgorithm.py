#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
import sys

class GA:
    # Config Options
    def __init__(self,
                weights,
                function,
                generations=300,
                population_size=50,
                mutate_chance=0.50,
                elitism=True):
        self.elitism = elitism
        self.generations = generations
        self.population_size = population_size
        self.mutate_chance = mutate_chance
        self.weights = weights
        self.function = function

    def InitialGeneration(self):
        return 10. * np.random.rand(self.population_size, self.weights) - 5.

    def Fitness(self, chromosome):
        difference = self.function(chromosome)
        return difference

    def PopFitness(self, pop):
        fitnesspop = []
        for chromosome in pop:
            fitnesspop.append(self.Fitness(chromosome))
        return fitnesspop

    def Mutate(self, chromosome):
        for num in range(0,random.randint(1,len(chromosome))):
            index = random.randint(0,len(chromosome)-1)
            if random.random() >= 0.5:
                chromosome[index] = chromosome[index] * 1.25
            else:
                chromosome[index] = chromosome[index] * 0.75
        return chromosome

    def Crossover(self, chromosome1, chromosome2):
        if self.weights == 1:
            return chromosome1
        index = random.randint(1, self.weights -1)
        index2 = random.randint(index, self.weights)
        result = np.append(chromosome1[:index], chromosome2[index:index2])
        result = np.append(result, chromosome1[index2:])
        return result

    def TournamentSelection(self, pop):
        parents = np.random.choice(self.population_size,4)
        if self.Fitness(pop[parents[0]]) <= self.Fitness(pop[parents[1]]):
            parent1 = pop[parents[0]]
        else:
            parent1 = pop[parents[1]]

        if self.Fitness(pop[parents[2]]) <= self.Fitness(pop[parents[3]]):
            parent2 = pop[parents[2]]
        else:
            parent2 = pop[parents[3]]

        child = self.Crossover(parent1,parent2)
        if random.random() > self.mutate_chance:
            child = self.Mutate(child)
        return child

    def Evolve(self, pop):
        newpop = []
        bestIndex = np.argsort(self.PopFitness(pop))
        popbest = pop[bestIndex[0]]
        if self.elitism:
            newpop.append(np.array(popbest))
        while len(newpop) != len(pop):
            newpop.append(self.TournamentSelection(pop))
        return newpop

    def NewGeneration(self, pop):
        fbest = np.inf
        bestChromosome = pop[0]
        for gen in range(self.generations):
            pop = self.Evolve(pop)
            fvalues = self.PopFitness(pop)
            idx = np.argsort(fvalues)
            if fbest > fvalues[idx[0]]:
                fbest = fvalues[idx[0]]
                bestChromosome = pop[idx[0]]
            if fbest < 1e-8:  # task achieved
                break
        return bestChromosome

    def Run(self):
        pop = self.InitialGeneration()
        return self.NewGeneration(pop)


def GetBestWeights(weightNum, function):
    ga = GA(weightNum, function)
    return ga.Run()

def GetBestWeightsFull(weightNum, function, generations, population_size, mutate_chance, elitism):
    ga = GA(weightNum, function, generations, population_size, mutate_chance, elitism)
    return ga.Run()
