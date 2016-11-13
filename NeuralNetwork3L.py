#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import GeneticAlgorithm as ga

class Neural_Network(object):
    def __init__(self, fun_id, dim=2):
        self.inputLayerSize = dim
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        self.fun_id = fun_id
        self.testData = self.loadFile()

        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

    def forward(self, X):
        #X = self.descale(X)
        self.z2 = np.dot(X, self.W1)
        #self.a2 = self.tanH(self.z2)
        self.a2 = self.bentId(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        #yHat = self.tanH(self.z3)
        yHat = self.bentId(self.z3)
        #return self.scale(yHat)
        return yHat

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def tanH(self, z):
        return (2/(1 + np.exp(-2 * z))) - 1

    def bentId(self, z):
        return ((np.sqrt((z**2) + 1) - 1) / 2) + z

    def scale(self, z):
        return z * self.scaleValue

    def descale(self, z):
        x2 = []
        for x in z:
            x2.append(x / self.scaleValue)
        return x2

    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def setWeights(self, weights):
        index = 0

        d1 = len(self.W1)
        d2 = len(self.W1[0])
        arrayList = []
        for i in range(d1):
            arrayList.append(np.array(weights[index:index+d2]))
        self.W1 = np.array(arrayList[0])
        for i in range(1,len(arrayList)):
            if i == 1:
                self.W1 = np.append([self.W1], [arrayList[i]], axis = 0)
            else:
                self.W1 = np.append(self.W1, [arrayList[i]], axis = 0)

        d1 = len(self.W2)
        d2 = len(self.W2[0])
        arrayList = []
        for i in range(d1):
            arrayList.append(np.array(weights[index:index+d2]))
        self.W2 = np.array(arrayList[0])
        for i in range(1,len(arrayList)):
            if i == 1:
                self.W2 = np.append([self.W2], [arrayList[i]], axis = 0)
            else:
                self.W2 = np.append(self.W2, [arrayList[i]], axis = 0)

    def weightTest(self, weights):
        self.setWeights(weights)
        cost = 0.
        for dataSet in self.testData:
            setInput = dataSet[:-1]
            setOutput = dataSet[-1:]
            cost = cost + self.costFunction(setInput, setOutput)
        return cost

    def weightAmount(self):
        amount = 0
        amount = amount + (len(self.W1) * len(self.W1[0]))
        amount = amount + (len(self.W2) * len(self.W2[0]))
        return amount

    def loadFile(self):
        filename = "FunctionData/DATA-FunID-%d-DIM-%d" % (self.fun_id, self.inputLayerSize)
        file_ = open(filename, 'r')
        testData = []
        scaleValue = 0
        for line in file_:
            tempList = [float(x.strip()) for x in line.split(', ')]
            if tempList[len(tempList) - 1] > scaleValue:
                scaleValue = tempList[len(tempList) - 1]
            testData.append(tempList)
        self.scaleValue = 2 * int(scaleValue)
        print "SCALE VALUE: ",
        print self.scaleValue
        return testData

    def train(self, generations=100, popsize=50, mutate_chance=0.33, elitism=True):
        self.setWeights(ga.GetBestWeightsFull(self.weightAmount(), self.weightTest, generations, popsize, mutate_chance, elitism))

def Run(fun_id, dim):
    return Neural_Network(fun_id, dim)
