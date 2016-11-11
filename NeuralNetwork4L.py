#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import GeneticAlgorithm as ga

# Whole Class with additions:
class Neural_Network(object):
    def __init__(self, fun_id, scaleValue):
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        self.hiddenLayerSize2 = 2
        self.fun_id = fun_id
        self.scaleValue = scaleValue
        self.testData = self.loadFile()

        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.hiddenLayerSize2)
        self.W3 = np.random.randn(self.hiddenLayerSize2,self.outputLayerSize)

    def forward(self, X):
        #Propagate inputs through network
        #X = self.descale(X)
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.tanH(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.tanH(self.z3)
        self.z4 = np.dot(self.a3, self.W3)
        yHat = self.tanH(self.z4)
        return self.scale(yHat)
        #return yHat

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

    def tanH(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return (2/(1 + np.exp(-2 * z))) - 1

    def scale(self, z):
        return z * self.scaleValue

    def descale(self, z):
        x2 = []
        for x in z:
            x2.append(x / self.scaleValue)
        return x2

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def setWeights(self, weights):
        d1 = len(self.W1)
        d2 = len(self.W1[0])
        index = 0
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

        d1 = len(self.W3)
        d2 = len(self.W3[0])
        arrayList = []
        for i in range(d1):
            arrayList.append(np.array(weights[index:index+d2]))
        self.W3 = np.array(arrayList[0])
        for i in range(1,len(arrayList)):
            if i == 1:
                self.W3 = np.append([self.W3], [arrayList[i]], axis = 0)
            else:
                self.W3 = np.append(self.W3, [arrayList[i]], axis = 0)

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
        amount = amount + (len(self.W3) * len(self.W3[0]))
        return amount

    def loadFile(self):
        filename = "FunctionData/DATA-FunID-%d-DIM-%d" % (self.fun_id, self.inputLayerSize)
        file_ = open(filename, 'r')
        testData = []
        for line in file_:
            testData.append([float(x.strip()) for x in line.split(', ')])
        return testData

NN = Neural_Network(1, 100)
print "INPUT: (%f, %f)" % (-0.504408146844,0.428269435691)
input_ = np.array(([-0.504408146844,0.428269435691]), dtype=float)
print "OUTPUT: %.8f" % NN.forward(input_)

newWeights = ga.GetBestWeights(NN.weightAmount(), NN.weightTest)
NN.setWeights(newWeights)
print "TRAINED OUTPUT: %.8f" % NN.forward(input_)
print "INTENDED OUTPUT: %.8f" % (-78.982882327)
