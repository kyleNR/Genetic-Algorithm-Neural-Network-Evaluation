#!/usr/bin/env python
# -*- coding: utf-8 -*-

import NeuralNetwork3L as NN3
import NeuralNetwork4L as NN4
import NeuralNetwork5L as NN5

def PickNN(dim, fun_id):
	return NN4.Run(fun_id, dim)
