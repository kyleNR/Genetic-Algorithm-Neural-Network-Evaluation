# Genetic-Algorithm-Neural-Network-Evaluation
Evaluating a GA in a Neural Network environment.

##Install
Place Files inside [COCO](http://coco.gforge.inria.fr/) python folder

##Run System
1.  Run *DataGenerator.py* to create Training Data for Neural Network(s)

2.  Run *NNCompareexperiment.py* to run 

##Configuration
Edit *DataGenerator.py* to change dataset or use command line argument:

1.  Number of sets of data
```
python DataGenerator.py
python DataGenerator.py 50
```

Edit *NNCompareexperiment.py* to change GA attributes or use command line arguments:

1.  Generations
2.  Population Size
3.  Mutation Chance (float)
4.  Elitism (boolean)
```
NNCompareexperiment.py
NNCompareexperiment.py 500 50 0.25 True
```

##Further Experiments
1.  Run *ResultNNCompareexperiment.py* to to run the NN experiment and also log the average error between functions to a separate folder

2.  Run *ResultsAnalyser.py* to analyse the results gathered from *ResultNNCompareexperiment.py*

##Credits
Coursework by
- Nicholas Robinson
- Sam McNaughton
