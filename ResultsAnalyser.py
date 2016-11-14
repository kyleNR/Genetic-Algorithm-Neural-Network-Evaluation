#!/usr/bin/env python
# -*- coding: utf-8 -*-

fileLocation = "NNCOMPAREDATA"
dimensions = [2,3,5]
function_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
instances = range(1, 6) + range(41, 51)

def loadDataDimensions():
	totalErr = 0.
	count = 0
	for dim in dimensions:
		dimTotalErr = 0.
		dimCount = 0
		for fun_id in function_ids:
			funTotalErr = 0.
			funCount = 0
			for instance in instances:
				filename = fileLocation + "/DATA-FunID-%d-DIM-%d-INSTANCE-%d" % (fun_id, dim, instance)
				file_ = open(filename, 'r')
				for line in file_:
					tempList = [float(x.strip()) for x in line.split(',')]
					totalErr = totalErr + float(tempList[2])
					dimTotalErr = dimTotalErr + float(tempList[2])
					funTotalErr = funTotalErr + float(tempList[2])
					count = count + 1
					dimCount = dimCount + 1
					funCount = funCount + 1
				file_.close()
			#print "DIM %d  FUN %d  TOTAL ERROR %f  AVG ERROR %f" %(dim, fun_id, funTotalErr, (funTotalErr / funCount))
		print "DIM %d  TOTAL ERROR %f  AVG ERROR %f" %(dim, dimTotalErr, (dimTotalErr / dimCount))
	print "TOTAL ERROR %f  AVG ERROR %f" %(totalErr, (totalErr / count))

def loadDataFunctions():
	totalErr = 0.
	count = 0
	for fun_id in function_ids:
		funTotalErr = 0.
		funCount = 0
		for dim in dimensions:
			dimTotalErr = 0.
			dimCount = 0
			for instance in instances:
				filename = fileLocation + "/DATA-FunID-%d-DIM-%d-INSTANCE-%d" % (fun_id, dim, instance)
				file_ = open(filename, 'r')
				for line in file_:
					tempList = [float(x.strip()) for x in line.split(',')]
					totalErr = totalErr + float(tempList[2])
					dimTotalErr = dimTotalErr + float(tempList[2])
					funTotalErr = funTotalErr + float(tempList[2])
					count = count + 1
					dimCount = dimCount + 1
					funCount = funCount + 1
				file_.close()
			#print "DIM %d  FUN %d  TOTAL ERROR %f  AVG ERROR %f" %(dim, fun_id, funTotalErr, (funTotalErr / funCount))
		print "FUN %d  TOTAL ERROR %f  AVG ERROR %f" %(fun_id, dimTotalErr, (dimTotalErr / dimCount))
	print "TOTAL ERROR %f  AVG ERROR %f" %(totalErr, (totalErr / count))

loadDataDimensions()
print "\n\n"
loadDataFunctions()
