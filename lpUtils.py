import numpy as np
from cvxopt import matrix,solvers

def constructObjMatrix(objString, variableDict):
	objStringList = objString.split(" ")
	objFun = objStringList[0]
	cMat = np.zeros((len(variableDict)))
	#print "variableDict in constructObjMatrix"
	#print variableDict
	for i in range(len(objStringList)):
		#print "objStringList ", i, objStringList[i], type(objStringList[i])
		if objStringList[i] in variableDict:
			#print "objStringList[i-1] ", objStringList[i-1]
			cMat[variableDict[objStringList[i]]] = float(objStringList[i-1])
			if objFun == "max":
				cMat[variableDict[objStringList[i]]] = -float(objStringList[i-1])
	xoptCMat = matrix(cMat)
	#print "xoptCMat"
	#print xoptCMat
	return xoptCMat
	#return cMat

def constructCoeffMatrices(constraintString):
	constraintList = constraintString.split("\n")
	constraintList = constraintList[:len(constraintList)-1]
	variableDict = {}
	variableNumber = 0
	for i in range(len(constraintList)):
		constraint = constraintList[i]
		brokenConstraintList = constraint.split(" ")
		j = 1
		while j < len(brokenConstraintList)-2:
			var = brokenConstraintList[j]
			if var not in variableDict:
				variableDict[var] = variableNumber
				variableNumber+=1
			j+=3
		
	#print "variableDict "
	#print variableDict
	bMat = np.zeros((len(constraintList),1))
	AMat = np.zeros((len(variableDict),len(constraintList)))
	for i in range(len(constraintList)):
		constraint = constraintList[i]
		brokenConstraintList = constraint.split(" ")
		if len(brokenConstraintList) <=1:
			continue
		sign = brokenConstraintList[len(brokenConstraintList)-2]
		bVal = float(brokenConstraintList[len(brokenConstraintList)-1])
		bMat[i][0] = bVal
		if sign == ">=":
			bMat[i][0] = -bVal
		
		j = 1
		while j < len(brokenConstraintList)-2:
			var = brokenConstraintList[j]
			val = float(brokenConstraintList[j-1])
			AMat[variableDict[var],i] = val
			if sign == ">=":
				AMat[variableDict[var],i] = -val
			j+=3		
	AMat = AMat.transpose()
	xoptBMat = matrix(bMat)
	xoptAMat = matrix(AMat)
	#print "xoptBMat"
	#print xoptBMat
	#print "xoptAMat"
	#print xoptAMat
	return [variableDict,xoptAMat, xoptBMat]
	#return [variableDict, AMat, bMat]
