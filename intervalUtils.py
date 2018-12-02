# Functions implementing interval verification algorithm - the Krawczyk
# operator and its helper functions
# @author Itrat Ahmed Akhter

import time
import numpy as np
import random
import math
from intervalBasics import *


def multiplyRegMatWithInMat(regMat, inMat):
	regMatAbs = np.absolute(regMat)
	inMatMid = (inMat[:,:,0] + inMat[:,:,1])/2.0
	inMatRad = (inMat[:,:,1] - inMat[:,:,0])/2.0
	newMatMid = np.dot(regMat, inMatMid)
	newMatRad = np.dot(regMatAbs, inMatRad)
	inMatMidAbs = np.absolute(inMatMid)
	upperLimitMid = np.dot(regMatAbs, inMatMidAbs)
	upperLimitMidWithUlp = np.nextafter(upperLimitMid, np.float("inf"))
	upperLimitRadWithUlp = np.nextafter(newMatRad, np.float("inf"))
	upperLimit = upperLimitMidWithUlp - upperLimitMid + upperLimitRadWithUlp - newMatRad
	upperLimit = ((regMat.shape[1] + 1)/2.0)*upperLimit
	newMatRad += upperLimit
	resultMat = np.zeros((regMat.shape[0], regMat.shape[1], 2))
	resultMat[:,:,0] = newMatMid - newMatRad
	resultMat[:,:,1] = newMatMid + newMatRad
	return resultMat

def multiplyRegMatWithInVec(regMat, inVec):
	regMatAbs = np.absolute(regMat)
	inVecMid = (inVec[:,0] + inVec[:,1])/2.0
	inVecRad = (inVec[:,1] - inVec[:,0])/2.0
	newVecMid = np.dot(regMat, inVecMid)
	newVecRad = np.dot(regMatAbs, inVecRad)
	inVecMidAbs = np.absolute(inVecMid)
	upperLimitMid = np.dot(regMatAbs, inVecMidAbs)
	upperLimitMidWithUlp = np.nextafter(upperLimitMid, np.float("inf"))
	upperLimitRadWithUlp = np.nextafter(newVecRad, np.float("inf"))
	upperLimit = upperLimitMidWithUlp - upperLimitMid + upperLimitRadWithUlp - newVecRad
	upperLimit = ((regMat.shape[1] + 1)/2.0)*upperLimit
	newVecRad += upperLimit
	resultVec = np.zeros((regMat.shape[0], 2))
	resultVec[:,0] = newVecMid - newVecRad
	resultVec[:,1] = newVecMid + newVecRad
	return resultVec


def multiplyInMatWithInVec(inMat, inVec):
	inMatMid = (inMat[:,:,0] + inMat[:,:,1])/2.0
	inMatMidAbs = np.absolute(inMatMid)
	inMatRad = (inMat[:,:,1] - inMat[:,:,0])/2.0
	inVecMid = (inVec[:,0] + inVec[:,1])/2.0
	inVecMidAbs = np.absolute(inVecMid)
	inVecRad = (inVec[:,1] - inVec[:,0])/2.0

	newVecMid = np.dot(inMatMid, inVecMid)
	newVecRad = np.dot(inMatMidAbs, inVecRad) + np.dot(inVecMidAbs, inMatRad) + np.dot(inMatRad, inVecRad)

	upperLimitMid = np.dot(inMatMidAbs, inVecMidAbs)
	upperLimitMidWithUlp = np.nextafter(upperLimitMid, np.float("inf"))
	upperLimitRadWithUlp = np.nextafter(newVecRad, np.float("inf"))
	upperLimit = upperLimitMidWithUlp - upperLimitMid + upperLimitRadWithUlp - newVecRad
	upperLimit = ((inMat.shape[1] + 1)/2.0)*upperLimit
	newVecRad += upperLimit
	resultVec = np.zeros((inMat.shape[0], 2))
	resultVec[:,0] = newVecMid - newVecRad
	resultVec[:,1] = newVecMid + newVecRad
	return resultVec

def multiplyInMatWithInVecZeroMid(inMat, inVec):
	inMatMid = (inMat[:,:,0] + inMat[:,:,1])/2.0
	inMatMidAbs = np.absolute(inMatMid)
	inMatRad = (inMat[:,:,1] - inMat[:,:,0])/2.0
	inVecRad = (inVec[:,1] - inVec[:,0])/2.0

	#newVecMid = np.dot(inMatMid, inVecMid)
	newVecMid = np.zeros((inVec.shape[0]))
	#newVecRad = np.dot(inMatMidAbs, inVecRad) + np.dot(inVecMidAbs, inMatRad) + np.dot(inMatRad, inVecRad)
	newVecRad = np.dot(inMatMidAbs, inVecRad) + np.dot(inMatRad, inVecRad)

	upperLimitRadWithUlp = np.nextafter(newVecRad, np.float("inf"))
	upperLimit = upperLimitRadWithUlp - newVecRad
	upperLimit = ((inMat.shape[1] + 1)/2.0)*upperLimit
	newVecRad += upperLimit
	resultVec = np.zeros((inMat.shape[0], 2))
	resultVec[:,0] = newVecMid - newVecRad
	resultVec[:,1] = newVecMid + newVecRad
	return resultVec


def subtractInMatFromRegMat(regMat, inMat):
	resultMat = np.zeros((regMat.shape[0], regMat.shape[1], 2))
	resultMat[:,:,1] = regMat - inMat[:,:,0]
	resultMat[:,:,0] = regMat - inMat[:,:,1]
	resultMat[:,:,0] = np.nextafter(resultMat[:,:,0], np.float("-inf"))
	resultMat[:,:,1] = np.nextafter(resultMat[:,:,1], np.float("inf"))
	return resultMat

def subtractInVecFromInVec(vec1, vec2):
	resultVec = np.zeros((vec1.shape[0], 2))
	resultVec[:,0] = vec1[:,0] - vec2[:,1]
	resultVec[:,1] = vec1[:,1] - vec2[:,0]
	resultVec[:,0] = np.nextafter(resultVec[:,0], np.float("-inf"))
	resultVec[:,1] = np.nextafter(resultVec[:,1], np.float("inf"))
	return resultVec

def addInVecToInVec(vec1, vec2):
	resultVec = np.zeros((vec1.shape[0], 2))
	resultVec[:,0] = vec1[:,0] + vec2[:,0]
	resultVec[:,1] = vec1[:,1] + vec2[:,1]
	resultVec[:,0] = np.nextafter(resultVec[:,0], np.float("-inf"))
	resultVec[:,1] = np.nextafter(resultVec[:,1], np.float("inf"))
	return resultVec


'''
Turn a regular matrix into an interval matrix by 
subtracting ulp from each element to create a lower bound and 
adding ulp to each element to create an upper bound
'''
def turnRegMatToIntervalMat(mat):
	intervalMat = np.zeros((mat.shape[0], mat.shape[1], 2))
	intervalMat[:,:,0] = np.nextafter(mat, np.float("-inf"))
	intervalMat[:,:,1] = np.nextafter(mat, np.float("inf"))

	return intervalMat

'''
Turn a regular vector into an interval vector by 
subtracting ulp from each element to create a lower bound and 
adding ulp to each element to create an upper bound
'''
def turnRegVecToIntervalVec(vec):
	intervalVec = np.zeros((len(vec), 2))
	intervalVec[:,0] = np.nextafter(vec, np.float("-inf"))
	intervalVec[:,1] = np.nextafter(vec, np.float("inf"))

	return intervalVec


'''
Return the volume of the hyperrectangle
'''
def volume(hyperRectangle):
	if hyperRectangle is None:
		return None
	vol = 1
	hyperDist = hyperRectangle[:,1] - hyperRectangle[:,0]
	for i in range(hyperRectangle.shape[0]):
		vol *= hyperDist[i]
	return vol

'''
Apply newton's method to find a solution using
function defined by model
@param model defines the problem
@param soln the starting point for Newton's method
@return (False, soln) if Newton's method doesn't find a solution
					within the bounds defined by model
@return (True, soln) if Newton's method can find a solution within
					the bounds defined by model
'''
def newton(model, soln, overallHyper=None, normThresh=1e-8, maxIter=100):
	h = soln
	count = 0
	maxIter = 100
	bounds = model.bounds
	lenV = len(soln)
	if overallHyper is None:
		overallHyper = np.zeros((lenV,2))
		overallHyper[:,0] = bounds[:,0]
		overallHyper[:,1] = bounds[:,1]
	oldRes, newRes = None, None
	while count < maxIter and (np.linalg.norm(h) > normThresh or count == 0):
		newRes = model.f(soln)
		res = -np.array(newRes)
		jac = model.jacobian(soln)
		
		h = np.linalg.lstsq(jac, res, rcond=None)[0]
		soln = soln + h
		if ((oldRes is not None and np.linalg.norm(newRes) > np.linalg.norm(oldRes)) or
			np.less(soln, overallHyper[:,0] - 0.001).any() or np.greater(soln, overallHyper[:,1] + 0.001).any()):
			return (False, soln)
		count+=1
		oldRes = np.copy(newRes)
	if count >= maxIter and np.linalg.norm(h) > normThresh:
		return(False, soln)
	return (True,soln)



'''
Do a krawczyk update on hyperrectangle defined by startBounds
@param startBounds hyperrectangle
@param jacInterval interval jacobian over startBounds
@param samplePoint mid point in startBounds
@param fSamplePoint function evaluation at samplePoint
@param jacSamplePoint jacobian at samplePoint
@return (True, refinedHyper) if hyperrectangle contains a unique solution.
		refinedHyper also contains the solution and might be smaller
		than hyperRectangle
@return (False, refinedHyper) if hyperrectangle may contain more
		than one solution. refinedHyper also contains all the solutions
		that hyperRectangle might contain and might be smaller than
		hyperRectangle
@return (False, None) if hyperrectangle contains no solution
'''
def krawczykHelp(startBounds, jacInterval, samplePoint, fSamplePoint, jacSamplePoint):
	numV = startBounds.shape[0]
	I = np.identity(numV)
	xi_minus_samplePoint = subtractInVecFromInVec(startBounds, samplePoint)
	

	'''print ("startBounds")
	printHyper(startBounds)
	print ("samplePoint", samplePoint)
	print ("fSamplePoint", fSamplePoint)
	print ("jacInterval", jacInterval)
	print ("jacSamplePoint", jacSamplePoint)'''


	try:
		C = np.linalg.inv(jacSamplePoint)
	except:
		# In case jacSamplePoint is singular
		C = np.linalg.pinv(jacSamplePoint)
	
	C_fSamplePoint = multiplyRegMatWithInVec(C, fSamplePoint)
	#print ("C_fSamplePoint after")
	#print (C_fSamplePoint)

	
	C_jacInterval = multiplyRegMatWithInMat(C, jacInterval)
	#print ("C_jacInterval after")
	#print (C_jacInterval)

	
	I_minus_C_jacInterval = subtractInMatFromRegMat(I, C_jacInterval)
	#print ("I_minus_C_jacInterval after")
	#print (I_minus_C_jacInterval)
	
	lastTerm = multiplyInMatWithInVecZeroMid(I_minus_C_jacInterval, xi_minus_samplePoint)
	#print ("lastTerm after")
	#print (lastTerm)
	
	kInterval = addInVecToInVec(subtractInVecFromInVec(samplePoint, C_fSamplePoint), lastTerm)
	#print ("kInterval after")
	#print (kInterval)
	
	#print ("startBounds")
	#printHyper(startBounds)
	#print ("kInterval")
	#printHyper(kInterval)
	# if kInterval is in the interior of startBounds, found a unique solution
	if np.all(kInterval[:,0] > startBounds[:,0]) and np.all(kInterval[:,1] < startBounds[:,1]):
		return [True, kInterval]
	
	intersect = np.zeros((numV,2))
	for i in range(numV):
		#print ("i", i)
		#print ("startBounds", startBounds[i][0], startBounds[i][1])
		#print ("kInterval", kInterval[i][0], kInterval[i][1])
		intersectVar = interval_intersect(kInterval[i], startBounds[i])
		if intersectVar is not None:
			intersect[i] = intersectVar
		else:
			# no solution
			return [False, None]

	return [False, intersect]



'''Print hyperrectangle hyper'''
def printHyper(hyper):
	for i in range(hyper.shape[0]):
		print (hyper[i,0], hyper[i,1])




'''
Check whether hyperrectangle hyperRectangle contains
a unique solution, no solution or maybe more than one solution
to function identified by the model with Krawczyk operator. 
Use rounded interval arithmetic for every operation in the 
Krawczyk update
@param model defines the problem
@param hyperRectangle the hyperRectangle
@param alpha indicates how many times the Krawczyk operator is 
		used to refine hyperRectangle before the function returns
		If the reduction in volume is below alpha, then we are done
@param epsilonInflation the amount by which either side of the hyper-rectangle
		is inflated before applying the Krawczyk method. This allows for quicker
		convergence to a unique solution if one exists
@return (True, refinedHyper) if hyperrectangle contains a unique solution.
		refinedHyper also contains the solution and might be smaller
		than hyperRectangle
@return (False, refinedHyper) if hyperrectangle may contain more
		than one solution. refinedHyper also contains all the solutions
		that hyperRectangle might contain and might be smaller than
		hyperRectangle
@return (False, None) if hyperrectangle contains no solution
'''
def checkExistenceOfSolution(model,hyperRectangle, alpha = 1.0, epsilonInflation=0.001):
	epsilonBounds = 1e-12
	numV = len(hyperRectangle[0])

	startBounds = np.copy(hyperRectangle)


	# Start the Krawczyk update
	iteration = 0
	while True:
		oldVolume = volume(startBounds)
		#print ("startBounds before")
		#printHyper(startBounds)
		dist = startBounds[:,1] - startBounds[:,0]
		startBounds[:,0] = startBounds[:,0] - (epsilonInflation*dist + epsilonBounds)
		startBounds[:,1] = startBounds[:,1] + (epsilonInflation*dist + epsilonBounds)
	
		#print ("startBounds after")
		#printHyper(startBounds)
		samplePointSing = (startBounds[:,0] + startBounds[:,1])/2.0
		samplePoint = turnRegVecToIntervalVec(samplePointSing)
		fSamplePoint = np.array(model.f(samplePoint))
		jacSamplePoint = model.jacobian(samplePointSing)
		jacInterval = model.jacobian(startBounds)
	
		kHelpResult = krawczykHelp(startBounds, jacInterval, samplePoint, fSamplePoint, jacSamplePoint)
		
		if kHelpResult[0] or kHelpResult[1] is None:
			return kHelpResult
		
		intersect = kHelpResult[1]
		#print("intersect")
		#printHyper(intersect)

		newVolume = volume(intersect)
		volReduc = (oldVolume - newVolume)/(oldVolume*1.0)
		#print ("volReduc", volReduc)

		# If the reduction of volume is less than equal to alpha
		# then do no more Krawczyk updates. We are done
		if (math.isnan(volReduc) or volReduc <= alpha):
			intersect[:,0] = np.maximum(hyperRectangle[:,0], intersect[:,0])
			intersect[:,1] = np.minimum(hyperRectangle[:,1], intersect[:,1])
			return [False,intersect]
		else:
			startBounds = intersect

		iteration += 1




# First do an interval arithmetic test
# Calculate the interval evaluation of the function
# for hyperrectangle. If any component of the result
# does not contain zero then the hyperrectangle does not
# contain any solution and return False in that case
def intervalEval(model, startBounds):
	numV = startBounds.shape[0]
	if hasattr(model, 'f'):
		#print ("startBounds", startBounds)
		funVal = model.f(startBounds)
		#print ("funVal", funVal)
		funValMult = np.multiply(funVal[:,0], funVal[:,1])
		if np.any(funValMult > 0.0):
			return False
	return True

