# Functions implementing interval verification algorithm - the Krawczyk
# operator and its helper functions
# @author Itrat Ahmed Akhter

import time
import numpy as np
import random
import math
from intervalBasics import *
#import mcInterval


'''
Multiply 2 matrices and return the resulting matrix. 
Any matrix can be an interval matrix
'''
def multiplyMats(mat1, mat2):
	isInterval = len(mat1.shape) == 3 or len(mat2.shape) == 3
	if isInterval:
		result = np.zeros((mat1.shape[0],mat2.shape[1],2))
	else:
		result = np.zeros((mat1.shape[0],mat2.shape[1]))
	
	for i in range(mat1.shape[0]):
		for j in range(mat2.shape[1]):
			for k in range(mat2.shape[1]):
				result[i,j] = interval_add(result[i,j], interval_mult(mat1[i,k],mat2[k,j]))

	return result


def multiplyRegMatWithInMat(regMat, inMat):
	regMatAbs = np.absolute(regMat)
	inMatMid = (inMat[:,:,0] + inMat[:,:,1])/2.0
	inMatRad = (inMat[:,:,1] - inMat[:,:,0])/2.0
	newMatMid = np.dot(regMat, inMatMid)
	newMatRad = np.dot(regMatAbs, inMatRad)
	inMatMidAbs = np.absolute(inMatMid)
	upperLimit = np.dot(regMatAbs, inMatMidAbs)
	for i in range(upperLimit.shape[0]):
		for j in range(upperLimit.shape[1]):
			upperLimit[i,j] = np.nextafter(upperLimit[i,j], np.float("inf")) - upperLimit[i,j]
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
	upperLimit = np.dot(regMatAbs, inVecMidAbs)
	for i in range(upperLimit.shape[0]):
		upperLimit[i] = np.nextafter(upperLimit[i], np.float("inf")) - upperLimit[i]
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

	upperLimit = np.dot(inMatMidAbs, inVecMidAbs)
	for i in range(upperLimit.shape[0]):
		upperLimit[i] = np.nextafter(upperLimit[i], np.float("inf")) - upperLimit[i]
	upperLimit = ((inMat.shape[1] + 1)/2.0)*upperLimit
	newVecRad += upperLimit
	resultVec = np.zeros((inMat.shape[0], 2))
	resultVec[:,0] = newVecMid - newVecRad
	resultVec[:,1] = newVecMid + newVecRad
	return resultVec


'''
Subtract 2 matrices. Either of the matrix can be an interval matrix
and return the resulting matrix
'''
def subtractMats(mat1, mat2):
	isInterval = len(mat1.shape) == 3 or len(mat2.shape) == 3
	if isInterval:
		result = np.zeros((mat1.shape[0],mat2.shape[1],2))
	else:
		result = np.zeros((mat1.shape[0],mat2.shape[1]))

	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			result[i,j] = interval_sub(mat1[i,j], mat2[i,j])
	return result

'''
Multiply interval or regular matrix with interval or regular vector
and regular or interval vector depending on whether mat and vec 
are interval or regular
'''
def multiplyMatWithVec(mat,vec):
	isInterval = interval_p(mat[0,0]) or interval_p(vec[0])

	if isInterval:
		result = np.zeros((mat.shape[0],2))
	else:
		result = np.zeros((mat.shape[0],1))
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			result[i] = interval_add(result[i], interval_mult(mat[i,j],vec[j]))
	return result


'''
Turn a regular matrix into an interval matrix by 
subtracting ulp from each element to create a lower bound and 
adding ulp to each element to create an upper bound
'''
def turnRegMatToIntervalMat(mat):
	intervalMat = np.zeros((mat.shape[0], mat.shape[1], 2))
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			intervalMat[i,j] = np.array([np.nextafter(mat[i,j], float("-inf")), np.nextafter(mat[i,j], float("inf"))])

	return intervalMat

'''
Turn a regular vector into an interval vector by 
subtracting ulp from each element to create a lower bound and 
adding ulp to each element to create an upper bound
'''
def turnRegVecToIntervalVec(vec):
	intervalVec = np.zeros((len(vec), 2))
	for i in range(len(vec)):
		intervalVec[i] = np.array([np.nextafter(vec[i], float("-inf")), np.nextafter(vec[i], float("inf"))])

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
	while count < maxIter and (np.linalg.norm(h) > normThresh or count == 0):
		res = model.f(soln)
		res = -np.array(res)
		jac = model.jacobian(soln)
		
		h = np.linalg.lstsq(jac, res)[0]
		soln = soln + h
		if np.less(soln, overallHyper[:,0] - 0.001).any() or np.greater(soln, overallHyper[:,1] + 0.001).any():
			return (False, soln)
		count+=1
	if count >= maxIter and np.linalg.norm(h) > normThresh:
		return(False, soln)
	return (True,soln)

def newtonSingleStep(soln, fSoln, jac, overallHyper):
	res = fSoln
	res = -np.array(res)
	
	h = np.linalg.lstsq(jac, res)[0]
	return soln + h
	'''if np.less(soln, overallHyper[:,0] - 0.001).any() or np.greater(soln, overallHyper[:,1] + 0.001).any():
		return (False, soln)
	return (True,soln)'''




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
	xi_minus_samplePoint = np.zeros((numV, 2))
	for i in range(numV):
		xi_minus_samplePoint[i] = interval_sub(startBounds[i], samplePoint[i])
	

	'''print ("startBounds")
	printHyper(startBounds)
	print ("samplePoint", samplePoint)
	print ("fSamplePoint", fSamplePoint)
	print ("jacInterval", jacInterval)
	print ("jacSamplePoint", jacSamplePoint)'''


	#start = time.time()
	try:
		C = np.linalg.inv(jacSamplePoint)
	except:
		# In case jacSamplePoint is singular
		C = np.linalg.pinv(jacSamplePoint)
	#end = time.time()
	#print ("step1", end - start)
	'''print ("startBounds", startBounds)
	#print ("xi_minus_samplePoint", xi_minus_samplePoint)
	#print ("C")
	#print (C)
	midSample = (samplePoint[:,0] + samplePoint[:,1])/2.0
	midFunval = (funStartBounds[:,0] + funStartBounds[:,1])/2.0
	CmidFunVal = np.dot(C, midFunval)
	xNewton = midSample - CmidFunVal
	radFunVal = np.zeros((numV, 2))
	radFunVal[:,0] = funStartBounds[:,0] - midFunval
	radFunVal[:,1] = funStartBounds[:,1] - midFunval
	radNewton = multiplyMatWithVec(np.absolute(C), radFunVal)
	newHyper = np.zeros((numV, 2))
	#print ("xNewton", xNewton)
	for i in range(numV):
		#print ("xNewton[i]", xNewton[i])
		#print ("radNewton[i]", radNewton[i])
		newHyper[i,:] = interval_add(xNewton[i], radNewton[i])

	print ("newHyper", newHyper)
	if(all([ (interval_lo(newHyper[i]) > interval_lo(startBounds[i])) and (interval_hi(newHyper[i]) < interval_hi(startBounds[i]))
		 for i in range(numV) ])):
		pass

	else:
		intersect = np.zeros((numV,2))
		for i in range(numV):
			#print ("i", i)
			#print ("startBounds", startBounds[i][0], startBounds[i][1])
			#print ("kInterval", kInterval[i][0], kInterval[i][1])
			intersectVar = interval_intersect(newHyper[i], startBounds[i])
			if intersectVar is not None:
				intersect[i] = intersectVar
			else:
				intersect = None
				break

		if intersect is not None:
			#print ("startBounds", startBounds)
			#print ("newtonIntersect", newHyper)
			return [False, startBounds]'''

	#print ("C", C)
	#print ("fSamplePoint", fSamplePoint)
	#start = time.time()
	#C_fSamplePoint = multiplyMatWithVec(C,fSamplePoint)
	#print ("C_fSamplePoint before")
	#print (C_fSamplePoint)
	C_fSamplePoint = multiplyRegMatWithInVec(C, fSamplePoint)
	#print ("C_fSamplePoint after")
	#print (C_fSamplePoint)
	#end = time.time()
	#print ("step2", end - start)
	#print ("C_fSamplePoint", C_fSamplePoint)

	#start = time.time()
	#C_jacInterval = multiplyMats(C,jacInterval)
	#print ("C_jacInterval before")
	#print (C_jacInterval)
	C_jacInterval = multiplyRegMatWithInMat(C, jacInterval)
	#print ("C_jacInterval after")
	#print (C_jacInterval)
	#end = time.time()
	#print ("step3", end - start)

	#print ("C_jacInterval", C_jacInterval)

	#start = time.time()
	I_minus_C_jacInterval = subtractMats(I,C_jacInterval)
	#end = time.time()
	#print ("step4", end - start)

	#print ("I_minus_C_jacInterval", I_minus_C_jacInterval)
	

	#print ("xi_minus_samplePoint", xi_minus_samplePoint)
	#start = time.time()
	#lastTerm = multiplyMatWithVec(I_minus_C_jacInterval, xi_minus_samplePoint)
	#print ("lastTerm before")
	#print (lastTerm)
	lastTerm = multiplyInMatWithInVec(I_minus_C_jacInterval, xi_minus_samplePoint)
	#print ("lastTerm after")
	#print (lastTerm)
	#end = time.time()
	#print ("step5", end - start)

	#print ("lastTerm", lastTerm)
	
	#start = time.time()
	kInterval = np.zeros((numV,2))
	for i in range(numV):
		kInterval[i,:] = interval_add(interval_sub(samplePoint[i], C_fSamplePoint[i]), lastTerm[i])
	end = time.time()
	#print ("step6", end - start)
	
	#print ("startBounds")
	#printHyper(startBounds)
	#print ("kInterval")
	#printHyper(kInterval)
	# if kInterval is in the interior of startBounds, found a unique solution
	if(all([ (interval_lo(kInterval[i]) > interval_lo(startBounds[i])) and (interval_hi(kInterval[i]) < interval_hi(startBounds[i]))
		 for i in range(numV) ])):
		return (True, startBounds)

	
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
			return (False, None)

	return (False, intersect)



'''Print hyperrectangle hyper'''
def printHyper(hyper):
	for i in range(hyper.shape[0]):
		print (hyper[i,0], hyper[i,1])


'''
Find a Newton's solution in hyper. If there exists a solution
inflate current hyper so that solution is in centre and
check the inflated hyper with Krawczyk
@param model defines the problem
@param hyper hyperRectangle
@param epsilonInflation indicates the proportion of hyper-rectangle distance by which the 
 	hyper-rectangle needs to be inflated before the Krawczyk operator is applied
@return Krawczyk result of inflated hyper
'''
def checkInflatedHyper(model, hyper, epsilonBounds):
	startBounds = np.copy(hyper)
	prevIntersect, intersect = newtonInflation(model, startBounds, epsilonBounds)

	if prevIntersect is None:
		return [False, hyper]

	startBounds = np.copy(intersect)
	samplePoint = (startBounds[:,0] + startBounds[:,1])/2.0
	fSamplePoint = np.array(model.f(samplePoint))
	jacSamplePoint = model.jacobian(samplePoint)
	jacInterval = model.jacobian(startBounds)
	
	# Krawczyk update
	kHelpResult = krawczykHelp(startBounds, jacInterval, samplePoint, fSamplePoint, jacSamplePoint)

	if kHelpResult[0]:
		return [True, kHelpResult[1]]
	else:
		return [False, hyper]


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
def checkExistenceOfSolution(model,hyperRectangle, alpha = 1.0, epsilonInflation=0.001, samplePointSing=None, samplePoint=None, fSamplePoint=None, jacSamplePoint=None):
	epsilonBounds = 1e-12
	numV = len(hyperRectangle[0])

	startBounds = np.copy(hyperRectangle)

	# First do an interval arithmetic test
	# Calculate the interval evaluation of the function
	# for hyperrectangle. If any component of the result
	# does not contain zero then the hyperrectangle does not
	# contain any solution
	'''if hasattr(model, 'f'):
		#print ("startBounds", startBounds)
		funVal = model.f(startBounds)
		#print ("funVal", funVal)
		if(any([np.nextafter(funVal[i,0], np.float("-inf"))*np.nextafter(funVal[i,1], np.float("inf")) > np.nextafter(0.0, np.float("inf")) 
				for i in range(numV)])):
			return [False, None]'''

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
		if iteration > 0 or samplePointSing is None:
			samplePointSing = (startBounds[:,0] + startBounds[:,1])/2.0
			samplePoint = turnRegVecToIntervalVec(samplePointSing)
			fSamplePoint = np.array(model.f(samplePoint))
			jacSamplePoint = model.jacobian(samplePointSing)
		#funStartBounds = model.f(startBounds)
		jacInterval = model.jacobian(startBounds)
		
		# Krawczyk update
		#start = time.time()
		kHelpResult = krawczykHelp(startBounds, jacInterval, samplePoint, fSamplePoint, jacSamplePoint)
		#end = time.time()
		#print ("total", end - start)
		
		if kHelpResult[0] or kHelpResult[1] is None:
			if kHelpResult[0]:
				return [True, startBounds]
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

def findSlabs(bigHyper, smallHyper):
	slabs = []
	lenV = bigHyper.shape[0]
	origHyper = np.copy(bigHyper)
	for i in range(lenV):
		newHyper1 = np.copy(origHyper)
		newHyper2 = np.copy(origHyper)
		if newHyper1[i,0] < smallHyper[i,0]:
			newHyper1[i,1] = smallHyper[i,0]
			slabs.append(newHyper1)
		if smallHyper[i,1] < newHyper2[i,1]:
			newHyper2[i,0] = smallHyper[i,1]
			slabs.append(newHyper2)
		origHyper[i,:] = smallHyper[i,:]
	return slabs

def findMaximalHyperFromNewton(model, soln, fSoln, jac, hyper):
	#print ("soln", soln)
	newSoln = newtonSingleStep(soln, fSoln, jac, hyper)
	fNewSoln = model.f(newSoln)
	#soln = newton(model, sample, overallHyper=hyper, normThresh=1e-4,maxIter=1)
	if np.linalg.norm(fNewSoln) > np.linalg.norm(fSoln):
		return [False, None]
	soln = newton(model, newSoln, overallHyper=hyper)
	numV = hyper.shape[0]
	solnInHyper = True
	if soln[0]:
		# the new hyper must contain the solution in the middle and enclose old hyper
		#print ("soln ", soln[1][0], soln[1][1], soln[1][2])
		if np.any(soln[1] < hyper[:,0]) or np.any(soln[1] > hyper[:,1]):
			solnInHyper = False
	else:
		return [False, None]
	if not(solnInHyper):
		return [False, None]

	smallHyper = np.zeros((numV, 2))
	minDiff = np.float("inf")
	startingRad = (hyper[:,1] - hyper[:,0])/2.0
	if soln[0] and solnInHyper:
		#maxDiff = np.zeros((numV))
		#for si in range(numV):
		#	maxDiff[si] = max(abs(hyper[si,1] - soln[1][si]), abs(soln[1][si] - hyper[si,0]))
		
		startingRad = (hyper[:,1] - hyper[:,0])/2.0
		while True:
			smallHyper[:,0] = soln[1] - startingRad
			smallHyper[:,1] = soln[1] + startingRad
			feasHyper = checkExistenceOfSolution(model, smallHyper)
			if feasHyper[0]:
				return feasHyper
			else:
				startingRad = startingRad*0.6



		#print ("smallHyper", smallHyper)
		'''maxHyper = np.copy(smallHyper)
		#print ("maxHyper", maxHyper)
		feasibility = checkExistenceOfSolution(model, np.transpose(maxHyper))
		#print ("maxFeasibility", feasibility)
		if not(feasibility[0]):
			return [False, None]
		while True:
			#print ("maxHyper", maxHyper)
			newHyper = np.copy(maxHyper)
			dist = (maxHyper[:,1] - maxHyper[:,0])/2.0
			newHyper[:,0] = newHyper[:,0] - dist
			newHyper[:,1] = newHyper[:,1] + dist
			#print ("newHyper", newHyper)
			feasNewHyper = checkExistenceOfSolution(model, np.transpose(newHyper))
			if feasNewHyper[0]:
				maxHyper = np.copy(feasNewHyper[1])
			else:
				return [True, maxHyper]'''




def intervalEval(model, startBounds):
	numV = startBounds.shape[0]
	# First do an interval arithmetic test
	# Calculate the interval evaluation of the function
	# for hyperrectangle. If any component of the result
	# does not contain zero then the hyperrectangle does not
	# contain any solution
	if hasattr(model, 'f'):
		#print ("startBounds", startBounds)
		funVal = model.f(startBounds)
		#print ("funVal", funVal)
		if(any([np.nextafter(funVal[i,0], np.float("-inf"))*np.nextafter(funVal[i,1], np.float("inf")) > np.nextafter(0.0, np.float("inf")) 
				for i in range(numV)])):
			return False
	return True



def checkExistenceOfSolutionWithNewton(model,hyperRectangle, alpha = 1.0, epsilonInflation=0.001):
	epsilonBounds = 1e-12
	numV = len(hyperRectangle[0])

	startBounds = np.copy(hyperRectangle)

	# First do an interval arithmetic test
	# Calculate the interval evaluation of the function
	# for hyperrectangle. If any component of the result
	# does not contain zero then the hyperrectangle does not
	# contain any solution
	'''if hasattr(model, 'f'):
		#print ("startBounds", startBounds)
		funVal = model.f(startBounds)
		#print ("funVal", funVal)
		if(any([np.nextafter(funVal[i,0], np.float("-inf"))*np.nextafter(funVal[i,1], np.float("inf")) > np.nextafter(0.0, np.float("inf")) 
				for i in range(numV)])):
			return [False, None]'''

	samplePointSing = (startBounds[:,0] + startBounds[:,1])/2.0
	samplePoint = turnRegVecToIntervalVec(samplePointSing)
	fSamplePoint = np.array(model.f(samplePoint))
	fSamplePointSing = (fSamplePoint[:,0] + fSamplePoint[:,1])/2.0
	jacSamplePoint = model.jacobian(samplePointSing)
	maximalHyperWithNewton = findMaximalHyperFromNewton(model, samplePointSing, fSamplePointSing, jacSamplePoint, startBounds)
	#maximalHyperWithNewton = findMaximalHyperFromNewton(model, startBounds)
	#print ("maximalHyperWithNewton", maximalHyperWithNewton)
	if not(maximalHyperWithNewton[0]):
		#return [False, startBounds]
		return checkExistenceOfSolution(model, hyperRectangle, alpha, epsilonInflation, samplePointSing, samplePoint, fSamplePoint, jacSamplePoint)
	slabs = findSlabs(startBounds, maximalHyperWithNewton[1])
	return (maximalHyperWithNewton[1], slabs)



'''
Use newton's method to find a solution in hyperrectangle hyper.
If a solution exists then construct a hyperrectnagle with the 
solution in center and contains hyper. Return the original hyper
and the inflated hyper.
@param model defines the problem
@param hyper hyperRectangle
@epsilonBounds defines how far away from the hyperrectangle hyper
we can allow the newton solution to be to start the hyperrectangle 
inflation process
@return (hyper, inflatedHyper) if there is a Newton solution within
epsilon bounds of hyper. Otherwise return (None, hyper)
'''
def newtonInflation(model, hyper, epsilonBounds):
	numV = hyper.shape[0]
	exampleVolt = (hyper[:,0] + hyper[:,1])/2.0
	soln = newton(model,exampleVolt)
	prevHyper = None
	solnInHyper = True
	if soln[0]:
		# the new hyper must contain the solution in the middle and enclose old hyper
		#print ("soln ", soln[1][0], soln[1][1], soln[1][2])
		for sl in range(numV):
			if (soln[1][sl] < hyper[sl][0] - epsilonBounds or
				soln[1][sl] > hyper[sl][1] + epsilonBounds):
				solnInHyper = False
				break
	if soln[0] and solnInHyper:
		prevHyper = np.copy(hyper)
		for si in range(numV):
			maxDiff = max(abs(hyper[si,1] - soln[1][si]), abs(soln[1][si] - hyper[si,0]))
			#print ("si", si, "maxDiff", maxDiff)
			if maxDiff < epsilonBounds:
				maxDiff = epsilonBounds
			hyper[si,0] = soln[1][si] - maxDiff
			hyper[si,1] = soln[1][si] + maxDiff

	return (prevHyper, hyper)


