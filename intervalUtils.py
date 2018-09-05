# @author Itrat Ahmed Akhter

import numpy as np
import random
import math
from intervalBasics import *


'''
Multiply regular matrix with interval matrix
@param regMat regular matrix
@param intervalMat interval matrix
@return resulting interval matrix
'''
def multiplyRegularMatWithIntervalMat(regMat,intervalMat):
	result = np.zeros((regMat.shape[0],intervalMat.shape[1],2))
	for i in range(regMat.shape[0]):
		for j in range(intervalMat.shape[1]):
			intervalVal = np.zeros(2)
			for k in range(intervalMat.shape[1]):
				intervalVal += interval_round(interval_mult(regMat[i,k],intervalMat[k,j]))
				intervalVal = interval_round(intervalVal)
			result[i,j,:] = interval_round(intervalVal)

	return result


'''
Subtract interval matrix from regular matrix
and return the resulting matrix
@param regMat regular matrix
@param intervalMat interval matrix
@return resulting matrix
'''
def subtractIntervalMatFromRegularMat(regMat,intervalMat):
	mat1 = regMat - intervalMat[:,:,0]
	mat2 = regMat - intervalMat[:,:,1]
	result = np.zeros((regMat.shape[0],regMat.shape[1],2))
	result[:,:,0] = np.minimum(mat1,mat2)
	result[:,:,1] = np.maximum(mat1,mat2)
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			result[i,j,0] = np.nextafter(result[i,j,0], np.float("-inf"))
			result[i,j,1] = np.nextafter(result[i,j,1], np.float("inf"))
	return result

'''
Multiply interval or regular matrix with interval or regular vector
@param mat regular or interval matrix
@param vec regular or interval vector
@return regular or interval vector depending on whether mat and vec 
		are interval or regular
'''
def multiplyMatWithVec(mat,vec):
	isInterval = interval_p(mat[0,0]) or interval_p(vec[0])

	if isInterval:
		result = np.zeros((mat.shape[0],2))
	else:
		result = np.zeros((mat.shape[0],1))
	for i in range(mat.shape[0]):
		if isInterval:
			intervalVal = np.zeros((2))
		else:
			intervalVal = np.zeros((1))
		for j in range(mat.shape[1]):
			mult = interval_mult(mat[i,j],vec[j])
			if isInterval:
				intervalVal += interval_round(mult)
				intervalVal = interval_round(intervalVal)
			else:
				intervalVal += mult
		if isInterval:
			result[i,:] = interval_round(intervalVal)
		else:
			result[i,:] = intervalVal
	return result


'''
Return the volume of the hyperrectangle
@param hyperRectangle hyperrectangle
@return the volume of the hyperRectangle
'''
def volume(hyperRectangle):
	if hyperRectangle is None:
		return None
	vol = 1
	for i in range(hyperRectangle.shape[0]):
		vol *= (hyperRectangle[i,1] - hyperRectangle[i,0])
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
def newton(model,soln):
	h = soln
	count = 0
	maxIter = 100
	bounds = model.bounds
	lenV = len(soln)
	overallHyper = np.zeros((lenV,2))
	for i in range(lenV):
		overallHyper[i,0] = bounds[i][0]
		overallHyper[i,1] = bounds[i][1]
	while count < maxIter and (np.linalg.norm(h) > 1e-8 or count == 0):
		res = model.f(soln)
		res = -np.array(res)
		jac = model.jacobian(soln)
		try:
			h = np.linalg.solve(jac,res)
		except np.linalg.LinAlgError:
			h = np.linalg.lstsq(jac, res)[0]
		soln = soln + h
		if np.less(soln, overallHyper[:,0] - 0.001).any() or np.greater(soln, overallHyper[:,1]+0.001).any():
			return (False, soln)
		count+=1
	if count >= maxIter and np.linalg.norm(h) > 1e-8:
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
	epsilonBounds = 1e-12

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

	#print ("C", C)
	if interval_p(fSamplePoint[0]):
		C_fSamplePoint = multiplyMatWithVec(C, fSamplePoint)
	else:
		C_fSamplePoint = np.dot(C,fSamplePoint)
	#print ("C_fSamplePoint", C_fSamplePoint)

	C_jacInterval = multiplyRegularMatWithIntervalMat(C,jacInterval)

	#print ("C_jacInterval", C_jacInterval)

	I_minus_C_jacInterval = subtractIntervalMatFromRegularMat(I,C_jacInterval)

	#print ("I_minus_C_jacInterval", I_minus_C_jacInterval)
	
	xi_minus_samplePoint = np.column_stack((startBounds[:,0] - samplePoint, startBounds[:,1] - samplePoint))
	
	xi_minus_samplePoint = np.array([interval_round(xi_minus_samplePoint[i]) for i in range(numV)])

	#print ("xi_minus_samplePoint", xi_minus_samplePoint)
	lastTerm = multiplyMatWithVec(I_minus_C_jacInterval, xi_minus_samplePoint)

	#print ("lastTerm", lastTerm)
	
	kInterval = np.zeros((numV,2))
	for i in range(numV):
		kInterval[i,:] = interval_round(interval_add(interval_round(interval_sub(samplePoint[i], C_fSamplePoint[i])), lastTerm[i]))

	#print ("startBounds", startBounds)
	#print ("kInterval", kInterval)
	# if kInterval is in the interior of startBounds, found a unique solution
	if(all([ (interval_lo(kInterval[i]) > interval_lo(startBounds[i])) and (interval_hi(kInterval[i]) < interval_hi(startBounds[i]))
		 for i in range(numV) ])):
		return (True, kInterval)

	
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
@return (True, refinedHyper) if hyperrectangle contains a unique solution.
		refinedHyper also contains the solution and might be smaller
		than hyperRectangle
@return (False, refinedHyper) if hyperrectangle may contain more
		than one solution. refinedHyper also contains all the solutions
		that hyperRectangle might contain and might be smaller than
		hyperRectangle
@return (False, None) if hyperrectangle contains no solution
'''
def checkExistenceOfSolution(model,hyperRectangle, alpha = 1.0):
	epsilonBounds = 1e-12
	numV = len(hyperRectangle[0])

	startBounds = np.zeros((numV,2))
	startBounds[:,0] = hyperRectangle[0,:]
	startBounds[:,1] = hyperRectangle[1,:]

	# First do an interval arithmetic test
	# Calculate the interval evaluation of the function
	# for hyperrectangle. If any component of the result
	# does not contain zero then the hyperrectangle does not
	# contain any solution
	if hasattr(model, 'f'):
		funVal = model.f(startBounds)
		if(any([np.nextafter(funVal[i,0], np.float("-inf"))*np.nextafter(funVal[i,1], np.float("inf")) > np.nextafter(0.0, np.float("inf")) 
				for i in range(numV)])):
			return [False, None]

	# Start the Krawczyk update
	constructBiggerHyper = False
	iteration = 0
	prevIntersect = None
	while True:
		samplePoint = (startBounds[:,0] + startBounds[:,1])/2.0
		fSamplePoint = np.array(model.f(samplePoint))
		jacSamplePoint = model.jacobian(samplePoint)
		jacInterval = model.jacobian(startBounds)
		
		# Krawczyk update
		kHelpResult = krawczykHelp(startBounds, jacInterval, samplePoint, fSamplePoint, jacSamplePoint)

		if kHelpResult[0] or kHelpResult[1] is None:
			if kHelpResult[0]:
				return [True, kHelpResult[1]]
			return kHelpResult
		
		intersect = kHelpResult[1]


		oldVolume = volume(startBounds)
		newVolume = volume(intersect)
		volReduc = (oldVolume - newVolume)/(oldVolume*1.0)

		# If the reduction of volume is less than equal to alpha
		# then do no more Krawczyk updates. We are done
		if (math.isnan(volReduc) or volReduc <= alpha):
			# Use newton's method to find a solution
			# in the intersect and construct a bigger hyper than intersect
			# with the solution at the center. This is to take care of cases
			# when the solution is at the boundary
			if constructBiggerHyper == False:
				constructBiggerHyper = True
				prevIntersect, intersect = newtonInflation(model, intersect, epsilonBounds)
				if prevIntersect is None:
					return [False, intersect]
				else:
					startBounds = intersect

			else:
				intersect[:,0] = np.maximum(prevIntersect[:,0], intersect[:,0])
				intersect[:,1] = np.minimum(prevIntersect[:,1], intersect[:,1])
				return [False,intersect]
		else:
			startBounds = intersect

		iteration += 1


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


