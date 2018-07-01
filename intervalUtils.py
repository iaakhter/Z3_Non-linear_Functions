import numpy as np
import copy
import random
import math
from intervalBasics import *


def multiplyRegularMatWithIntervalMat(regMat,intervalMat):
	result = np.zeros((regMat.shape[0],intervalMat.shape[1],2))
	for i in range(regMat.shape[0]):
		for j in range(intervalMat.shape[1]):
			intervalVal = np.zeros(2)
			for k in range(intervalMat.shape[1]):
				intervalVal += interval_mult(regMat[i,k],intervalMat[k,j])
			result[i,j,:] = intervalVal

	return result

def subtractIntervalMatFromRegularMat(regMat,intervalMat):
	mat1 = regMat - intervalMat[:,:,0]
	mat2 = regMat - intervalMat[:,:,1]
	result = np.zeros((regMat.shape[0],regMat.shape[1],2))
	result[:,:,0] = np.minimum(mat1,mat2)
	result[:,:,1] = np.maximum(mat1,mat2)
	return result

def multiplyIntervalMatWithIntervalVec(mat,vec):
	result = np.zeros((mat.shape[0],vec.shape[1]))
	for i in range(mat.shape[0]):
		intervalVal = np.zeros(2)
		for j in range(mat.shape[1]):
			intervalVal += interval_mult(mat[i,j],vec[j])
		result[i,:] = intervalVal
	return result


def volume(hyperRectangle):
	if hyperRectangle is None:
		return None
	vol = 1
	for i in range(hyperRectangle.shape[0]):
		vol *= (hyperRectangle[i,1] - hyperRectangle[i,0])
	return vol

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
	#print ("overallHyper", overallHyper)
	while count < maxIter and (np.linalg.norm(h) > 1e-8 or count == 0) :
		#print ("soln", soln)
		#print ("h", h)
		res = model.f(soln)
		#print ("res", res)
		res = -np.array(res)
		jac = model.jacobian(soln)
		#print ("res", res)
		#print ("jac", jac)
		try:
			h = np.linalg.solve(jac,res)
		except np.linalg.LinAlgError:
			h = np.linalg.lstsq(jac, res)[0]
		#print ("h",  h)
		soln = soln + h
		#print ("new soln", soln)
		if np.less(soln, overallHyper[:,0] - 0.001).any() or np.greater(soln, overallHyper[:,1]+0.001).any():
			return (False, soln)
		count+=1
	if count >= maxIter and np.linalg.norm(h) > 1e-8:
		return(False, soln)
	return (True,soln)


'''
Check existence of solution within a certain hyperRectangle
using the Gauss-Siedel operator - implemented from 
https://epubs.siam.org/doi/abs/10.1137/0727047

return (True, hyper) if hyperRectangle contains a unique
solution and hyper maybe smaller than hyperRectangle containing the solution
return (False, None) if hyperRectangle contains no solution
return (False, hyper) if hyperRectangle may contain more
than 1 solution and hyper maybe smaller than hyperRectangle containing the solutions
'''
def checkExistenceOfSolutionGS(model,hyperRectangle):
	numVolts = len(hyperRectangle[0])
	startBounds = np.zeros((numVolts,2))
	startBounds[:,0] = hyperRectangle[0,:]
	startBounds[:,1] = hyperRectangle[1,:]

	#print "startBounds = " + str(startBounds)
	if hasattr(model, 'f'):
		funVal = model.f(startBounds)
		if(not all([funVal[i,0]*funVal[i,1] <= 1e-12 for i in range(numVolts)])):
			return (False, None)

	midPoint = (startBounds[:,0] + startBounds[:,1])/2.0
	#print "midPoint = " str(midPoint)

	#IMidPoint is the value of the function evaluated at midPoint
	_,_,IMidPoint = np.array(model.oscNum(midPoint))
	#jacMidPoint is the jacobian of the function evaluated at midPoint
	jacMidPoint = model.jacobian(midPoint)
	#print "jacMidPoint= " + str(jacMidPoint)
	
	try:
		C = np.linalg.inv(jacMidPoint)
	except np.linalg.linalg.LinAlgError:
		C = np.linalg.pinv(jacMidPoint)

	#print "C = " + str(C)
	#print "cond(C) = C" + str(np.linalg.cond(C))

	#Jacobian interval matrix for startBounds
	jacInterval = model.jacobianInterval(startBounds)
	#print "jacInterval = " + str(jacInterval)
	
	#Multiply preconditioner with function value at MidPoint
	C_IMidPoint = np.dot(C,IMidPoint)
	#print "C_IMidPoint = " str(C_IMidPoint)

	#Multiply preconditioner with jacobian interval matrix
	C_jacInterval = multiplyRegularMatWithIntervalMat(C,jacInterval)
	#print "C_jacInterval = " + str(C_jacInterval)

	if(any([interval_lo(C_jacInterval[i,i])*interval_hi(C_jacInterval[i,i]) <= 0 for i in range(numVolts)])):
		# C_jacInterval is not diagonal-dominant, give-up on Gauss-Seidel
		return(False, startBounds)

	#Intersection between startBounds and gsInterval
	newBounds = np.copy(startBounds)
	x = np.zeros(startBounds.shape)
	
	#This loop basically calculates gsInterval using 
	#equation 1.8 in the paper https://epubs.siam.org/doi/abs/10.1137/0727047
	for i in range(numVolts):
		newBounds[i] = np.array([0,0])
		x[i] = \
			interval_sub(
				interval_mid(startBounds[i]),
				interval_div(
					interval_add(
						C_IMidPoint[i],
						interval_dotprod(C_jacInterval[i], [interval_r(newBounds[j]) for j in range(numVolts)])),
					C_jacInterval[i][i]))
		xx = interval_intersect(startBounds[i], x[i])
		if(xx is None): newBounds[i] = startBounds[i]
		else: newBounds[i] = xx
	# if(x subseteq startBounds): we have a unique solution
	if(all([ (interval_lo(startBounds[i]) <= interval_lo(x[i])) and (interval_hi(x[i]) <= interval_hi(startBounds[i]))
			 for i in range(numVolts) ])):
		return(True, newBounds)
	# if(x intersect startBounds == emptySet): there is no solution in startBounds
	elif(any([newBounds[i] is None for i in range(numVolts)])):
		return(False, None)
	'''elif(all([interval_r(newBounds[i]) < 0.001 for i in range(numVolts)])):
		print "jacInterval = " + str(jacInterval)
		print "IMidPoint = " + str(IMidPoint)
		print "C_jacInterval = " + str(C_jacInterval)
		print "C_IMidPoint = " + str(C_IMidPoint)
		print "startBounds = " + str(startBounds)
		print "x = " + str(x)
		print "newBounds = " + str(newBounds)
		raise Exception('tiny')'''
	return(False, newBounds)
	

'''
Check existence of solution within a certain hyperRectangle
using the Krawczyk operator
'''
def checkExistenceOfSolution(model,hyperRectangle, alpha = 1.0):
	numV = len(hyperRectangle[0])

	startBounds = np.zeros((numV,2))
	startBounds[:,0] = hyperRectangle[0,:]
	startBounds[:,1] = hyperRectangle[1,:]

	if hasattr(model, 'f'):
		funVal = model.f(startBounds)
		if(not all([funVal[i,0]*funVal[i,1] <= 1e-12 for i in range(numV)])):
			return (False, None)
	constructBiggerHyper = False
	iteration = 0
	while True:
		#print "iteration number: ", iteration
		#print ("startBounds ", startBounds)
		midPoint = (startBounds[:,0] + startBounds[:,1])/2.0
		#print "midPoint"
		#print midPoint
		fMidPoint = np.array(model.f(midPoint))
		jacMidPoint = model.jacobian(midPoint)
		#print "jacMidPoint"
		#print jacMidPoint
		C = np.linalg.inv(jacMidPoint)
		#print "C"
		#print C
		#print "condition number of C", np.linalg.cond(C)

		#print "C ", C
		I = np.identity(numV)

		jacInterval = model.jacobian(startBounds)
		#print "jacInterval"
		#print jacInterval
		#print "IMidPoint"
		#print IMidPoint
		C_fMidPoint = np.dot(C,fMidPoint)
		#print "C_IMidPoint", C_IMidPoint

		C_jacInterval = multiplyRegularMatWithIntervalMat(C,jacInterval)
		#print "C_jacInterval"
		#print C_jacInterval
		I_minus_C_jacInterval = subtractIntervalMatFromRegularMat(I,C_jacInterval)
		#print "I_minus_C_jacInterval"
		#print I_minus_C_jacInterval
		xi_minus_midPoint = np.zeros((numV,2))
		xi_minus_midPoint[:,0] = startBounds[:,0] - midPoint
		xi_minus_midPoint[:,1] = startBounds[:,1] - midPoint
		
		#print "xi_minus_midPoint", xi_minus_midPoint
		lastTerm = multiplyIntervalMatWithIntervalVec(I_minus_C_jacInterval, xi_minus_midPoint)
		#print "lastTerm "
		#print lastTerm

		kInterval = np.zeros((numV,2))
		kInterval[:,0] = midPoint - C_fMidPoint + lastTerm[:,0]
		kInterval[:,1] = midPoint - C_fMidPoint + lastTerm[:,1]

		#print ("kInterval ")
		#print (kInterval)

		# if kInterval is in the interior of startBounds, found a unique solution
		if(all([ (interval_lo(kInterval[i]) > interval_lo(startBounds[i])) and (interval_hi(kInterval[i]) < interval_hi(startBounds[i]))
			 for i in range(numV) ])):
			return (True, kInterval)


		epsilonBounds = 1e-8
		intersect = np.zeros((numV,2))
		for i in range(numV):
			intersectVar = interval_intersect(kInterval[i], startBounds[i])
			if intersectVar is not None:
				intersect[i] = intersectVar
			
			elif np.less_equal(np.absolute(kInterval[i,:] - startBounds[i,:]),epsilonBounds*np.ones((2))).all():
				# being careful about numerical issues here
				intersect[i] = startBounds[i]
			else:
				# no solution
				#print ("problem index", i)
				return (False, None)

		#print ("intersect")
		#print (intersect)

		oldVolume = volume(startBounds)
		newVolume = volume(intersect)
		volReduc = (oldVolume - newVolume)/(oldVolume*1.0)

		tinyIntersect = np.less_equal(intersect[:,1] - intersect[:,0],epsilonBounds*np.ones((numV))).all()
		
		if tinyIntersect or math.isnan(volReduc) or volReduc < alpha:
			# If intersect is tiny, use newton's method to find a solution
			# in the intersect and construct a bigger hyper than intersect
			# with the solution at the center. This is to take care of cases
			# when the solution is at the boundary
			if constructBiggerHyper == False and tinyIntersect:
				constructBiggerHyper = True
				exampleVolt = (intersect[:,0] + intersect[:,1])/2.0
				soln = newton(model,exampleVolt)
				#print ("soln ", soln)
				# the new hyper must contain the solution in the middle and enclose old hyper
				# and then we check for uniqueness of solution in the newer bigger hyperrectangle
				if soln[0]:
					for si in range(numV):
						maxDiff = max(abs(intersect[si,1] - soln[1][si]), abs(soln[1][si] - intersect[si,0]))
						#print ("maxDiff", maxDiff)
						if maxDiff < epsilonBounds:
							maxDiff = epsilonBounds
						#print "maxDiff ", maxDiff
						intersect[si,0] = soln[1][si] - maxDiff
						intersect[si,1] = soln[1][si] + maxDiff

					#print ("bigger hyper ", intersect)
					startBounds = intersect
			else:
				intersect[:,0] = np.maximum(hyperRectangle[0,:], intersect[:,0])
				intersect[:,1] = np.minimum(hyperRectangle[1,:], intersect[:,1])
				return (False,intersect)
		else:
			startBounds = intersect

		iteration += 1

