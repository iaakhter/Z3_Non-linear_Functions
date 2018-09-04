import numpy as np
import random
import math
from intervalBasics import *


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
		#print ("calculating inverse")
		C = np.linalg.inv(jacSamplePoint)
	except:
		# If this has happened then the jacInterval is seriously
		# inconditioned even though doesn't contain all zeros in a row
		# or a column
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

			'''elif np.less_equal(np.absolute(kInterval[i,:] - startBounds[i,:]),epsilonBounds*np.ones((2))).all():
				# being careful about numerical issues here
				intersect[i] = startBounds[i]'''
		else:
			# no solution
			return (False, None)

	return (False, intersect)




def determineDegeneracy(model, startBounds, epsilonBounds):
	numV = startBounds.shape[0]
	samplePoint = (startBounds[:,0] + startBounds[:,1])/2.0
	fSamplePoint = np.array(model.f(samplePoint))
	jacSamplePoint = model.jacobian(samplePoint)
	jacInterval = model.jacobian(startBounds)

	degenRow, degenCol = None, None

	condJac = np.linalg.cond(jacSamplePoint)
	print ("condJac", condJac)

	if condJac > 1e+6:
		prevIntersect, intersect = newtonInflation(model, startBounds, epsilonBounds)
		if prevIntersect is not None:	
			# Find the degen row - row with the minimum sum across columns
			colSum = np.sum(np.absolute(jacSamplePoint), axis = 1)
			degenRow = np.argmin(colSum)

			# Find the degen column - column with the minimum sum across rows
			rowSum = np.sum(np.absolute(jacSamplePoint), axis = 0)
			degenCol = np.argmin(rowSum)
		
	return degenRow, degenCol


# Assume that startBounds is very tiny and contains a solution (confirmed by Newton's method)
def krawczykDecoupled(model, startBounds, degenRow, degenCol, epsilonBounds, alpha = 1.0):
	# while loop to deal with system not including degenRow and degenCol
	constructBiggerHyper = False
	iteration = 0
	startBounds3d = np.copy(startBounds)
	prevIntersect = None

	samplePoint = (startBounds[:,0] + startBounds[:,1])/2.0
	rad = (startBounds[:,1] - startBounds[:,0])/2.0
	fSamplePoint = np.array(model.f(samplePoint))
	jacSamplePoint = model.jacobian(samplePoint)
	jacInterval = model.jacobian(startBounds)


	jacIntWeak = np.delete(jacInterval[:,degenCol], degenRow, axis = 0)
	contribFromWeak = np.zeros((len(jacIntWeak), 2))
	for cI in range(len(jacIntWeak)):
		contribFromWeak[cI,:] = interval_mult(jacIntWeak[cI], rad[degenCol])
	
	fSamplePoint = np.delete(fSamplePoint, degenRow, axis = 0)
	fSamplePointWithContrib = np.zeros((len(fSamplePoint), 2))
	#print ("fSamplePoint before", fSamplePoint.shape)
	#print (fSamplePoint)
	for fi in range(len(fSamplePoint)):
		fSamplePointWithContrib[fi,:] = interval_add(fSamplePoint[fi], contribFromWeak[fi,:])
	#print ("fSamplePointWithContrib", fSamplePointWithContrib.shape)
	#print (fSamplePointWithContrib)
	
	kHelpResult = krawczykHelp(np.delete(startBounds, degenCol, axis = 0), 
								np.delete(np.delete(jacInterval, degenRow, axis = 0), degenCol, axis = 1),
								np.delete(samplePoint, degenCol, axis = 0), fSamplePointWithContrib, 
								np.delete(np.delete(jacSamplePoint, degenRow, axis = 0), degenCol, axis = 1))
	
	#print ("kHelpResult")
	#print (kHelpResult)

	if kHelpResult[0] or kHelpResult[1] is None:
		if kHelpResult[0]:
			intersect = kHelpResult[1]
			intersect = np.insert(intersect, [degenCol], startBounds3d[degenCol], axis = 0)
			startBounds = intersect
			#print ("found converging stuff")
		else:
			raise Exception("Error: krawczykDecoupled::strongly coupled system should have contained a solution as confirmed by Newton's method")
	
	else:
		return [False, startBounds3d]

	# Deal with system with just degenRow and degenCol
	samplePoint = (startBounds[:,0] + startBounds[:,1])/2.0
	rad = (startBounds[:,1] - startBounds[:,0])/2.0
	fSamplePoint = np.array(model.f(samplePoint))
	jacSamplePoint = model.jacobian(samplePoint)
	jacInterval = model.jacobian(startBounds)
	
	jacIntStrong = np.delete(jacInterval[degenRow, :], degenCol, axis = 0)
	radStrong = np.delete(rad, degenCol, axis = 0)
	contribFromStrong = multiplyMatWithVec(np.expand_dims(jacIntStrong, axis = 0), radStrong)
	#print ("contribFromStrong", contribFromStrong)

	fSamplePointWithContrib = interval_add(fSamplePoint[degenRow], contribFromStrong)
	#print ("fSamplePointWithContrib", fSamplePointWithContrib)
	
	kHelpResult = krawczykHelp(np.expand_dims(startBounds[degenCol, :], axis = 0), 
								np.expand_dims(np.expand_dims(jacInterval[degenRow, degenCol, :], axis = 0), axis = 1), 
								np.array([samplePoint[degenCol]]), fSamplePointWithContrib, 
								np.expand_dims(np.expand_dims(jacSamplePoint[degenRow, degenCol], axis = 0), axis = 1))
	
	#print ("kHelpResult")
	#print (kHelpResult)

	if kHelpResult[0] or kHelpResult[1] is None:
		if kHelpResult[0]:
			intersect = np.copy(startBounds3d)
			intersect[degenCol, :] = kHelpResult[1]
			return [True, intersect, '0dim']
		else:
			raise Exception("Error: krawczykDecoupled::weakly coupled system should have contained a solution as confirmed by Newton's method")
	
	return [False, startBounds3d]
	


# Print hyperrectangle
def printHyper(hyper):
	for i in range(hyper.shape[0]):
		print (hyper[i,0], hyper[i,1])

'''
Check existence of solution within a certain hyperRectangle
using the Krawczyk operator
'''
def checkExistenceOfSolution(model,hyperRectangle, alpha = 1.0):
	epsilonBounds = 1e-12
	numV = len(hyperRectangle[0])

	startBounds = np.zeros((numV,2))
	startBounds[:,0] = hyperRectangle[0,:]
	startBounds[:,1] = hyperRectangle[1,:]

	if hasattr(model, 'f'):
		funVal = model.f(startBounds)
		'''if(any([np.nextafter(funVal[i,0], np.float("-inf"))*np.nextafter(funVal[i,1], np.float("inf")) > epsilonBounds 
				for i in range(numV)])):
			return [False, None]'''
		if(any([np.nextafter(funVal[i,0], np.float("-inf"))*np.nextafter(funVal[i,1], np.float("inf")) > np.nextafter(0.0, np.float("inf")) 
				for i in range(numV)])):
			return [False, None]

	constructBiggerHyper = False
	iteration = 0
	prevIntersect = None
	while True:
		samplePoint = (startBounds[:,0] + startBounds[:,1])/2.0
		fSamplePoint = np.array(model.f(samplePoint))
		jacSamplePoint = model.jacobian(samplePoint)
		jacInterval = model.jacobian(startBounds)
		kHelpResult = krawczykHelp(startBounds, jacInterval, samplePoint, fSamplePoint, jacSamplePoint)
		#print ("startBounds")
		#printHyper(startBounds)
		#print (startBounds)
		#print ("jacInterval")
		#print (jacInterval)
		
		
		#print ("kHelpResult")
		#print (kHelpResult)

		if kHelpResult[0] or kHelpResult[1] is None:
			if kHelpResult[0]:
				return [True, kHelpResult[1], "0dim"]
			return kHelpResult
		
		intersect = kHelpResult[1]


		oldVolume = volume(startBounds)
		newVolume = volume(intersect)
		volReduc = (oldVolume - newVolume)/(oldVolume*1.0)

		'''print ("oldVolume", oldVolume)
		print ("newVolume", newVolume)
		print ("volReduc", volReduc)
		print ("alpha", alpha)'''
		
		if (math.isnan(volReduc) or volReduc <= alpha):
			# If intersect is tiny, use newton's method to find a solution
			# in the intersect and construct a bigger hyper than intersect
			# with the solution at the center. This is to take care of cases
			# when the solution is at the boundary
			#print ("Coming here?")
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

def newtonInflation(model, intersect, epsilonBounds):
	numV = intersect.shape[0]
	exampleVolt = (intersect[:,0] + intersect[:,1])/2.0
	soln = newton(model,exampleVolt)
	# the new hyper must contain the solution in the middle and enclose old hyper
	# and then we check for uniqueness of solution in the newer bigger hyperrectangle
	prevIntersect = None
	solnInIntersect = True
	if soln[0]:
		#print ("soln ", soln[1][0], soln[1][1], soln[1][2])
		for sl in range(numV):
			if (soln[1][sl] < intersect[sl][0] - epsilonBounds or
				soln[1][sl] > intersect[sl][1] + epsilonBounds):
				solnInIntersect = False
				break
	if soln[0] and solnInIntersect:
		prevIntersect = np.copy(intersect)
		for si in range(numV):
			maxDiff = max(abs(intersect[si,1] - soln[1][si]), abs(soln[1][si] - intersect[si,0]))
			#print ("si", si, "maxDiff", maxDiff)
			if maxDiff < epsilonBounds:
				maxDiff = epsilonBounds
			intersect[si,0] = soln[1][si] - maxDiff
			intersect[si,1] = soln[1][si] + maxDiff

	return (prevIntersect, intersect)


