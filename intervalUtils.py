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
				intervalVal += interval_mult(regMat[i,k],intervalMat[k,j])
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

def multiplyIntervalMatWithIntervalVec(mat,vec):
	result = np.zeros((mat.shape[0],vec.shape[1]))
	for i in range(mat.shape[0]):
		intervalVal = np.zeros(2)
		for j in range(mat.shape[1]):
			mult = interval_mult(mat[i,j],vec[j])
			intervalVal += mult
		result[i,:] = interval_round(intervalVal)
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

	try:
		C = np.linalg.inv(jacSamplePoint)
	except:
		# If this has happened then the jacInterval is seriously
		# inconditioned even though doesn't contain all zeros in a row
		# or a column
		C = np.linalg.pinv(jacSamplePoint)

	C_fSamplePoint = np.dot(C,fSamplePoint)

	C_jacInterval = multiplyRegularMatWithIntervalMat(C,jacInterval)

	I_minus_C_jacInterval = subtractIntervalMatFromRegularMat(I,C_jacInterval)
	
	xi_minus_samplePoint = np.column_stack((startBounds[:,0] - samplePoint, startBounds[:,1] - samplePoint))
	
	xi_minus_samplePoint = np.array([interval_round(xi_minus_samplePoint[i]) for i in range(numV)])

	lastTerm = multiplyIntervalMatWithIntervalVec(I_minus_C_jacInterval, xi_minus_samplePoint)
	
	kInterval = np.column_stack((samplePoint - C_fSamplePoint + lastTerm[:,0], samplePoint - C_fSamplePoint + lastTerm[:,1]))

	kInterval = np.array([interval_round(kInterval[i]) for i in range(numV)])

	# if kInterval is in the interior of startBounds, found a unique solution
	if(all([ (interval_lo(kInterval[i]) > interval_lo(startBounds[i])) and (interval_hi(kInterval[i]) < interval_hi(startBounds[i]))
		 for i in range(numV) ])):
		return (True, kInterval)


	intersect = np.zeros((numV,2))
	for i in range(numV):
		intersectVar = interval_intersect(kInterval[i], startBounds[i])
		if intersectVar is not None:
			intersect[i] = intersectVar
		else:
			# no solution
			return (False, None)

	return (False, intersect)




# Find a sample point in hyperRectangle that has non-singular
# jacobian. If after a couple of tries, you are not successful,
# Check if a row of the jacobian interval is zero 
# (meaning that no matter however each voltage changes, the current
# represented by that row is always the same in the hyper). In this case, 
# if the current at a sample point is non-zero, then we are done and
# hyperrectangle does not contain a solution. 
# Since now we are dealing with a non-square matrix, let's also
# find the degenerate column (that can be expressed in terms of the
# other columns). Right now, let's just pick the column with all zeros
# (this means that that specific voltage does not have any effect on
# any of the currents). Either return (True, sample point). 
# Or (False, None) (if the current at sample point is non-zero)
# return (True, (degenRows, degenCols)) -> reduced problem. 
# IF we reduce the problem, solve Krawczyk with the reduced problem.
# In this case, the solution is not a point but a line or surface
# depending on the number of dimensions
def determineDegeneracy(model, startBounds):
	numV = startBounds.shape[0]
	numIter = 0
	zeroEpsilon = 1e-12

	jacInterval = model.jacobian(startBounds)
	#print ("jacInterval")
	#print (jacInterval)

	degenRows, degenCols = [], []
	for row in range(jacInterval.shape[0]):
		#if np.all(np.absolute(jacInterval[row,:,:]) < zeroEpsilon):
		if np.all(np.absolute(jacInterval[row,:,:]) == 0.0):
			degenRows.append(row)

	for col in range(jacInterval.shape[1]):
		#if np.all(np.absolute(jacInterval[:, col, :]) < zeroEpsilon):
		if np.all(np.absolute(jacInterval[:, col, :]) == 0.0):
			degenCols.append(col)

	'''print ("startBounds")
	for i in range(numV):
		print (startBounds[i,0], startBounds[i,1])
	print ("jacInterval")
	print (jacInterval)
	print ("degenRows", degenRows, "degenCols", degenCols)'''
	
	if len(degenRows) != len(degenCols):
		#print ("len(degenRows)", len(degenRows))
		raise Exception ('Number of degenrate rows ' + str(len(degenRows)) +' in jacobian interval ' + str(jacInterval)+ 
			' in startBounds ' + str(startBounds) + " does not equal number of degenerate columns " + str(len(degenCols)))

	if len(degenRows) == 0:
		return None

	if len(degenCols) > 1 :
		#print ("len(degenCols)", len(degenCols))
		raise Exception ('Number of degenerate cols (with all zeros) in jacobian interval ' + str(jacInterval)+ 
			' in startBounds ' + str(startBounds) + " is greater than 1: " + str(len(degenCols)))

	if len(degenRows) > 1 :
		#print ("len(degenCols)", len(degenCols))
		raise Exception ('Number of degenerate rows (with all zeros) in jacobian interval ' + str(jacInterval)+ 
			' in startBounds ' + str(startBounds) + " is greater than 1: " + str(len(degenRows)))

	
	samplePoint = (startBounds[:,0] + startBounds[:,1])/2.0
	#print ("samplePoint", samplePoint)
	fSamplePoint = model.f(samplePoint)
	#print ("fSamplePoint", fSamplePoint)
	if abs(fSamplePoint[degenRows[0]]) >= zeroEpsilon:
		return (False, None)

	return (True, (degenRows, degenCols))


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
		if(any([np.nextafter(funVal[i,0]*funVal[i,1], np.float("inf")) > np.nextafter(0.0, np.float("inf")) 
				for i in range(numV)])):
			return [False, None]

	constructBiggerHyper = False
	iteration = 0
	startBounds3d = np.copy(startBounds)
	prevIntersect = None
	while True:
		#print ("startBounds")
		#print (startBounds)
		samplePoint = (startBounds[:,0] + startBounds[:,1])/2.0
		fSamplePoint = np.array(model.f(samplePoint))
		jacSamplePoint = model.jacobian(samplePoint)
		jacInterval = model.jacobian(startBounds)

		#print ("jacInterval")
		#print (jacInterval)

		'''if(all([interval_r(startBounds[i]) < 1e-10 for i in range(startBounds.shape[0])])):
			print ("startBounds = " + str(startBounds))
			print ("jacInterval = " + str(jacInterval))
			print ("samplePoint = " + str(samplePoint))
			print ("fSamplePoint = " + str(fSamplePoint))
			print ("jacSamplePoint = " + str(jacSamplePoint))
			print ("")
			u, s, v = np.linalg.svd(jacSamplePoint, full_matrices=True)
			print ("svd of jacSamplePoint")
			print ("singular values = " + str(s))
			print ("u = " + str(u))
			print ("v = " + str(v))
			maxSingularValue = np.amax(s)
			print ("very low singular values/vectors (< 1e-8 * maxSingularValue)")
			for si in range(len(s)):
				if s[si] < 1e-8*maxSingularValue:
					print ("singular value " + str(s[si]))
					print ("singular left vector " + str(u[:,si]))	
			raise Exception('tiny')'''

		
		degenResult, degenRow, degenCol = None, None, None
		
		#try:
		degenResult = determineDegeneracy(model, startBounds)
		#except:
		#	pass

		#print ("degenResult", degenResult)
		if degenResult is not None:	
			#TODO: need to handle the case where jacSample point is singular
			pass
	

		#print ("jacInterval")
		#print (jacInterval)
		
		kHelpResult = krawczykHelp(startBounds, jacInterval, samplePoint, fSamplePoint, jacSamplePoint)
		
		#print ("kHelpResult")
		#print (kHelpResult)
		# Deal with reduced dimension after getting the result if we arrive at that situation

		if kHelpResult[0] or kHelpResult[1] is None:
			if kHelpResult[0]:
				return [True, kHelpResult[1], "0dim"]
			return kHelpResult
		
		intersect = kHelpResult[1]

		epsilonBounds = 1e-12


		oldVolume = volume(startBounds)
		newVolume = volume(intersect)
		volReduc = (oldVolume - newVolume)/(oldVolume*1.0)
		
		if (math.isnan(volReduc) or volReduc < alpha):
			# If intersect is tiny, use newton's method to find a solution
			# in the intersect and construct a bigger hyper than intersect
			# with the solution at the center. This is to take care of cases
			# when the solution is at the boundary
			#print ("Coming here?")
			if constructBiggerHyper == False:
				constructBiggerHyper = True
				exampleVolt = (intersect[:,0] + intersect[:,1])/2.0
				soln = newton(model,exampleVolt)
				#print ("soln ", soln)
				# the new hyper must contain the solution in the middle and enclose old hyper
				# and then we check for uniqueness of solution in the newer bigger hyperrectangle
				if soln[0]:
					prevIntersect = np.copy(intersect)
					for si in range(numV):
						maxDiff = max(abs(intersect[si,1] - soln[1][si]), abs(soln[1][si] - intersect[si,0]))
						if maxDiff < epsilonBounds:
							maxDiff = epsilonBounds
						intersect[si,0] = soln[1][si] - maxDiff
						intersect[si,1] = soln[1][si] + maxDiff

					print("bigger hyper " +str(intersect))
					startBounds = intersect
				else:
					return [False, intersect]

			else:
				intersect[:,0] = np.maximum(prevIntersect[:,0], intersect[:,0])
				intersect[:,1] = np.minimum(prevIntersect[:,1], intersect[:,1])
				return [False,intersect]
		else:
			startBounds = intersect

		iteration += 1

