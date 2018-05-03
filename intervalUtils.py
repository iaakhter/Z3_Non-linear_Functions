import numpy as np
import copy
import random


def multiplyRegularMatWithIntervalMat(regMat,intervalMat):
	result = np.zeros((regMat.shape[0],intervalMat.shape[1],2))
	for i in range(regMat.shape[0]):
		for j in range(intervalMat.shape[1]):
			intervalVal = np.zeros(2)
			for k in range(intervalMat.shape[1]):
				multMin = min(regMat[i,k]*intervalMat[k,j,0], regMat[i,k]*intervalMat[k,j,1])
				multMax = max(regMat[i,k]*intervalMat[k,j,0], regMat[i,k]*intervalMat[k,j,1])
				intervalVal[0] += multMin
				intervalVal[1] += multMax
			result[i,j,0] = intervalVal[0]
			result[i,j,1] = intervalVal[1]

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
			multMin = min(mat[i,j,0]*vec[j,0], mat[i,j,0]*vec[j,1], mat[i,j,1]*vec[j,0], mat[i,j,1]*vec[j,1])
			multMax = max(mat[i,j,0]*vec[j,0], mat[i,j,0]*vec[j,1], mat[i,j,1]*vec[j,0], mat[i,j,1]*vec[j,1])
			intervalVal[0] += multMin
			intervalVal[1] += multMax
		result[i,0] = intervalVal[0]
		result[i,1] = intervalVal[1]
	return result

def newton(model,soln):
	h = soln
	count = 0
	maxIter = 100
	boundMap = model.boundMap
	lenV = len(soln)
	'''overallHyper = np.zeros((lenV,2))
	for i in range(lenV):
		overallHyper[i,0] = boundMap[i][0][0]
		overallHyper[i,1] = boundMap[i][1][1]'''
	#print ("overallHyper", overallHyper)
	while count < maxIter and (np.linalg.norm(h) > 1e-8 or count == 0) :
		#print ("soln", soln)
		#print ("h", h)
		_,_,res = model.oscNum(soln)
		#print ("res", res)
		res = -np.array(res)
		jac = model.jacobian(soln)
		#print ("res", res)
		#print ("jac", jac)
		try:
			h = np.linalg.solve(jac,res)
		except np.linalg.LinAlgError:
			h = np.linalg.lstsq(jac, res)[0]
		soln = soln + h
		#print ("new soln", soln)
		'''if np.less(soln, overallHyper[:,0]).any() or np.greater(soln, overallHyper[:,1]).any():
			return (False, soln)'''
		count+=1
	if count >= maxIter and np.linalg.norm(h) > 1e-8:
		return(False, soln)
	return (True,soln)

'''
Check existence of solution within a certain hyperRectangle
using the Gauss-Siedel operator
'''
def checkExistenceOfSolutionGS(model,hyperRectangle):
	numVolts = len(hyperRectangle[0])
	#print ("start K operator", hyperRectangle)
	startBounds = np.zeros((numVolts,2))
	startBounds[:,0] = hyperRectangle[0,:]
	startBounds[:,1] = hyperRectangle[1,:]
	#print "startBounds ", startBounds
	iteration = 0
	constructBiggerHyper = False
	while True:
		#print ("iteration number: ", iteration)
		#print ("startBounds ", startBounds)
		midPoint = (startBounds[:,0] + startBounds[:,1])/2.0
		#midPoint = startBounds[:,0] + (startBounds[:,1] - startBounds[:,0])*0.25
		#print ("midPoint")
		#print (midPoint)
		_,_,IMidPoint = np.array(model.oscNum(midPoint))
		jacMidPoint = model.jacobian(midPoint)
		#print ("jacMidPoint")
		#print (jacMidPoint)
		C = None
		numIterations = 0

		while True:
			fail = False
			try:
				#print ("midPoint", midPoint)
				#print ("jacobian in K", jacMidPoint)
				C = np.linalg.pinv(jacMidPoint)
				#print ("C", C)
			except np.linalg.linalg.LinAlgError:
				fail = True
				randomVal = random.uniform(0.1,0.9)
				midPoint = startBounds[:,0] + (startBounds[:,1] - startBounds[:,0])*randomVal
				#print ("newMidPoint", midPoint)
				_,_,IMidPoint = np.array(model.oscNum(midPoint))
				jacMidPoint = model.jacobian(midPoint)

			if not(fail):
				break
			numIterations += 1
			'''if numIterations == 200:
				return'''
			#return

		#print "C"
		#print C
		#print "condition number of C", np.linalg.cond(C)

		#print "C ", C
		#C = np.identity(numVolts)
		I = np.identity(numVolts)

		jacInterval = model.jacobianInterval(startBounds)
		#print "jacInterval"
		#print jacInterval
		#print "IMidPoint"
		#print IMidPoint
		C_IMidPoint = np.dot(C,IMidPoint)
		#print "C_IMidPoint", C_IMidPoint

		C_jacInterval = multiplyRegularMatWithIntervalMat(C,jacInterval)

		I_minus_C_jacInterval = subtractIntervalMatFromRegularMat(I,C_jacInterval)

		gsInterval = np.zeros((numVolts,2))
		gsIntersect = np.copy(startBounds)
		for i in range(numVolts):
			sumTerm = np.zeros((2))
			for j in range(numVolts):
				subTerm1 = gsIntersect[j,0] - midPoint[j]
				subTerm2 = gsIntersect[j,1] - midPoint[j]
				mult = np.zeros((2))
				mult1 = I_minus_C_jacInterval[i,j,0] * subTerm1
				mult2 = I_minus_C_jacInterval[i,j,0] * subTerm2
				mult3 = I_minus_C_jacInterval[i,j,1] * subTerm1
				mult4 = I_minus_C_jacInterval[i,j,1] * subTerm2
				mult[0] = min(mult1, mult2, mult3, mult4)
				mult[1] = max(mult1, mult2, mult3, mult4)
				sumTerm += mult
			C_ImidPoint_minus_sumTerm = np.zeros((2))
			C_ImidPoint_minus_sumTerm[0] = min(C_IMidPoint[i] - sumTerm[0], C_IMidPoint[i] - sumTerm[1])
			C_ImidPoint_minus_sumTerm[1] = max(C_IMidPoint[i] - sumTerm[0], C_IMidPoint[i] - sumTerm[1])
			#divTerm = np.zeros((2))
			#div1 = C_ImidPoint_minus_sumTerm[0]/C_jacInterval[i,i,0]
			#div2 = C_ImidPoint_minus_sumTerm[0]/C_jacInterval[i,i,1]
			#div3 = C_ImidPoint_minus_sumTerm[1]/C_jacInterval[i,i,0]
			#div4 = C_ImidPoint_minus_sumTerm[1]/C_jacInterval[i,i,1]
			#divTerm[0] = min(div1,div2,div3,div4)
			#divTerm[1] = max(div1,div2,div3,div4)
			#print "divTerm ", divTerm
			gsInterval[i][0] = min(midPoint[i] - C_ImidPoint_minus_sumTerm[0], midPoint[i] - C_ImidPoint_minus_sumTerm[1])
			gsInterval[i][1] = max(midPoint[i] - C_ImidPoint_minus_sumTerm[0], midPoint[i] - C_ImidPoint_minus_sumTerm[1])
			minVal = max(gsInterval[i][0],startBounds[i][0])
			maxVal = min(gsInterval[i][1],startBounds[i][1])
			if minVal <= maxVal and \
				minVal >= gsInterval[i][0] and minVal >= startBounds[i][0] and \
				minVal <= gsInterval[i][1] and minVal <= startBounds[i][1] and \
				maxVal >= gsInterval[i][0] and maxVal >= startBounds[i][0] and \
				maxVal <= gsInterval[i][1] and maxVal <= startBounds[i][1]:
				gsIntersect[i] = [minVal,maxVal]
			elif np.less_equal(np.absolute(gsInterval[i,:] - startBounds[i,:]),1e-8*np.ones((2))).all():
				gsIntersect[i] = startBounds[i]
			else:
				#print ("problem i ", i)
				#print ("gsInterval[i] ", gsInterval[i])
				#print (np.absolute(gsInterval[i,:] - startBounds[i,:]))
				#print (minVal <= maxVal)
				return (False, None)

		#print ("gsInterval ")
		#print (gsInterval)
		#print ("gsIntersect ")
		#print (gsIntersect)
		uniqueSoln = True
		for i in range(numVolts):
			if gsInterval[i][0] <= startBounds[i][0] or gsInterval[i][0] >= startBounds[i][1]:
				uniqueSoln = False
			if gsInterval[i][1] <= startBounds[i][0] or gsInterval[i][1] >= startBounds[i][1]:
				uniqueSoln = False

		#print (gsInterval[:,0] - startBounds[:,0])
		#print (startBounds[:,1] - gsInterval[:,1])
		if uniqueSoln:
			#print "Hyperrectangle with unique solution found"
			#print kInterval
			return (True,gsInterval)
		
		#print ("constructBiggerHyper before", constructBiggerHyper)
		if np.less_equal(gsIntersect[:,1] - gsIntersect[:,0],1e-8*np.ones((numVolts))).all() or  np.less_equal(np.absolute(gsIntersect - startBounds),1e-4*np.ones((numVolts,2))).all():
			if constructBiggerHyper == False and np.less_equal(gsIntersect[:,1] - gsIntersect[:,0],1e-8*np.ones((numVolts))).all():
				#print ("gsIntersect before")
				#print (gsIntersect)
				constructBiggerHyper = True
				exampleVolt = (gsIntersect[:,0] + gsIntersect[:,1])/2.0
				soln = newton(model,exampleVolt)
				#print ("soln ", soln)
				# the new hyper must contain the solution in the middle and enclose old hyper
				# and then we check for uniqueness of solution in the newer bigger hyperrectangle
				if soln[0]:
					for si in range(numVolts):
						maxDiff = max(abs(gsIntersect[si,1] - soln[1][si]), abs(soln[1][si] - gsIntersect[si,0]))
						#print ("maxDiff", maxDiff)
						if maxDiff < 1e-9:
							maxDiff = 1e-9
						#maxDiff = 0.01
						#print "maxDiff ", maxDiff
						gsIntersect[si,0] = soln[1][si] - maxDiff
						gsIntersect[si,1] = soln[1][si] + maxDiff

					#print ("bigger hyper ", gsIntersect)
					startBounds = gsIntersect
				#print ("after if constructBiggerHyper", constructBiggerHyper)
			else:
				return (False,gsIntersect)
		else:
			startBounds = gsIntersect
		iteration += 1

'''
Check existence of solution within a certain hyperRectangle
using the Krawczyk operator
'''
def checkExistenceOfSolution(model,hyperRectangle):
	numVolts = len(hyperRectangle[0])

	startBounds = np.zeros((numVolts,2))
	startBounds[:,0] = hyperRectangle[0,:]
	startBounds[:,1] = hyperRectangle[1,:]
	#print "startBounds ", startBounds
	constructBiggerHyper = False
	iteration = 0
	while True:
		#print "iteration number: ", iteration
		#print ("startBounds ", startBounds)
		midPoint = (startBounds[:,0] + startBounds[:,1])/2.0
		#midPoint = startBounds[:,0] + (startBounds[:,1] - startBounds[:,0])*0.25
		#print "midPoint"
		#print midPoint
		_,_,IMidPoint = np.array(model.oscNum(midPoint))
		jacMidPoint = model.jacobian(midPoint)
		#print "jacMidPoint"
		#print jacMidPoint
		C = np.linalg.pinv(jacMidPoint)
		#print "C"
		#print C
		#print "condition number of C", np.linalg.cond(C)

		#print "C ", C
		I = np.identity(numVolts)

		jacInterval = model.jacobianInterval(startBounds)
		#print "jacInterval"
		#print jacInterval
		#print "IMidPoint"
		#print IMidPoint
		C_IMidPoint = np.dot(C,IMidPoint)
		#print "C_IMidPoint", C_IMidPoint

		C_jacInterval = multiplyRegularMatWithIntervalMat(C,jacInterval)
		#print "C_jacInterval"
		#print C_jacInterval
		I_minus_C_jacInterval = subtractIntervalMatFromRegularMat(I,C_jacInterval)
		#print "I_minus_C_jacInterval"
		#print I_minus_C_jacInterval
		xi_minus_midPoint = np.zeros((numVolts,2))
		for i in range(numVolts):
			xi_minus_midPoint[i][0] = min(startBounds[i][0] - midPoint[i], startBounds[i][1] - midPoint[i])
			xi_minus_midPoint[i][1] = max(startBounds[i][0] - midPoint[i], startBounds[i][1] - midPoint[i])
		#print "xi_minus_midPoint", xi_minus_midPoint
		lastTerm = multiplyIntervalMatWithIntervalVec(I_minus_C_jacInterval, xi_minus_midPoint)
		#print "lastTerm "
		#print lastTerm

		kInterval1 = midPoint - C_IMidPoint + lastTerm[:,0]
		kInterval2 = midPoint - C_IMidPoint + lastTerm[:,1]
		kInterval = np.zeros((numVolts,2))
		#print "kInterval1 ", kInterval1, " kInterval2 ", kInterval2
		kInterval[:,0] = np.minimum(kInterval1, kInterval2)
		kInterval[:,1] = np.maximum(kInterval1, kInterval2)

		#print ("kInterval ")
		#print (kInterval)

		uniqueSoln = True
		for i in range(numVolts):
			if kInterval[i][0] <= startBounds[i][0] or kInterval[i][0] >= startBounds[i][1]:
				uniqueSoln = False
			if kInterval[i][1] <= startBounds[i][0] or kInterval[i][1] >= startBounds[i][1]:
				uniqueSoln = False

		if uniqueSoln:
			#print "Hyperrectangle with unique solution found"
			#print kInterval
			return (True,kInterval)

		intersect = np.zeros((numVolts,2))
		for i in range(numVolts):
			minVal = max(kInterval[i][0],startBounds[i][0])
			maxVal = min(kInterval[i][1],startBounds[i][1])
			if minVal <= maxVal and \
				minVal >= kInterval[i][0] and minVal >= startBounds[i][0] and \
				minVal <= kInterval[i][1] and minVal <= startBounds[i][1] and \
				maxVal >= kInterval[i][0] and maxVal >= startBounds[i][0] and \
				maxVal <= kInterval[i][1] and maxVal <= startBounds[i][1]:
				intersect[i] = [minVal,maxVal]
				intervalLength =  intersect[:,1] - intersect[:,0]
			else:
				#print "problem index ", i
				#print "kInterval[i]", kInterval[i][0], kInterval[i][1]
				#print "startBounds[i]", startBounds[i][0], startBounds[i][1]
				#print "minVal ", minVal, "maxVal",maxVal
				#print minVal <= maxVal
				intersect = None
				break

		#print "intersect"
		#print intersect

		if intersect is None:
			#print "hyperrectangle does not contain any solution"
			return (False,None)
		
		
		if np.less_equal(intersect[:,1] - intersect[:,0],1e-8*np.ones((numVolts))).all() or  np.less_equal(np.absolute(intersect - startBounds),1e-4*np.ones((numVolts,2))).all():
			if constructBiggerHyper == False and np.less_equal(intersect[:,1] - intersect[:,0],1e-8*np.ones((numVolts))).all():
				#print ("gsIntersect before")
				#print (gsIntersect)
				constructBiggerHyper = True
				exampleVolt = (intersect[:,0] + intersect[:,1])/2.0
				soln = newton(model,exampleVolt)
				#print ("soln ", soln)
				# the new hyper must contain the solution in the middle and enclose old hyper
				# and then we check for uniqueness of solution in the newer bigger hyperrectangle
				if soln[0]:
					for si in range(numVolts):
						maxDiff = max(abs(intersect[si,1] - soln[1][si]), abs(soln[1][si] - intersect[si,0]))
						#print ("maxDiff", maxDiff)
						if maxDiff < 1e-9:
							maxDiff = 1e-9
						#print "maxDiff ", maxDiff
						intersect[si,0] = soln[1][si] - maxDiff
						intersect[si,1] = soln[1][si] + maxDiff

					#print ("bigger hyper ", intersect)
					startBounds = intersect
				#print ("after if constructBiggerHyper", constructBiggerHyper)
			else:
				return (False,intersect)
		else:
			startBounds = intersect

		iteration += 1