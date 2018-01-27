from cvxopt import matrix,solvers
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import lpUtils
import intervalUtils

'''
takes in non-symbolic python values
calculates the tanhFun of val
'''
def tanhFun(a,val):
	return np.tanh(a*val)
	#return -(exp(a*val) - exp(-a*val))/(exp(a*val) + exp(-a*val))

'''
takes in non-symbolic python values
calculates the derivative of tanhFun of val
'''
def tanhFunder(a,val):
	den = np.cosh(a*val)*np.cosh(a*val)
	#print "den ", den
	return a/den


'''
when Vlow < 0 and Vhigh > 0, take the convex hull of 
two triangles (one formed on the left side and one on the
	right side)
'''
def triangleConvexHullBounds(a, Vin, Vout, Vlow, Vhigh):
	if (Vlow >= 0 and Vhigh >= 0) or (Vlow <= 0 and Vhigh <= 0):
		print "Vlow and Vhigh are in the same quadrants"
		return
	VZero = 0.0
	tanhFunVlow = tanhFun(a,Vlow)
	tanhFunVhigh = tanhFun(a,Vhigh)
	tanhFunVZero = tanhFun(a,VZero)
	dLow = tanhFunder(a,Vlow)
	dHigh = tanhFunder(a,Vhigh)
	dZero = tanhFunder(a,0)
	cLow = tanhFunVlow - dLow*Vlow
	cHigh = tanhFunVhigh - dHigh*Vhigh
	cZero = tanhFunVZero - dZero*0

	#print "dHigh ", dHigh, " cHigh ", cHigh

	if abs(dLow - dZero) < 1e-8:
		leftIntersectX = Vlow
		leftIntersectY = tanhFunVlow
	else:
		leftIntersectX = (cZero - cLow)/(dLow - dZero)
		leftIntersectY = dLow*leftIntersectX + cLow
	#print "dLow, ", dLow, "dZero", dZero
	#print "leftIntersectX ", leftIntersectX, " leftIntersectY ", leftIntersectY
	#print "Vlow ", Vlow, "Vhigh ", Vhigh
	if abs(dHigh - dZero) < 1e-8:
		rightIntersectX = Vhigh
		rightIntersectY = tanhFunVhigh
	else:
		rightIntersectX = (cZero - cHigh)/(dHigh - dZero)
		rightIntersectY = dHigh*rightIntersectX + cHigh

	overallConstraint = "1 " + Vin + " >= " + str(Vlow) + "\n"
	overallConstraint += "1 " + Vin + " <= " + str(Vhigh) + "\n"

	# Construct constraints from the convex hull of (Vlow, tanhFunVlow),
	# (leftIntersectX, leftIntersectY), (0,0), (rightIntersectX, rightIntersectY),
	# and (Vhigh, tanhFunVhigh)
	# Use jarvis algorithm from https://www.geeksforgeeks.org/convex-hull-set-1-jarviss-algorithm-or-wrapping/
	origPoints = [(Vlow, tanhFunVlow),(leftIntersectX, leftIntersectY),
				(0,0),(rightIntersectX, rightIntersectY), (Vhigh, tanhFunVhigh)]
	points = []
	for point in origPoints:
		if point not in points:
			points.append(point)
	#print "points"
	#print points
	leftMostIndex = 0
	convexHullIndices = []
	nextIndex = leftMostIndex
	iters = 0
	while(iters == 0 or nextIndex != leftMostIndex):
		convexHullIndices.append(nextIndex)
		otherIndex = (nextIndex + 1)%len(points)
		for i in range(len(points)):
			orientation = ((points[i][1] - points[nextIndex][1]) * (points[otherIndex][0] - points[i][0]) - 
				(points[i][0] - points[nextIndex][0]) * (points[otherIndex][1] - points[i][1]))
			if orientation < 0:
				otherIndex = i
		nextIndex = otherIndex
		iters += 1

	#print "convexHull", convexHullIndices
	for ci in range(len(convexHullIndices)):
		i = convexHullIndices[ci]
		ii = convexHullIndices[(ci + 1)%len(convexHullIndices)]
		grad = (points[ii][1] - points[i][1])/(points[ii][0] - points[i][0])
		c = points[i][1] - grad*points[i][0]
		if points[i] == (Vlow, tanhFunVlow) and points[ii] == (rightIntersectX, rightIntersectY):
			overallConstraint += "1 "+Vout + " + " +str(-grad) + " " + Vin + " >= "+str(c) + "\n"
		elif points[i] == (rightIntersectX, rightIntersectY) and points[ii] == (Vhigh, tanhFunVhigh):
			overallConstraint += "1 "+Vout + " + " +str(-grad) + " " + Vin + " >= "+str(c) + "\n"
		elif points[i] == (Vhigh, tanhFunVhigh) and points[ii] == (leftIntersectX, leftIntersectY):
			overallConstraint += "1 "+Vout + " + " +str(-grad) + " " + Vin + " <= "+str(c) + "\n"
		elif points[i] == (leftIntersectX, leftIntersectY) and points[ii] == (Vlow, tanhFunVlow):
			#print "grad", grad, "c", c
			#print "dLow", dLow, "cLow", cLow
			overallConstraint += "1 "+Vout + " + " +str(-grad) + " " + Vin + " <= "+str(c) + "\n"

		elif points[i] == (Vhigh, tanhFunVhigh) and points[ii] == (Vlow, tanhFunVlow):
			overallConstraint += "1 "+Vout + " + " +str(-grad) + " " + Vin + " <= "+str(c) + "\n"

		elif points[i] == (Vlow, tanhFunVlow) and points[ii] == (Vhigh, tanhFunVhigh):
			#print "coming here?"
			overallConstraint += "1 "+Vout + " + " +str(-grad) + " " + Vin + " >= "+str(c) + "\n"

	#print "overallConstraint", overallConstraint
	return overallConstraint


def triangleBounds(a, Vin, Vout, Vlow, Vhigh):
	tanhFunVlow = tanhFun(a,Vlow)
	tanhFunVhigh = tanhFun(a,Vhigh)
	dLow = tanhFunder(a,Vlow)
	dHigh = tanhFunder(a,Vhigh)
	diff = Vhigh - Vlow
	if(diff == 0):
		diff = 1e-10
	dThird = (tanhFunVhigh - tanhFunVlow)/diff
	cLow = tanhFunVlow - dLow*Vlow
	cHigh = tanhFunVhigh - dHigh*Vhigh
	cThird = tanhFunVlow - dThird*Vlow

	overallConstraint = "1 " + Vin + " >= " + str(Vlow) + "\n"
	overallConstraint += "1 " + Vin + " <= " + str(Vhigh) + "\n"

	#print "dLow ", dLow, "dHigh ", dHigh, "dThird ", dThird
	#print "cLow ", cLow, "cHigh ", cHigh, "cThird ", 

	#print "a ", a, " Vlow ", Vlow, " Vhigh ", Vhigh

	if a > 0:
		if Vlow >= 0 and Vhigh >=0:
			return overallConstraint + "1 "+ Vout + " + " +str(-dThird) + " " + Vin + " >= "+str(cThird)+"\n" +\
			"1 "+Vout + " + " +str(-dLow) + " " + Vin + " <= "+str(cLow)+"\n" +\
			"1 "+Vout + " + " +str(-dHigh) + " " + Vin + " <= "+str(cHigh) + "\n"

		elif Vlow <=0 and Vhigh <=0:
			return overallConstraint + "1 "+ Vout + " + " +str(-dThird) + " " + Vin + " <= "+str(cThird)+"\n" +\
			"1 "+Vout + " + " +str(-dLow) + " " + Vin + " >= "+str(cLow)+"\n" +\
			"1 "+Vout + " + " +str(-dHigh) + " " + Vin + " >= "+str(cHigh) + "\n"

	elif a < 0:
		if Vlow <= 0 and Vhigh <=0:
			return overallConstraint + "1 "+Vout + " + " +str(-dThird) + " " + Vin + " >= "+str(cThird)+"\n" +\
			"1 "+Vout + " + " +str(-dLow) + " " + Vin + " <= "+str(cLow)+"\n" +\
			"1 "+Vout + " + " +str(-dHigh) + " " + Vin + " <= "+str(cHigh) + "\n"
		elif Vlow >=0 and Vhigh >=0:
			return overallConstraint + "1 "+Vout + " + " +str(-dThird) + " " + Vin + " <= "+str(cThird)+"\n" +\
			"1 "+Vout + " + " +str(-dLow) + " " + Vin + " >= "+str(cLow)+"\n" +\
			"1 "+Vout + " + " +str(-dHigh) + " " + Vin + " >= "+str(cHigh) + "\n"
			
def oscNum(V,a,g_cc,g_fwd = 1):
	lenV = len(V)
	Vin = [V[i % lenV] for i in range(-1,lenV-1)]
	Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
	VoutFwd = [tanhFun(a,Vin[i]) for i in range(lenV)]
	VoutCc = [tanhFun(a,Vcc[i]) for i in range(lenV)]
	return (VoutFwd, VoutCc, [((tanhFun(a,Vin[i])-V[i])*g_fwd
			+(tanhFun(a,Vcc[i])-V[i])*g_cc) for i in range(lenV)])


'''Get jacobian of rambus oscillator at V
'''
def getJacobian(V,a,g_cc,g_fwd = 1):
	lenV = len(V)
	Vin = [V[i % lenV] for i in range(-1,lenV-1)]
	Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
	jacobian = np.zeros((lenV, lenV))
	for i in range(lenV):
		jacobian[i,i] = -(g_fwd+g_cc)
		jacobian[i,(i-1)%lenV] = g_fwd*tanhFunder(a,V[(i-1)%lenV])
		jacobian[i,(i + lenV/2) % lenV] = g_cc*tanhFunder(a,V[(i + lenV/2) % lenV])

	return jacobian

def getJacobianInterval(bounds,a,g_cc,g_fwd=1):
	lowerBound = bounds[:,0]
	upperBound = bounds[:,1]
	lenV = len(lowerBound)
	jacobian = np.zeros((lenV, lenV,2))
	jacobian[:,:,0] = jacobian[:,:,0] 
	jacobian[:,:,1] = jacobian[:,:,1]
	zerofwd =  g_fwd*tanhFunder(a,0)
	zerocc = g_cc*tanhFunder(a,0)
	for i in range(lenV):
		jacobian[i,i,0] = -(g_fwd+g_cc)
		jacobian[i,i,1] = -(g_fwd+g_cc)
		gfwdVal1 = g_fwd*tanhFunder(a,lowerBound[(i-1)%lenV])
		gfwdVal2 = g_fwd*tanhFunder(a,upperBound[(i-1)%lenV])
		if lowerBound[(i-1)%lenV] < 0 and upperBound[(i-1)%lenV] > 0:
			jacobian[i,(i-1)%lenV,0] = min(gfwdVal1,gfwdVal2,zerofwd)
			jacobian[i,(i-1)%lenV,1] = max(gfwdVal1,gfwdVal2,zerofwd)
		else:
			jacobian[i,(i-1)%lenV,0] = min(gfwdVal1,gfwdVal2)
			jacobian[i,(i-1)%lenV,1] = max(gfwdVal1,gfwdVal2)
		gccVal1 = g_cc*tanhFunder(a,lowerBound[(i + lenV/2) % lenV])
		gccVal2 = g_cc*tanhFunder(a,upperBound[(i + lenV/2) % lenV])
		if lowerBound[(i + lenV/2) % lenV] < 0 and upperBound[(i + lenV/2) % lenV] > 0:
			jacobian[i,(i + lenV/2) % lenV,0] = min(gccVal1,gccVal2,zerocc)
			jacobian[i,(i + lenV/2) % lenV,1] = max(gccVal1,gccVal2,zerocc)
		else:
			jacobian[i,(i + lenV/2) % lenV,0] = min(gccVal1,gccVal2)
			jacobian[i,(i + lenV/2) % lenV,1] = max(gccVal1,gccVal2)
	return jacobian


def ifFeasibleOrdering(a,params,xs,ys,zs,ordering,boundMap, hyperBound):
	lenV = len(xs)
	#print "ordering ", ordering
	hyperRectangle = np.zeros((lenV,2))
	for i in range(lenV):
		if(ordering[i] is not None):
			hyperRectangle[i][0] = boundMap[i][ordering[i]][0]
			hyperRectangle[i][1] = boundMap[i][ordering[i]][1]
		else:
			hyperRectangle[i][0] = boundMap[i][0][0]
			hyperRectangle[i][1] = boundMap[i][1][1]
	return ifFeasibleHyper(a,params,xs,ys,zs,hyperRectangle,hyperBound)



'''
Check if a particular ordering of intervalIndices produces a feasible
solution.
@param a parameter of the non-linear function
@param params parameters of the non-linear function
@param xs input variables in the non-linear function
@param ys output variables in the non-linear function
@param zs output variables in the non-linear function
@param ordering a particular ordering of interval indices.
@param boundMap dictionary from interval indices to actual bounds that 
	the values can take. For example, if we have a bound map of
	{0:[-1.0, -0.1], 1:[0.1,1.0]}, this means that the combinationNodes will
	represent arrays containing values of 0 and 1. If the length of xs is 4,
	then a possible ordering of the interval indices maybe [0,1,0,1] means that
	xs[0] must be in the range [-1.0,0.1], xs[1] must be in the range [0.1,1.0]
	and so on.
'''
def ifFeasibleHyper(a,params,xs,ys,zs,hyperRectangle, hyperBound):
	solvers.options["show_progress"] = False
	lenV = len(xs)
	print "hyperRectangle"
	print hyperRectangle
	iterNum = 0
	while True:
		# TODO: replace this with more precise numerical methods
		'''for i in range(lenV):
			if hyperRectangle[i][0] < 0 and abs(hyperRectangle[i][0]) < 1e-7:
				hyperRectangle[i][0] = -hyperBound
			if hyperRectangle[i][0] > 0 and abs(hyperRectangle[i][0]) < 1e-7:
				hyperRectangle[i][0] = 0.0
			if hyperRectangle[i][1] < 0 and abs(hyperRectangle[i][1]) < 1e-7:
				hyperRectangle[i][1] = 0.0
			if hyperRectangle[i][1] > 0 and abs(hyperRectangle[i][1]) < 1e-7:
				hyperRectangle[i][1] = hyperBound

		print "hyperRectangle "
		print hyperRectangle'''
		kResult = intervalUtils.checkExistenceOfSolutionGS(a,params[0],params[1],hyperRectangle.transpose(), oscNum, getJacobian, getJacobianInterval)
		if kResult[0]:
			#print "LP feasible ", newHyperRectangle
			return (True, kResult[1])

		if kResult[0] == False and kResult[1] is None:
			print "K operator not feasible"
			return (False, None)
		#print "kResult"
		#print kResult
		allConstraints = ""
		for i in range(lenV):
			fwdInd = (i-1)%lenV
			ccInd = (i+lenV/2)%lenV
			#print "fwdInd ", fwdInd, " ccInd ", ccInd
			#print "hyperRectangle[fwdInd][0]", hyperRectangle[fwdInd][0], "hyperRectangle[fwdInd][1]", hyperRectangle[fwdInd][1]
			triangleClaimFwd = ""
			if hyperRectangle[fwdInd,0] < 0 and hyperRectangle[fwdInd,1] > 0:
				triangleClaimFwd += triangleConvexHullBounds(a,xs[fwdInd],ys[i],hyperRectangle[fwdInd,0],hyperRectangle[fwdInd,1])
			else:
				triangleClaimFwd += triangleBounds(a,xs[fwdInd],ys[i],hyperRectangle[fwdInd,0],hyperRectangle[fwdInd,1])
			allConstraints += triangleClaimFwd

			triangleClaimCc = ""
			if hyperRectangle[ccInd,0] < 0 and hyperRectangle[ccInd,1] > 0:
				triangleClaimCc += triangleConvexHullBounds(a,xs[ccInd],zs[i],hyperRectangle[ccInd,0],hyperRectangle[ccInd,1])
			else:
				triangleClaimCc += triangleBounds(a,xs[ccInd],zs[i],hyperRectangle[ccInd,0],hyperRectangle[ccInd,1])
			allConstraints += triangleClaimCc
				
			allConstraints += str(params[0]) + " " + ys[i] + " + " + str(-params[0]-params[1]) + \
			" " + xs[i] + " + " + str(params[1]) + " "  + zs[i] + " >= 0.0\n"
			allConstraints += str(params[0]) + " " + ys[i] + " + " + str(-params[0]-params[1]) + \
			" " + xs[i] + " + " + str(params[1]) + " "  + zs[i] + " <= 0.0\n"

		'''allConstraintList = allConstraints.splitlines()
		allConstraints = ""
		for i in range(len(allConstraintList)):
			allConstraints += allConstraintList[i] + "\n"
		print "numConstraints ", len(allConstraintList)
		print "allConstraints"
		print allConstraints'''
		variableDict, A, B = lpUtils.constructCoeffMatrices(allConstraints)
		newHyperRectangle = copy.deepcopy(hyperRectangle)
		feasible = True
		for i in range(lenV):
			#print "min max ", i
			minObjConstraint = "min 1 " + xs[i]
			maxObjConstraint = "max 1 " + xs[i]
			Cmin = lpUtils.constructObjMatrix(minObjConstraint,variableDict)
			Cmax = lpUtils.constructObjMatrix(maxObjConstraint,variableDict)
			minSol = solvers.lp(Cmin,A,B)
			maxSol = solvers.lp(Cmax,A,B)
			if (minSol["status"] == "primal infeasible" or minSol["status"] == "dual infeasible")  and (maxSol["status"] == "primal infeasible" or maxSol["status"] == "dual infeasible"):
				feasible = False
				break
			else:
				if minSol["status"] == "optimal":
					newHyperRectangle[i,0] = minSol['x'][variableDict[xs[i]]]
				if maxSol["status"] == "optimal":
					newHyperRectangle[i,1] = maxSol['x'][variableDict[xs[i]]]
			
			#if infeasible
			if (minSol["status"] == "primal infeasible" or maxSol["status"] == "primal infeasible"):
				feasible = False
				break
			else:
				# if optimal
				if minSol["status"] == "optimal":
					newHyperRectangle[i,0] = minSol['x'][variableDict[xs[i]]] - 1e-5
				if maxSol["status"] == "optimal":
					newHyperRectangle[i,1] = maxSol['x'][variableDict[xs[i]]] + 1e-5

		print "newHyperRectangle ", newHyperRectangle
		if feasible == False:
			print "LP not feasible"
			return (False, None)

		if np.less_equal(newHyperRectangle[:,1] - newHyperRectangle[:,0],hyperBound*np.ones((lenV))).all() or np.less_equal(np.absolute(newHyperRectangle - hyperRectangle),1e-4*np.ones((lenV,2))).all():
			'''for i in range(lenV):
				# because this might be possible
				if(newHyperRectangle[i,1] < newHyperRectangle[i,0]):
					newHyperRectangle[i,0] = hyperRectangle[i,0]
					newHyperRectangle[i,1] = hyperRectangle[i,1]
			kResult = intervalUtils.checkExistenceOfSolutionGS(a,params[0],params[1],newHyperRectangle.transpose(), oscNum, getJacobian, getJacobianInterval)
			if kResult[0]:
				#print "LP feasible ", newHyperRectangle
				return (True, kResult[1])'''
			if kResult[0] == False and kResult[1] is not None:
				return (False, kResult[1])
			'''else:
				print "K operator not feasible"
				return (False, None)'''
		hyperRectangle = newHyperRectangle
		iterNum+=1
	

'''
Find the leaves of the rootCombinationNode are feasible according
to the broblem that we are trying to solve(in this case two-stage oscillator) 
recursively.
Note in this case we do not need to check though all possible leaves.
If a parent combinationNode is infeasible, the children will also be
infeasible. 
@param rootCombinationNode parent combinationNode
@param a parameter of two stage oscillator
@param params parameters of two stage oscillators
@param xs variable names indicating input voltages
@param ys variable names indicating forward output voltages
@param zs variable names indicating cross-coupled output voltages
@param boundMap dictionary from interval indices to actual bounds that 
	the values can take. For example, if we have a bound map of
	{0:[-1.0, -0.1], 1:[0.1,1.0]}, this means that the combinationNodes will
	represent arrays containing values of 0 and 1. If the length of xs is 4,
	then a combinationNode representing an array of [0,1,0,1] means that
	xs[0] must be in the range [-1.0,0.1], xs[1] must be in the range [0.1,1.0]
	and so on.
@param outValidIntervalIndices a list of interval indices for which
	the problem has a feasible solution. For example if [0, 1, 0, 1] is 
	a valid interval array, according to the bound map of {0:[-1.0, -0.1], 1:[0.1,1.0]}
	For -1.0 <= xs[0] <= -0.1 and 0.1 <= xs[1] <= 1.0 and so on, the problem
	has a feasible solution 
'''
def getFeasibleIntervalIndices(rootCombinationNode,a,params,xs,ys,zs,boundMap,hyperBound, excludingBound,refinedHypers):
	intervalIndices = rootCombinationNode.rootArray
	print "intervalIndices", intervalIndices
	print "boundMap", boundMap
	lenV = len(intervalIndices)
	
	feasibility = ifFeasibleOrdering(a,params,xs,ys,zs,intervalIndices,boundMap,hyperBound)
	print "feasibility"
	print feasibility
	if feasibility[0]:
		refinedHypers.append(feasibility[1])
		return

	# Generate a conflict interval array if constraints are infeasible
	if feasibility[0] == False and feasibility[1] is None:
		return

	newBoundMap = copy.deepcopy(boundMap)
	hyperRectangle = feasibility[1]
	for i in range(lenV):
		if intervalIndices[i] is None:
			lowBound = hyperRectangle[i][0]
			upperBound = hyperRectangle[i][1]
			newBoundMap[i][0][0] = lowBound
			newBoundMap[i][1][1] = upperBound
			if lowBound < 0 and upperBound > 0 and upperBound > excludingBound:
				newBoundMap[i][0][1] = excludingBound
				newBoundMap[i][1][0] = excludingBound
			else:
				newBoundMap[i][0][1] = (lowBound + upperBound)/2.0
				newBoundMap[i][1][0] = (lowBound + upperBound)/2.0

	indexOfNone = None
	for i in range(len(intervalIndices)):
		if intervalIndices[i] is None:
			indexOfNone = i
			break
	print "indexOfNone ", indexOfNone
	if indexOfNone is None:
		bisectionHypers = refineHyper(a, params, xs, ys, zs, intervalIndices, newBoundMap, hyperBound)
		print "bisectionHypers"
		print bisectionHypers
		print "len(refinedHypers) before ", len(refinedHypers)
		for hyper in bisectionHypers:
			refinedHypers.append(hyper)
		print "len(refinedHypers) after ", len(refinedHypers)
	for i in range(len(rootCombinationNode.children)):
		getFeasibleIntervalIndices(rootCombinationNode.children[i],a,params,xs,ys,zs,newBoundMap,hyperBound,excludingBound,refinedHypers)

def refineHyper(a, params, xs, ys, zs, ordering, boundMap, maxHyperBound):
	lenV = len(xs)
	hyperRectangle = np.zeros((lenV,2))
	excludingRegConstraint = ""
	for i in range(lenV):
		hyperRectangle[i][0] = boundMap[i][ordering[i]][0]
		hyperRectangle[i][1] = boundMap[i][ordering[i]][1]
	finalHyper = []
	count = 0
	volumes = []

	print "before bisecting num ", len(finalHyper)
	bisectHyper(a,params,xs,ys,zs,maxHyperBound,hyperRectangle,0,finalHyper)
	print "after bisecting num ", len(finalHyper)

	return finalHyper

def bisectHyper(a,params,xs,ys,zs,hyperBound,hyperRectangle,bisectingIndex, finalHypers):
	#print "hyperRectangle"
	#print hyperRectangle
	lenV = hyperRectangle.shape[0]
	intervalLength = hyperRectangle[:,1] - hyperRectangle[:,0]
	bisectingIndex = np.argmax(intervalLength)
	'''if bisectingIndex >= lenV:
		bisectingIndex = 0'''
	leftHyper = copy.deepcopy(hyperRectangle)
	rightHyper = copy.deepcopy(hyperRectangle)
	midVal = (hyperRectangle[bisectingIndex][0] + hyperRectangle[bisectingIndex][1])/2.0
	leftHyper[bisectingIndex][1] = midVal
	rightHyper[bisectingIndex][0] = midVal
	print "leftHyper"
	print leftHyper
	print "rightHyper"
	print rightHyper
	feasLeft = ifFeasibleHyper(a,params,xs,ys,zs,leftHyper, hyperBound)
	feasRight = ifFeasibleHyper(a,params,xs,ys,zs,rightHyper, hyperBound)
	print "feasLeft"
	print feasLeft
	print "feasRight"
	print feasRight

	if feasLeft[0]:
		finalHypers.append(feasLeft[1])
	if feasLeft[0] == False and feasLeft[1] is not None:
		bisectHyper(a,params,xs,ys,zs,hyperBound,leftHyper,bisectingIndex+1,finalHypers)

	
	if feasRight[0]:
		finalHypers.append(feasRight[1])
	if feasRight[0] == False and feasRight[1] is not None:
		bisectHyper(a,params,xs,ys,zs,hyperBound,rightHyper,bisectingIndex+1,finalHypers)

def findExcludingBound(a,params,xs,ys,zs,ordering,boundMap, maxDiff = 0.2):
	lenV = len(xs)
	hyperRectangle = np.zeros((lenV,2))
	hyperBound = 0.001
	for i in range(lenV):
		hyperRectangle[i][0] = boundMap[i][ordering[i]][0]
		hyperRectangle[i][1] = boundMap[i][ordering[i]][1]
	soln = hyperRectangle[:,0] + (hyperRectangle[:,1] - hyperRectangle[:,0])*0.75;
	soln = intervalUtils.newton(a,params,soln,oscNum,getJacobian)
	diff = maxDiff
	while True:
		hyperRectangle[:,0] = soln - diff
		hyperRectangle[:,1] = soln + diff
		kResult = intervalUtils.checkExistenceOfSolutionGS(a,params[0],params[1],hyperRectangle.transpose(), oscNum, getJacobian, getJacobianInterval)
		if kResult[0] == False and kResult[1] is not None:
			diff = diff/2.0;
		else:
			break
	for i in range(lenV):
		boundMap[i][0][1] = diff
		boundMap[i][1][0] = diff
	return diff

'''
Return true if stable and false otherwise
'''
def determineStability(a,params,equilibrium):
	jac = getJacobian(equilibrium,a,params[1],params[0])
	eigVals,_ = np.linalg.eig(jac)
	maxEig = np.amax(eigVals.real)
	if maxEig > 0:
		return False
	return True

def rambusOscillator(a, numStages):
	startExp = time.time()
	params = [1.0,4.0]
	lenV = numStages*2
	xs = []
	ys = []
	zs = []
	exampleOrdering = []
	indexChoiceArray = []
	firstIndex = numStages - 1
	secondIndex = numStages*2 - 1
	for i in range(lenV):
		xs.append("x" + str(i))
		ys.append("y" + str(i))
		zs.append("z" + str(i))
		exampleOrdering.append(0)
		#indexChoiceArray.append(i)
		if i%2 == 0:
			indexChoiceArray.append(firstIndex)
			firstIndex -= 1
		else:
			indexChoiceArray.append(secondIndex)
			secondIndex -= 1
	
	print "indexChoiceArray", indexChoiceArray
	boundMap = []
	for i in range(lenV):
		boundMap.append({0:[-1.0,0.0],1:[0.0,1.0]})
	excludingBound = findExcludingBound(a,params,xs,ys,zs,exampleOrdering,boundMap)
	print "boundMap ", boundMap
	minBoundMap = 0
	maxBoundMap = 1
	rootCombinationNodes = intervalUtils.combinationWithTrees(lenV,[minBoundMap,maxBoundMap],indexChoiceArray)
	hyperBound = excludingBound
	'''hyperRectangle = np.zeros((lenV,2))
	hyperRectangle[0,:] = [-0.99983621, 0.99845864]
	hyperRectangle[1,:] = [-0.99983605, 0.2       ]
	hyperRectangle[2,:] = [-0.95224507, 0.9998362 ]
	hyperRectangle[3,:] = [-0.4754362, 0.99979469]

	feasibility = ifFeasibleHyper(a,params,xs,ys,zs,hyperRectangle, hyperBound)
	print feasibility'''
	'''#exampleSoln = (hyperRectangle[:,0] + hyperRectangle[:,1])/2.0
	exampleSoln = np.array([-0.86730826,  0.99985882,
 -0.99990911, -0.07034628, 0.86730826, -0.99985882,  0.99990911, 0.07034628])
	finalSoln = intervalUtils.newton(a,params,exampleSoln, oscNum, getJacobian)
	print "finalSoln", finalSoln'''
	'''ordering = [0,1,1,0,1,0,0,1]
	hypers = refineHyper(a, params, xs, ys, zs, ordering, boundMap, hyperBound)
	print "hypers"
	print hypers
	exampleSoln = (hypers[0][:,0] + hypers[0][:,1])/2.0
	finalSoln = intervalUtils.newton(a,params,exampleSoln, oscNum, getJacobian)
	print "finalSoln ", finalSoln'''
	allHypers = []
	for i in range(len(rootCombinationNodes)):
		getFeasibleIntervalIndices(rootCombinationNodes[i],a,params,xs,ys,zs,boundMap,hyperBound, excludingBound,allHypers)
	
	print "allHypers"
	print allHypers
	sampleSols = []
	rotatedSols = {}
	stableSols = []
	unstableSols = []
	allSols = []
	for hyper in allHypers:
		exampleSoln = (hyper[:,0] + hyper[:,1])/2.0
		finalSoln = intervalUtils.newton(a,params,exampleSoln, oscNum, getJacobian)
		#print "exampleSoln ", exampleSoln
		#print "finalSoln ", finalSoln
		stable = determineStability(a,params,finalSoln)
		if stable:
			stableSols.append(finalSoln)
		else:
			unstableSols.append(finalSoln)
		allSols.append(finalSoln)
		
		# Classify the solutions into equivalence classes
		if len(sampleSols) == 0:
			sampleSols.append(finalSoln)
			rotatedSols[0] = []
		else:
			foundSample = False
			for si in range(len(sampleSols)):
				sample = sampleSols[si]
				for ii in range(lenV):
					if abs(finalSoln[0] - sample[ii]) < 1e-8:
						rotatedSample = np.zeros_like(finalSoln)
						for ri in range(lenV):
							rotatedSample[ri] = sample[(ii+ri)%lenV]
						if np.less_equal(np.absolute(rotatedSample - finalSoln), np.ones((lenV))*1e-8 ).all():
							foundSample = True
							rotatedSols[si].append(ii)
							break
				if foundSample:
					break

			if foundSample == False:
				sampleSols.append(finalSoln)
				rotatedSols[len(sampleSols)-1] = []

	for hi in range(len(sampleSols)):
		print "equivalence class# ", hi
		print "main member ", sampleSols[hi]
		print "number of other members ", len(rotatedSols[hi])
		print "other member rotationIndices: "
		for mi in range(len(rotatedSols[hi])):
			print rotatedSols[hi][mi]
		print ""

	print "numSolutions, ", len(allHypers)
	print "num stable solutions ", len(stableSols)
	'''for si in range(len(stableSols)):
		print stableSols[si]'''
	#print "num unstable solutions ", len(unstableSols)
	'''for si in range(len(unstableSols)):
		print unstableSols[si]'''
	endExp = time.time()
	print "TOTAL TIME ", endExp - startExp


'''A = matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ])
b = matrix([ 1.0, -2.0, 0.0, 4.0 ])
c = matrix([ 2.0, 1.0 ])
print "size of A ", A.size
print "size of b ", b.size
print "size of c ", c.size
sol=solvers.lp(c,A,b)
print(sol['x'])
#print(triangleBounds(-5.0,"x0","y0",0.0,0.5))
objConstraint = "min 1 y0 + 1 x1 \n"
triangleConstraint1  = triangleBounds(-5.0,"x0","y0",0.0,0.5)
triangleConstraint2 = triangleBounds(-5.0,"x1","y1",0.0,0.5)
print objConstraint + triangleConstraint1 + triangleConstraint2
variableDict, A, B = constructCoeffMatrices(triangleConstraint1 + triangleConstraint2)
C = constructObjMatrix(objConstraint,variableDict)'''
rambusOscillator(-5.0,4)

