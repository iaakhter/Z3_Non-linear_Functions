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
	tanhFunVlow = tanhFun(a,Vlow)
	tanhFunVhigh = tanhFun(a,Vhigh)
	tanhFunVZero = tanhFun(a,0)
	dLow = tanhFunder(a,Vlow)
	dHigh = tanhFunder(a,Vhigh)
	dZero = tanhFunder(a,0)
	cLow = tanhFunVlow - dLow*Vlow
	cHigh = tanhFunVhigh - dHigh*Vhigh
	cZero = tanhFunVZero - dZero*0

	#print "dHigh ", dHigh, " cHigh ", cHigh

	VNeg1 = -1.0
	VPos1 = 1.0
	tanhFunVNeg1 = 0.0
	tanhFunVPos1 = 0.0

	leftIntersectX = (cZero - cLow)/(dLow - dZero)
	leftIntersectY = dLow*leftIntersectX + cLow
	#print "leftIntersectX ", leftIntersectX, " leftIntersectY ", leftIntersectY
	#print "Vlow ", Vlow, "tanhFunVlow ", tanhFunVlow
	#print "Vhigh ", Vhigh, " tanhFunVhigh ", tanhFunVhigh
	rightIntersectX = (cZero - cHigh)/(dHigh - dZero)
	rightIntersectY = dHigh*rightIntersectX + cHigh
	#print "rightIntersectX ", rightIntersectX , " rightIntersectY ", rightIntersectY
	dFromLeftIntersect = (tanhFunVhigh - leftIntersectY)/(Vhigh - leftIntersectX)
	dFromRightIntersect = (rightIntersectY - tanhFunVlow)/(rightIntersectX - Vlow)
	cFromLeftIntersect = leftIntersectY - dFromLeftIntersect*leftIntersectX
	cFromRightIntersect = rightIntersectY - dFromRightIntersect*rightIntersectX

	overallConstraint = "1 " + Vin + " >= " + str(Vlow) + "\n"
	overallConstraint += "1 " + Vin + " <= " + str(Vhigh) + "\n"
	if a > 0:
		return overallConstraint + "1 "+Vout + " + " +str(-dFromLeftIntersect) + " " + Vin + " >= "+str(cFromLeftIntersect) + "\n"+\
			"1 "+Vout + " + " +str(-dFromRightIntersect) + " " + Vin + " <= "+str(cFromRightIntersect) + "\n"

	elif a < 0:
		'''return overallConstraint + "1 "+Vout + " + " +str(-dLow) + " " + Vin + " <= "+str(cLow) + "\n"+\
			"1 "+Vout + " + " +str(-dHigh) + " " + Vin + " >= "+str(cHigh) + "\n"+\
			"1 "+Vout + " + " +str(-dFromLeftIntersect) + " " + Vin + " <= "+str(cFromLeftIntersect) + "\n"+\
			"1 "+Vout + " + " +str(-dFromRightIntersect) + " " + Vin + " >= "+str(cFromRightIntersect) + "\n"'''

		#print "Vlow", Vlow, "Vhigh", Vhigh
		overallConstraint += "1 "+Vout + " + " +str(-dFromLeftIntersect) + " " + Vin + " <= "+str(cFromLeftIntersect) + "\n"+\
			"1 "+Vout + " + " +str(-dFromRightIntersect) + " " + Vin + " >= "+str(cFromRightIntersect) + "\n"
		#constr = overallConstraint + "1 "+Vout + " + " +str(-dFromRightIntersect) + " " + Vin + " >= "+str(cFromRightIntersect) + "\n"
		#print constr
		#overallConstraint += "1 " + Vout + " <= " + str(tanhFunVlow) + "\n"
		#overallConstraint += "1 " + Vout + " >= " + str(tanhFunVhigh) + "\n"
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
	for i in range(lenV):
		jacobian[i,i,0] = -(g_fwd+g_cc)
		jacobian[i,i,1] = -(g_fwd+g_cc)
		gfwdVal1 = g_fwd*tanhFunder(a,lowerBound[(i-1)%lenV])
		gfwdVal2 = g_fwd*tanhFunder(a,upperBound[(i-1)%lenV])
		jacobian[i,(i-1)%lenV,0] = min(gfwdVal1,gfwdVal2)
		jacobian[i,(i-1)%lenV,1] = max(gfwdVal1,gfwdVal2)
		gccVal1 = g_cc*tanhFunder(a,lowerBound[(i + lenV/2) % lenV])
		gccVal2 = g_cc*tanhFunder(a,upperBound[(i + lenV/2) % lenV])
		jacobian[i,(i + lenV/2) % lenV,0] = min(gccVal1,gccVal2)
		jacobian[i,(i + lenV/2) % lenV,1] = max(gccVal1,gccVal2)

	return jacobian


def ifFeasibleOrdering(a,params,xs,ys,zs,ordering,boundMap, hyperBound):
	lenV = len(xs)
	#print "ordering ", ordering
	hyperRectangle = np.zeros((lenV,2))
	for i in range(lenV):
		if(ordering[i] is not None):
			hyperRectangle[i][0] = boundMap[ordering[i]][0]
			hyperRectangle[i][1] = boundMap[ordering[i]][1]
		else:
			hyperRectangle[i][0] = -1.0
			hyperRectangle[i][1] = 1.0
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
	possibleExistence = False
	while True:
		#print "hyperRectangle "
		#print hyperRectangle
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
			objIndex = fwdInd

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

		#print "allConstraints"
		#print allConstraints
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

		#print "newHyperRectangle ", newHyperRectangle
		if feasible == False:
			#print "LP not feasible"
			return (False, None)

		if np.less_equal(newHyperRectangle[:,1] - newHyperRectangle[:,1],hyperBound*np.ones((lenV))).any() or np.less_equal(np.absolute(newHyperRectangle - hyperRectangle),1e-4*np.ones((lenV,2))).all():
			# because due to numerical issues the actual solution
			# might lie slightly outside newHyperRectangle
			newHyperRectangle = hyperRectangle
			for i in range(lenV):
				# because this might be possible
				if(newHyperRectangle[i,1] < newHyperRectangle[i,0]):
					newHyperRectangle[i,0] = hyperRectangle[i,0]
					newHyperRectangle[i,1] = hyperRectangle[i,1]
			kResult = intervalUtils.checkExistenceOfSolution(a,params[0],params[1],newHyperRectangle.transpose(), oscNum, getJacobian, getJacobianInterval)
			if kResult[0] or (kResult[0] == False and kResult[1] is not None):
				#print "LP feasible ", newHyperRectangle
				return (True, newHyperRectangle)
			else:
				#print "LP not feasible"
				return (False, None)
		hyperRectangle = newHyperRectangle

	

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
def getFeasibleIntervalIndices(rootCombinationNode,a,params,xs,ys,zs,boundMap,hyperBound, outValidIntervalIndices, outConflictIntervalIndices):
	intervalIndices = rootCombinationNode.rootArray
	
	# Check if the current intervalIndices array
	# matches with any conflict array already discovered
	# If it does then we ignore it and its children 
	for conflict in outConflictIntervalIndices:
		foundConflict = True
		for i in range(len(conflict)):
			if conflict[i] is not None:
				if conflict[i] != intervalIndices[i]:
					foundConflict = False
					break
		if foundConflict:
			#print "found conflict: match is between"
			#print conflict, " and ", intervalIndices 
			return

	
	feasiblity = ifFeasibleOrdering(a,params,xs,ys,zs,intervalIndices,boundMap,hyperBound)
	#print "intervalIndices ", intervalIndices
	
	# Generate a conflict interval array if constraints are infeasible
	if feasiblity[0] == False:
		baseConflictInterval = copy.deepcopy(intervalIndices)
		indexChanged = None
		# Keep setting non None indices to None as long as
		# lp is infeasible. This will lead us to an array with
		# the least number of non None's that have been causing 
		# infeasibility
		while True:
			for i in range(len(intervalIndices)):
				if baseConflictInterval[i] is not None and i > indexChanged:
					indexChanged =  i
					break
				if i == len(intervalIndices) - 1:
					if baseConflictInterval != intervalIndices:
						outConflictIntervalIndices.append(baseConflictInterval)
						#print "finalBaseConflictInterval ", baseConflictInterval
					return
			indexBefore = baseConflictInterval[indexChanged]
			baseConflictInterval[indexChanged] = None
			feasibility = ifFeasibleOrdering(a,params,xs,ys,zs,baseConflictInterval,boundMap,hyperBound)
			if feasibility[0]:
				baseConflictInterval[indexChanged] = indexBefore
			#print "baseConflictInterval after ", baseConflictInterval
		return
	
	indexOfNone = None
	for i in range(len(intervalIndices)):
		if intervalIndices[i] is None:
			indexOfNone = i
			break
	if indexOfNone is None:
		outValidIntervalIndices.append(copy.deepcopy(intervalIndices))
	for i in range(len(rootCombinationNode.children)):
		getFeasibleIntervalIndices(rootCombinationNode.children[i],a,params,xs,ys,zs,boundMap,hyperBound,outValidIntervalIndices, outConflictIntervalIndices)

def refineHyper(a, params, xs, ys, zs, ordering, boundMap, maxHyperBound):
	lenV = len(xs)
	hyperRectangle = np.zeros((lenV,2))
	excludingRegConstraint = ""
	for i in range(lenV):
		hyperRectangle[i][0] = boundMap[ordering[i]][0]
		hyperRectangle[i][1] = boundMap[ordering[i]][1]
		
	finalHyper = []
	count = 0
	volumes = []
	#while True:
		#print "Iteration #", count
	diffHyper = hyperRectangle[:,1] - hyperRectangle[:,0]
	volume = np.prod(diffHyper)
	volumeRoot = volume**lenV
	volumes.append(volumeRoot)
	startRefine = time.time()
	feasibility = ifFeasibleOrdering(a,params,xs,ys,zs,ordering,boundMap, maxHyperBound)
	if feasibility[0] == False:
		return finalHyper
	newHyperRectangle = feasibility[1]
	endRefine = time.time()
	#print "hyperRectangle ", newHyperRectangle

	hyperRectangle = newHyperRectangle
	exampleSoln = (hyperRectangle[:,0] + hyperRectangle[:,1])/2.0
	finalSoln = intervalUtils.newton(a,params,exampleSoln, oscNum, getJacobian)
	#print "finalSoln ", finalSoln

	startKoperator = time.time()
	kResult = intervalUtils.checkExistenceOfSolution(a,params[0],params[1],hyperRectangle.transpose(), oscNum, getJacobian, getJacobianInterval)
	endKoperator = time.time()
	print "time taken to refine", endRefine - startRefine
	print "time taken for Koperator", endKoperator - startKoperator
	print "" 
	count += 1
	#TODO: need to deal with when K-operator does not know if unique solution or no solution
	if kResult[0]:
		finalHyper.append(hyperRectangle)
	elif kResult[0] == False and kResult[1] is not None:
		print "before bisecting num ", len(finalHyper)
		bisectHyper(a,params,xs,ys,zs,maxHyperBound,hyperRectangle,0,finalHyper)
		print "after bisecting num ", len(finalHyper)

	return finalHyper

def bisectHyper(a,params,xs,ys,zs,hyperBound,hyperRectangle,bisectingIndex, finalHypers):
	lenV = hyperRectangle.shape[0]
	if bisectingIndex >= lenV:
		bisectingIndex = 0
	leftHyper = copy.deepcopy(hyperRectangle)
	rightHyper = copy.deepcopy(hyperRectangle)
	midVal = (hyperRectangle[bisectingIndex][0] + hyperRectangle[bisectingIndex][1])/2.0
	leftHyper[bisectingIndex][1] = midVal
	rightHyper[bisectingIndex][0] = midVal
	feasLeft = ifFeasibleHyper(a,params,xs,ys,zs,leftHyper, hyperBound)
	feasRight = ifFeasibleHyper(a,params,xs,ys,zs,rightHyper, hyperBound)

	if feasLeft[0]:
		leftHyper = feasLeft[1]
		kResultLeft = intervalUtils.checkExistenceOfSolution(a,params[0],params[1],leftHyper.transpose(), oscNum, getJacobian, getJacobianInterval)
		if kResultLeft[0]:
			finalHypers.append(leftHyper)
		if kResultLeft[0] == False and kResultLeft[1] is not None:
			bisectHyper(a,params,xs,ys,zs,hyperBound,leftHyper,bisectingIndex+1,finalHypers)
	
	if feasRight[0]:
		rightHyper = feasRight[1]
		kResultRight = intervalUtils.checkExistenceOfSolution(a,params[0],params[1],rightHyper.transpose(), oscNum, getJacobian, getJacobianInterval)
		if kResultRight[0]:
			finalHypers.append(rightHyper)
		if kResultRight[0] == False and kResultRight[1] is not None:
			bisectHyper(a,params,xs,ys,zs,hyperBound,rightHyper,bisectingIndex+1,finalHypers)



def findExcludingBound(a,params,xs,ys,zs,ordering,boundMap, maxDiff = 0.2):
	lenV = len(xs)
	hyperRectangle = np.zeros((lenV,2))
	hyperBound = 0.001
	for i in range(lenV):
		hyperRectangle[i][0] = boundMap[ordering[i]][0]
		hyperRectangle[i][1] = boundMap[ordering[i]][1]
	soln = hyperRectangle[:,0] + (hyperRectangle[:,1] - hyperRectangle[:,0])*0.75;
	soln = intervalUtils.newton(a,params,soln,oscNum,getJacobian)
	diff = maxDiff
	while True:
		hyperRectangle[:,0] = soln - diff
		hyperRectangle[:,1] = soln + diff
		kResult = intervalUtils.checkExistenceOfSolution(a,params[0],params[1],hyperRectangle.transpose(), oscNum, getJacobian, getJacobianInterval)
		if kResult[0] == False and kResult[1] is not None:
			diff = diff/2.0;
		else:
			break
	boundMap[0][1] = diff
	boundMap[1][0] = diff
	'''boundMap[-1] = [-1.0,-diff]
	boundMap[0] = [-diff,0.0]
	boundMap[1] = [0.0, diff]
	boundMap[2] = [diff, 1.0]
	return [0,1]'''

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
	params = [1.0,0.5]
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
	boundMap = {0:[-1.0,0.0],1:[0.0,1.0]}
	findExcludingBound(a,params,xs,ys,zs,exampleOrdering,boundMap)
	print "boundMap ", boundMap
	minBoundMap = float('inf')
	maxBoundMap = float('-inf')
	for key in boundMap:
		if key > maxBoundMap:
			maxBoundMap = key
		if key < minBoundMap:
			minBoundMap = key

	print "minBoundMap ", minBoundMap, " maxBoundMap ", maxBoundMap
	rootCombinationNodes = intervalUtils.combinationWithTrees(lenV,[minBoundMap,maxBoundMap],indexChoiceArray)
	#intervalUtils.printCombinationNode(rootCombinationNodes[1])
	hyperBound = 0.0001
	#feasibility = ifFeasibleOrdering(a,params,xs,ys,zs,[0,1,0,1],boundMap, hyperBound)
	#print feasibility
	'''hyperRectangle = np.zeros((lenV,2))
	hyperRectangle[0,:] = [ 0.81635344, 0.90817673]
	hyperRectangle[1,:] = [-0.99986002, -0.99981331]
	hyperRectangle[2,:] = [-0.59942304, -0.54925001]
	hyperRectangle[3,:] = [ 0.99889157, 0.99889681]
	hyperRectangle[4,:] = [-0.03452613, 0.09640916]
	hyperRectangle[5,:] = [-0.1792397, 0.20000004]
	hyperRectangle[6,:] = [-0.92818052, -0.80495914]
	hyperRectangle[7,:] = [ 0.9997996, 0.9998448 ]
	hyperRectangle[8,:] = [ 0.59587838, 0.5958992 ]
	hyperRectangle[9,:] = [-0.99977589, -0.98718653]
	hyperRectangle[10,:] = [-0.07779784, -0.04154983]
	hyperRectangle[11,:] = [-0.1626185, 0.20000004]

	feasibility = ifFeasibleHyper(a,params,xs,ys,zs,hyperRectangle, hyperBound)
	print feasibility
	exampleSoln = (hyperRectangle[:,0] + hyperRectangle[:,1])/2.0
	finalSoln = intervalUtils.newton(a,params,exampleSoln, oscNum, getJacobian)
	print "finalSoln", finalSoln'''
	'''ordering = [1, 0, 1, 0, 0, 1, 0, 1]
	hypers = refineHyper(a, params, xs, ys, zs, ordering, boundMap, hyperBound)
	print "hypers"
	print hypers
	exampleSoln = (hypers[0][:,0] + hypers[0][:,1])/2.0
	finalSoln = intervalUtils.newton(a,params,exampleSoln, oscNum, getJacobian)
	print "finalSoln ", finalSoln'''
	feasibleIntervalIndices = []
	conflictIntervalIndices = []
	for i in range(len(rootCombinationNodes)):
		getFeasibleIntervalIndices(rootCombinationNodes[i],a,params,xs,ys,zs,boundMap,hyperBound, feasibleIntervalIndices,conflictIntervalIndices)
	
	#feasibleIntervalIndices = [[0, 0, 0, 0]]
	print "feasibleIntervalIndices"
	print feasibleIntervalIndices
	print "conflictIntervalIndices"
	print conflictIntervalIndices
	allHypers = []
	allSols = []
	sampleSols = []
	rotatedSols = {}
	stableSols = []
	unstableSols = []
	
	maxHyperBound = 0.1
	for fi in range(len(feasibleIntervalIndices)):
		feasibleIntervalIndex = feasibleIntervalIndices[fi]
		print "feasibleIntervalIndex ", feasibleIntervalIndex
		refinedHyper = refineHyper(a, params, xs, ys, zs, feasibleIntervalIndex, boundMap, maxHyperBound)
		print "refinedHyper "
		print refinedHyper
		if(len(refinedHyper) > 0):
			allHypers.append(refinedHyper[0])
			exampleSoln = (refinedHyper[0][:,0] + refinedHyper[0][:,1])/2.0
			finalSoln = intervalUtils.newton(a,params,exampleSoln, oscNum, getJacobian)
			print "exampleSoln ", exampleSoln
			print "finalSoln ", finalSoln
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
	
	'''for hi in range(len(allHypers)):
		print "sol#", hi
		print "hyper "
		print allHypers[hi]
		print "actual sol"
		print allSols[hi]
		print ""'''

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
	print "num unstable solutions ", len(unstableSols)
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

