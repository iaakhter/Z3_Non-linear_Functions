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
			'''return Implies(And(Vin >= Vlow, Vin <= Vhigh),
							And(Vout >= dThird*Vin + cThird,
								Vout <= dLow*Vin + cLow,
								Vout <= dHigh*Vin + cHigh))'''
			return overallConstraint + "1 "+ Vout + " + " +str(-dThird) + " " + Vin + " >= "+str(cThird)+"\n" +\
			"1 "+Vout + " + " +str(-dLow) + " " + Vin + " <= "+str(cLow)+"\n" +\
			"1 "+Vout + " + " +str(-dHigh) + " " + Vin + " <= "+str(cHigh) + "\n"

		elif Vlow <=0 and Vhigh <=0:
			'''return Implies(And(Vin >= Vlow, Vin <= Vhigh),
							And(Vout <= dThird*Vin + cThird,
								Vout >= dLow*Vin + cLow,
								Vout >= dHigh*Vin + cHigh))'''
			return overallConstraint + "1 "+ Vout + " + " +str(-dThird) + " " + Vin + " <= "+str(cThird)+"\n" +\
			"1 "+Vout + " + " +str(-dLow) + " " + Vin + " >= "+str(cLow)+"\n" +\
			"1 "+Vout + " + " +str(-dHigh) + " " + Vin + " >= "+str(cHigh) + "\n"

	elif a < 0:
		if Vlow <= 0 and Vhigh <=0:
			'''return Implies(And(Vin >= Vlow, Vin <= Vhigh),
							And(Vout >= dThird*Vin + cThird,
								Vout <= dLow*Vin + cLow,
								Vout <= dHigh*Vin + cHigh))'''
			return overallConstraint + "1 "+Vout + " + " +str(-dThird) + " " + Vin + " >= "+str(cThird)+"\n" +\
			"1 "+Vout + " + " +str(-dLow) + " " + Vin + " <= "+str(cLow)+"\n" +\
			"1 "+Vout + " + " +str(-dHigh) + " " + Vin + " <= "+str(cHigh) + "\n"

		elif Vlow >=0 and Vhigh >=0:
			'''return Implies(And(Vin >= Vlow, Vin <= Vhigh),
							And(Vout <= dThird*Vin + cThird,
								Vout >= dLow*Vin + cLow,
								Vout >= dHigh*Vin + cHigh))'''
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
def ifFeasible(a,params,xs,ys,zs,ordering,boundMap, hyperBound, excludingBound = None):
	solvers.options["show_progress"] = False
	lenV = len(xs)
	print "ordering ", ordering
	allConstraints = ""
	excludingRegConstraint = ""
	hyperRectangle = np.zeros((lenV,2))
	countNonNones = 0
	for i in range(lenV):
		if(ordering[i] is not None):
			hyperRectangle[i][0] = boundMap[ordering[i]][0]
			hyperRectangle[i][1] = boundMap[ordering[i]][1]
			countNonNones += 1
		else:
			hyperRectangle[i][0] = -1.0
			hyperRectangle[i][1] = 1.0
	while True:
		#print "hyperRectangle "
		#print hyperRectangle
		for i in range(lenV):
			fwdInd = (i-1)%lenV
			ccInd = (i+lenV/2)%lenV
			#print "fwdInd ", fwdInd, " ccInd ", ccInd
			#print "hyperRectangle[fwdInd][0]", hyperRectangle[fwdInd][0], "hyperRectangle[fwdInd][1]", hyperRectangle[fwdInd][1]
			if ordering[fwdInd] is not None:
				triangleClaimFwd = triangleBounds(a,xs[fwdInd],ys[i],hyperRectangle[fwdInd][0],hyperRectangle[fwdInd][1])
				allConstraints += triangleClaimFwd
				objIndex = fwdInd

				if(ordering[fwdInd] == 1):
					excludingRegConstraint += "1 " + xs[fwdInd]
				else:
					excludingRegConstraint += "-1 " + xs[fwdInd]
				if i < lenV-1:
					excludingRegConstraint += " + " 
			else:
				dummyConstraint = "1 " + xs[fwdInd] + " <= 1\n"
				dummyConstraint += "1 " + xs[fwdInd] + " >= -1\n"
				dummyConstraint += "1 " + ys[i] + " <= 1\n"
				dummyConstraint += "1 " + ys[i] + " >= -1\n"
				allConstraints += dummyConstraint

			if ordering[ccInd] is not None:
				triangleClaimCc = triangleBounds(a,xs[ccInd],zs[i],hyperRectangle[ccInd][0],hyperRectangle[ccInd][1])
				allConstraints += triangleClaimCc

			else:
				dummyConstraint = "1 " + xs[ccInd] + " <= 1\n"
				dummyConstraint += "1 " + xs[ccInd] + " >= -1\n"
				dummyConstraint += "1 " + zs[i] + " <= 1\n"
				dummyConstraint += "1 " + zs[i] + " >= -1\n"
				allConstraints += dummyConstraint
			
			allConstraints += str(params[0]) + " " + ys[i] + " + " + str(-params[0]-params[1]) + \
			" " + xs[i] + " + " + str(params[1]) + " "  + zs[i] + " >= 0.0\n"
			allConstraints += str(params[0]) + " " + ys[i] + " + " + str(-params[0]-params[1]) + \
			" " + xs[i] + " + " + str(params[1]) + " "  + zs[i] + " <= 0.0\n"

		if excludingBound is not None:
			excludingRegConstraint += " >= "+str(excludingBound)+"\n"
			allConstraints = allConstraints + excludingRegConstraint

		#print "allConstraints"
		#print allConstraints
		variableDict, A, B = lpUtils.constructCoeffMatrices(allConstraints)
		newHyperRectangle = copy.deepcopy(hyperRectangle)
		'''for i in range(lenV):
			if(ordering[i] is not None):
				newHyperRectangle[i][0] = boundMap[ordering[i]][0]
				newHyperRectangle[i][1] = boundMap[ordering[i]][1]'''
		feasible = True
		for i in range(lenV):
			#print "min max ", i
			if(ordering[i] is not None):
				minObjConstraint = "min 1 " + xs[i]
				maxObjConstraint = "max 1 " + xs[i]
				Cmin = lpUtils.constructObjMatrix(minObjConstraint,variableDict)
				Cmax = lpUtils.constructObjMatrix(maxObjConstraint,variableDict)
				minSol = solvers.lp(Cmin,A,B)
				maxSol = solvers.lp(Cmax,A,B)

				if (minSol["status"] != "primal infeasible" or minSol["status"] != "dual infeasible")  and (maxSol["status"] == "primal infeasible" or maxSol["status"] == "dual infeasible"):
					feasible = False
					break
				else:
					if minSol["status"] == "optimal":
						newHyperRectangle[i][0] = minSol['x'][variableDict[xs[i]]]
						if ordering[i] == 1 and newHyperRectangle[i][0] < 0:
							newHyperRectangle[i][0] = 0.0
						if ordering[i] == 0 and newHyperRectangle[i][0] > 0:
							newHyperRectangle[i][0] = 0.0
					if maxSol["status"] == "optimal":
						newHyperRectangle[i][1] = maxSol['x'][variableDict[xs[i]]]
						if ordering[i] == 0 and newHyperRectangle[i][1] > 0:
							newHyperRectangle[i][1] = 0.0
						if ordering[i] == 1 and newHyperRectangle[i][1] < 0:
							newHyperRectangle[i][1] = 0.0

		if feasible == False:
			print "LP not feasible"
			#findCauseOfInfeasibility(a,params,xs,ys,zs,hyperRectangle)
			return (False, None)
		if np.less_equal(newHyperRectangle[:,1] - newHyperRectangle[:,0], np.ones((lenV))*hyperBound ).all() or np.less_equal(np.absolute(newHyperRectangle - hyperRectangle),1e-4*np.ones((lenV,2))).all():
			#print "Coming here?"
			kResult = intervalUtils.checkExistenceOfSolution(a,params[0],params[1],newHyperRectangle.transpose(), oscNum, getJacobian, getJacobianInterval)
			if kResult[0] or (kResult[0] == False and kResult[1] is not None):
				return (True, hyperRectangle)
			else:
				print "LP not feasible"
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
def getFeasibleIntervalIndices(rootCombinationNode,a,params,xs,ys,zs,boundMap,hyperBound, excludingBound,outValidIntervalIndices, outConflictIntervalIndices):
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
			print "found conflict: match is between"
			print conflict, " and ", intervalIndices 
			return
	
	feasiblity = ifFeasible(a,params,xs,ys,zs,intervalIndices,boundMap,hyperBound,excludingBound)
	print "intervalIndices ", intervalIndices
	
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
						print "finalBaseConflictInterval ", baseConflictInterval
					return
			indexBefore = baseConflictInterval[indexChanged]
			baseConflictInterval[indexChanged] = None
			feasibility = ifFeasible(a,params,xs,ys,zs,baseConflictInterval,boundMap,hyperBound,excludingBound)
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
		getFeasibleIntervalIndices(rootCombinationNode.children[i],a,params,xs,ys,zs,boundMap,hyperBound, excludingBound,outValidIntervalIndices, outConflictIntervalIndices)

def refineHyper(a, params, xs, ys, zs, ordering, boundMap, excludingBound, maxHyperBound):
	lenV = len(xs)
	hyperRectangle = np.zeros((lenV,2))
	excludingRegConstraint = ""
	for i in range(lenV):
		hyperRectangle[i][0] = boundMap[ordering[i]][0]
		hyperRectangle[i][1] = boundMap[ordering[i]][1]
		if(ordering[i] == 1):
			excludingRegConstraint += "1 " + xs[i]
		else:
			excludingRegConstraint += "-1 " + xs[i]
		if i < lenV-1:
			excludingRegConstraint += " + " 
	excludingRegConstraint += " >= "+str(excludingBound)+"\n"
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
	feasibility = ifFeasible(a,params,xs,ys,zs,ordering,boundMap, maxHyperBound, excludingBound)
	if feasibility[0] == False:
		return finalHyper
	newHyperRectangle = feasibility[1]
	endRefine = time.time()
	print "hyperRectangle ", newHyperRectangle

	hyperRectangle = newHyperRectangle
	
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
		bisectHyper(a,params,hyperRectangle,0,finalHyper)

	return finalHyper

def bisectHyper(a,params,hyperRectangle,bisectingIndex, finalHypers):
	lenV = hyperRectangle.shape[0]
	if bisectingIndex >= lenV:
		bisectingIndex = 0
	leftHyper = copy.deepcopy(hyperRectangle)
	rightHyper = copy.deepcopy(hyperRectangle)
	midVal = (hyperRectangle[bisectingIndex][0] + hyperRectangle[bisectingIndex][1])/2.0
	leftHyper[bisectingIndex][1] = midVal
	rightHyper[bisectingIndex][0] = midVal
	kResultLeft = intervalUtils.checkExistenceOfSolution(a,params[0],params[1],leftHyper.transpose(), oscNum, getJacobian, getJacobianInterval)
	kResultRight = intervalUtils.checkExistenceOfSolution(a,params[0],params[1],rightHyper.transpose(), oscNum, getJacobian, getJacobianInterval)
	if kResultLeft[0]:
		finalHypers.append(leftHyper)
	if kResultRight[0]:
		finalHypers.append(rightHyper)

	if kResultLeft[0] == False and kResultLeft[1] is not None:
		bisectHyper(a,params,leftHyper,bisectingIndex+1,finalHypers)

	if kResultRight[0] == False and kResultRight[1] is not None:
		bisectHyper(a,params,rightHyper,bisectingIndex+1,finalHypers)

def newton(a,params,ordering,boundMap):
	lenV = len(ordering)
	hyperRectangle = np.zeros((lenV,2))
	for i in range(lenV):
		hyperRectangle[i][0] = boundMap[ordering[i]][0]
		hyperRectangle[i][1] = boundMap[ordering[i]][1]
	soln = hyperRectangle[:,0] + (hyperRectangle[:,1] - hyperRectangle[:,0])*0.75;
	h = soln
	count = 0
	maxIter = 100
	while np.linalg.norm(h) > 1e-8 and count < maxIter:
		_,_,res = oscNum(soln,a,params[1],params[0])
		res = -np.array(res)
		jac = getJacobian(soln,a,params[1],params[0])
		h = np.linalg.solve(jac,res)
		soln = soln + h
		count+=1
	return soln

def findExcludingBound(a,params,xs,ys,zs,ordering,boundMap, maxDiff = 0.2):
	lenV = len(xs)
	hyperRectangle = np.zeros((lenV,2))
	hyperBound = 0.001
	soln = newton(a,params,ordering,boundMap)
	diff = maxDiff
	while True:
		hyperRectangle[:,0] = soln - diff
		hyperRectangle[:,1] = soln + diff
		kResult = intervalUtils.checkExistenceOfSolution(a,params[0],params[1],hyperRectangle.transpose(), oscNum, getJacobian, getJacobianInterval)
		if kResult[0] == False and kResult[1] is not None:
			diff = diff/2.0;
		else:
			return diff

def twoStageOscillator(a):
	params = [1.0,4.0]
	xs = ["x0","x1","x2","x3"]
	ys = ["y0","y1","y2","y3"]
	zs = ["z0","z1","z2","z3"]
	lenV = 4
	exampleOrdering = [0,0,0,0]
	
	'''xs = ["x0","x1","x2","x3","x4","x5","x6","x7"]
	ys = ["y0","y1","y2","y3","y4","y5","y6","y7"]
	zs = ["z0","z1","z2","z3","z4","z5","z6","z7"]
	lenV = 8
	exampleOrdering = [0,0,0,0,0,0,0,0]'''
	
	rootCombinationNodes = intervalUtils.combinationWithTrees(lenV,[0,1])
	#printCombinationNode(rootCombinationNodes[1])
	excludingBound = 0.2
	hyperBound = 0.1
	boundMap = {0:[-1.0,0.0],1:[0.0,1.0]}
	excludingBound = findExcludingBound(a,params,xs,ys,zs,exampleOrdering,boundMap)
	#print "excludingBound ", excludingBound
	#ifFeasible(a,params,xs,ys,zs,[None,0,0,None,None,None,0,None],boundMap, hyperBound, excludingBound)
	feasibleIntervalIndices = []
	conflictIntervalIndices = []
	for i in range(len(rootCombinationNodes)):
		getFeasibleIntervalIndices(rootCombinationNodes[i],a,params,xs,ys,zs,boundMap,hyperBound, excludingBound,feasibleIntervalIndices,conflictIntervalIndices)
	
	#feasibleIntervalIndices = [[0,1,0,1,0,1,0,1]]
	print "feasibleIntervalIndices"
	print feasibleIntervalIndices
	print "conflictIntervalIndices"
	print conflictIntervalIndices
	allHypers = []
	
	maxHyperBound = 0.1
	for fi in range(len(feasibleIntervalIndices)):
		feasibleIntervalIndex = feasibleIntervalIndices[fi]
		refinedHyper = refineHyper(a, params, xs, ys, zs, feasibleIntervalIndex, boundMap, excludingBound, maxHyperBound)
		if(len(refinedHyper) > 0):
			allHypers.append(refinedHyper[0])
	
	#plt.plot(np.arange(len(volumes)),volumes)
	#plt.xlabel("Number of iterations")
	#plt.ylabel("Cube root of volumes")
	#plt.show()
	print "allHypers "
	print allHypers


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
twoStageOscillator(-5.0)

