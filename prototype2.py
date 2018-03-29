import numpy as np
import time
import copy
import intervalUtils
import tanhModel
import mosfetModel
from treelib import Node, Tree
from z3 import *

hyperNum = 0

def ifFeasibleOrdering(ordering,boundMap, hyperBound, model):
	lenV = model.numStages*2
	#print "ordering ", ordering
	hyperRectangle = np.zeros((lenV,2))
	for i in range(lenV):
		if(ordering[i] is not None):
			hyperRectangle[i][0] = boundMap[i][ordering[i]][0]
			hyperRectangle[i][1] = boundMap[i][ordering[i]][1]
		else:
			hyperRectangle[i][0] = boundMap[i][0][0]
			hyperRectangle[i][1] = boundMap[i][1][1]
	return ifFeasibleHyper(hyperRectangle,hyperBound,model)



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
def ifFeasibleHyper(hyperRectangle, hyperBound,model):
	lenV = model.numStages*2
	print ("hyperRectangle")
	print (hyperRectangle)
	iterNum = 0
	while True:
		#print ("hyperRectangle ")
		#print (hyperRectangle)
		kResult = intervalUtils.checkExistenceOfSolutionGS(model,hyperRectangle.transpose())
		#print ("kResult")
		#print (kResult)
		if kResult[0]:
			#print "LP feasible ", newHyperRectangle
			return (True, kResult[1])

		if kResult[0] == False and kResult[1] is None:
			print ("K operator not feasible")
			return (False, None)
		#print "kResult"
		#print kResult
		#print ("hyperRectangle ")
		#print (hyperRectangle)
		feasible, newHyperRectangle = model.linearConstraints(hyperRectangle)
	
		#print ("newHyperRectangle ", newHyperRectangle)
		if feasible == False:
			print ("LP not feasible")
			return (False, None)

		for i in range(lenV):
			if newHyperRectangle[i,0] < hyperRectangle[i,0]:
				newHyperRectangle[i,0] = hyperRectangle[i,0]
			if newHyperRectangle[i,1] > hyperRectangle[i,1]:
				newHyperRectangle[i,1] = hyperRectangle[i,1]
			 

		if np.less_equal(newHyperRectangle[:,1] - newHyperRectangle[:,0],hyperBound*np.ones((lenV))).all() or np.less_equal(np.absolute(newHyperRectangle - hyperRectangle),1e-4*np.ones((lenV,2))).all():
			if kResult[0] == False and kResult[1] is not None:
				return (False, kResult[1])
		hyperRectangle = newHyperRectangle
		iterNum+=1
		#print ("here?")
		#return
	

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

@param parentTree, parent, level to keep track of search tree
'''
def getFeasibleIntervalIndices(rootCombinationNode,boundMap,hyperBound, model,refinedHypers, parentTree, parent, level = 0):
	global hyperNum
	hyperId = str(hyperNum)
	hyperNum += 1

	intervalIndices = rootCombinationNode.rootArray
	print ("intervalIndices", intervalIndices)
	print ("boundMap", boundMap)
	lenV = len(intervalIndices)
	
	#feasibility = ifFeasibleOrdering(intervalIndices, boundMap, hyperBound,model)
	hyperRectangle = np.zeros((lenV,2))
	for i in range(lenV):
		if(intervalIndices[i] is not None):
			hyperRectangle[i][0] = boundMap[i][intervalIndices[i]][0]
			hyperRectangle[i][1] = boundMap[i][intervalIndices[i]][1]
		else:
			hyperRectangle[i][0] = boundMap[i][0][0]
			hyperRectangle[i][1] = boundMap[i][1][1]
	
	origHyperChild = np.copy(hyperRectangle)
	feasibility = ifFeasibleHyper(hyperRectangle,hyperBound,model)

	print ("feasibility at level", level)
	print (feasibility)

	hyper_feasHyperString = np.array_str(origHyperChild) + "\n"
	if feasibility[0]:
		refinedHypers.append(feasibility[1])
		hyper_feasHyperString += " TRUE\n" + np.array_str(feasibility[1]) + "\n"
		parentTree.create_node(hyper_feasHyperString, hyperId, parent = parent)
		return

	if feasibility[0] == False and feasibility[1] is None:
		hyper_feasHyperString += " FALSE\n" + " NONE\n"
		parentTree.create_node(hyper_feasHyperString, hyperId, parent = parent)
		return

	hyper_feasHyperString += " ****\n" + np.array_str(feasibility[1]) + "\n"
	parentTree.create_node(hyper_feasHyperString, hyperId, parent = parent)

	newBoundMap = copy.deepcopy(boundMap)
	hyperRectangle = feasibility[1]
	for i in range(lenV):
		lowBound = hyperRectangle[i][0]
		upperBound = hyperRectangle[i][1]
		if intervalIndices[i] is None:
			newBoundMap[i][0][0] = lowBound
			newBoundMap[i][1][1] = upperBound
			'''if lowBound < 0.5 and upperBound > 0.5 and upperBound < 0.6:
				newBoundMap[i][0][1] = (lowBound + upperBound)/2.0
				newBoundMap[i][1][0] = (lowBound + upperBound)/2.0
			elif (lowBound <= 0.5 and upperBound <= 0.5) or (lowBound >= 0.5 and upperBound >= 0.5):
				newBoundMap[i][0][1] = (lowBound + upperBound)/2.0
				newBoundMap[i][1][0] = (lowBound + upperBound)/2.0
			else:
				newBoundMap[i][0][1] = 0.6
				newBoundMap[i][1][0] = 0.6'''
			newBoundMap[i][0][1] = (lowBound + upperBound)/2.0
			newBoundMap[i][1][0] = (lowBound + upperBound)/2.0
		else:
			newBoundMap[i][0] = [lowBound, upperBound]
			newBoundMap[i][1] = [lowBound, upperBound]

	indexOfNone = None
	for i in range(len(intervalIndices)):
		if intervalIndices[i] is None:
			indexOfNone = i
			break
	print ("indexOfNone ", indexOfNone)
	if indexOfNone is None:
		bisectionHypers = refineHyper(intervalIndices, newBoundMap, hyperBound, model, parentTree, hyperId, level+1)
		print ("bisectionHypers")
		print (bisectionHypers)
		print ("len(refinedHypers) before ", len(refinedHypers))
		for hyper in bisectionHypers:
			refinedHypers.append(hyper)
		print ("len(refinedHypers) after ", len(refinedHypers))
	for i in range(len(rootCombinationNode.children)):
		getFeasibleIntervalIndices(rootCombinationNode.children[i],
			newBoundMap, hyperBound, model, refinedHypers, parentTree, hyperId, level+1)

def refineHyper(ordering, boundMap, maxHyperBound, model, parentTree, parent, level):
	lenV = model.numStages*2
	hyperRectangle = np.zeros((lenV,2))
	excludingRegConstraint = ""
	for i in range(lenV):
		if ordering[i] is not None:
			hyperRectangle[i][0] = boundMap[i][ordering[i]][0]
			hyperRectangle[i][1] = boundMap[i][ordering[i]][1]
		else:
			hyperRectangle[i][0] = boundMap[i][0][0]
			hyperRectangle[i][1] = boundMap[i][1][1]
	finalHyper = []
	count = 0
	volumes = []

	print ("before bisecting num ", len(finalHyper))
	bisectHyper(maxHyperBound, hyperRectangle, 0,model, finalHyper, parentTree, parent, level)
	print ("after bisecting num ", len(finalHyper))

	return finalHyper

def bisectHyper(hyperBound,hyperRectangle,bisectingIndex, model,finalHypers, parentTree, parent, level):
	global hyperNum
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
	leftHyper_feasHyperString = np.array_str(leftHyper) + "\n"
	leftHyperId = str(hyperNum)
	hyperNum += 1

	rightHyper[bisectingIndex][0] = midVal
	rightHyper_feasHyperString = np.array_str(rightHyper) + "\n"
	rightHyperId = str(hyperNum)
	hyperNum += 1
	print ("leftHyper at level", level)
	print (leftHyper)
	print ("rightHyper at level", level)
	print (rightHyper)
	feasLeft = ifFeasibleHyper(leftHyper, hyperBound, model)
	feasRight = ifFeasibleHyper(rightHyper, hyperBound, model)
	print ("feasLeft")
	print (feasLeft)
	print ("feasRight")
	print (feasRight)


	if feasLeft[0]:
		finalHypers.append(feasLeft[1])
		leftHyper_feasHyperString += " TRUE\n" + np.array_str(feasLeft[1]) + "\n"
		parentTree.create_node(leftHyper_feasHyperString, leftHyperId, parent = parent)
	elif feasLeft[0] == False and feasLeft[1] is not None:
		leftHyper_feasHyperString += " ****\n" + np.array_str(feasLeft[1]) + "\n"
		parentTree.create_node(leftHyper_feasHyperString, leftHyperId, parent = parent)
		bisectHyper(hyperBound,feasLeft[1],bisectingIndex+1,model,finalHypers, parentTree, leftHyperId, level+1)
	else:
		leftHyper_feasHyperString += " FALSE\n" + " NONE\n"
		parentTree.create_node(leftHyper_feasHyperString, leftHyperId, parent = parent)

	
	if feasRight[0]:
		finalHypers.append(feasRight[1])
		rightHyper_feasHyperString += " TRUE\n" + np.array_str(feasRight[1]) + "\n"
		parentTree.create_node(rightHyper_feasHyperString, rightHyperId, parent = parent)
	elif feasRight[0] == False and feasRight[1] is not None:
		rightHyper_feasHyperString += " ****\n" + np.array_str(feasRight[1]) + "\n"
		parentTree.create_node(rightHyper_feasHyperString, rightHyperId, parent = parent)
		bisectHyper(hyperBound,feasRight[1],bisectingIndex+1,model,finalHypers, parentTree, rightHyperId, level+1)
	else:
		rightHyper_feasHyperString += " FALSE\n" + " NONE\n"
		parentTree.create_node(rightHyper_feasHyperString, rightHyperId, parent = parent)

def findExcludingBound(ordering,boundMap, model, maxDiff = 0.2):
	lenV = model.numStages*2
	hyperRectangle = np.zeros((lenV,2))
	hyperBound = 0.001
	for i in range(lenV):
		hyperRectangle[i][0] = boundMap[i][ordering[i]][0]
		hyperRectangle[i][1] = boundMap[i][ordering[i]][1]
	soln = hyperRectangle[:,0] + (hyperRectangle[:,1] - hyperRectangle[:,0])*0.75;
	soln = intervalUtils.newton(model,soln)
	diff = maxDiff
	while True:
		hyperRectangle[:,0] = soln[1] - diff
		hyperRectangle[:,1] = soln[1] + diff
		kResult = intervalUtils.checkExistenceOfSolutionGS(model,hyperRectangle.transpose())
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
def determineStability(equilibrium,model):
	jac = model.jacobian(equilibrium)
	eigVals,_ = np.linalg.eig(jac)
	maxEig = np.amax(eigVals.real)
	if maxEig > 0:
		return False
	return True

def z3Version(a, numStages):
	start = time.time()
	model = tanhModel.TanhModel(modelParam = a, g_cc = 0.5, g_fwd = 1.0, numStages = numStages, useZ3 = True)
	lenV = numStages*2
	hyperRectangle1 = np.zeros((lenV,2))
	hyperRectangle2 = np.zeros((lenV,2))
	intervalMap = {}
	for i in range(lenV):
		hyperRectangle1[i,:] = [-1.0,0.0]
		hyperRectangle2[i,:] = [0.0,1.0]
		intervalMap[i] = [-1.0,0.0,1.0]
	s = Solver()

	domainConstraint = model.addDomainConstraint()
	s.add(domainConstraint)
	
	allHypers = []
	solutionsFoundSoFar = []
	while True:
		constraintList1 = model.linearConstraints(hyperRectangle1)
		constraint1 = And(*[constraintList1[ci] for ci in range(len(constraintList1))])
		constraintList2 = model.linearConstraints(hyperRectangle2)
		constraint2 = And(*[constraintList2[ci] for ci in range(len(constraintList2))])
		s.add(constraint1)
		s.add(constraint2)
		#s.push()
		'''for im in range(len(intervalMap[0])-1):
			hyperRectangle = np.zeros((lenV,2))
			for ii in range(lenV):
				hyperRectangle[ii,0] = intervalMap[ii][im]
				hyperRectangle[ii,1] = intervalMap[ii][im+1]
			constraintList = model.linearConstraints(hyperRectangle)
			constraint = And(*[constraintList[ci] for ci in range(len(constraintList))])
			s.add(constraint)'''

		ch = s.check()
		print ("ch", ch)
		if ch == sat:
			m = s.model()
			sol = np.zeros((lenV))
			for d in m.decls():
				dName = str(d.name())
				firstLetter = dName[0]
				if(dName[0]=="x" and dName[1]=="_"):
					index = int(dName[len(dName) - 1])
					val = float(Fraction(str(m[d])))
					#print "index: ", index, " val: ", val
					sol[index] = val

			#s.pop()
			print ("sol found before", sol)
			
			# See if the solution given by Z3 leads to an 
			# actual solution
			finalSoln = intervalUtils.newton(model,sol)
			# If an actual solution can be reached construct
			# biggest possible hyperrectangle and ask Z3 to ignore
			# it
			if finalSoln[0]:
				alreadyFound = False
				for exSol in solutionsFoundSoFar:
					diff = np.absolute(finalSoln[1] - exSol)
					if np.less_equal(diff, np.ones(diff.shape)*1e-15).all():
						alreadyFound = True
						break
				if not(alreadyFound):
					solutionsFoundSoFar.append(finalSoln[1])
					hyperWithUniqueSoln = np.zeros((lenV,2))
					diff = 0.2
					while True:
						#print ("diff", diff)
						hyperWithUniqueSoln[:,0] = finalSoln[1] - diff
						hyperWithUniqueSoln[:,1] = finalSoln[1] + diff
						kResult = intervalUtils.checkExistenceOfSolutionGS(model,hyperWithUniqueSoln.transpose())
						if kResult[0] == False and kResult[1] is not None:
							diff = diff/2.0;
						else:
							print ("found unique hyper", hyperWithUniqueSoln)
							allHypers.append(hyperWithUniqueSoln)
							ignoringConstraint = model.ignoreHyperInZ3(hyperWithUniqueSoln)
							#print ("ignorintConstraint", ignoringConstraint)
							s.add(ignoringConstraint)
							break

			'''for i in range(lenV):
				intervalMap[i].append(sol[i])
				intervalMap[i].sort()'''

			# The set of hyperrectangles to be used in addition
			# of constraints for the next iteration
			for i in range(lenV):
				intervals = intervalMap[i]
				for ii in range(len(intervals)-1):
					if sol[i] >= intervals[ii] and sol[i] <= intervals[ii+1]:
						hyperRectangle1[i][0] = intervals[ii]
						hyperRectangle1[i][1] = sol[i]
						hyperRectangle2[i][0] = sol[i]
						hyperRectangle2[i][1] = intervals[ii+1]
				intervalMap[i].append(sol[i])
				intervalMap[i] = list(set(intervalMap[i]))
				intervalMap[i].sort()
			#ignoringSolConstraint = model.ignoreSolInZ3(sol)
			#s.add(ignoringSolConstraint)
			print ("hyperRectangle1", hyperRectangle1)
			print ("hyperRectangle2", hyperRectangle2)
			print ("ignoring solution")
			
			# Construct biggest possible hyperrectangle around
			# solution proposed by Z3 that has no actual solution
			hyperWithoutSoln = np.zeros((lenV,2))
			diff = 0.2
			while True:
				hyperWithoutSoln[:,0] = sol - diff
				hyperWithoutSoln[:,1] = sol + diff
				kResult = intervalUtils.checkExistenceOfSolutionGS(model,hyperWithoutSoln.transpose())
				#print ("kResult", kResult)
				if diff > 0.0001 and (kResult[0] or (kResult[1] is not None)):
					diff = diff/2.0;
				elif diff <= 0.0001:
					print ("ignore solution by z3")
					ignoringSolConstraint = model.ignoreSolInZ3(sol)
					s.add(ignoringSolConstraint)
					break
				else:
					print ("found hyper with no solution", hyperWithoutSoln)
					ignoringConstraint = model.ignoreHyperInZ3(hyperWithoutSoln)
					#print ("ignorintConstraint", ignoringConstraint)
					s.add(ignoringConstraint)
					break
				'''else:
					ignoringSolConstraint = model.ignoreSolInZ3(sol)
					s.add(ignoringSolConstraint)
					break'''

			#print ("hyperRectangle1", hyperRectangle1)
			#print ("hyperRectangle2", hyperRectangle2)
			print ("sol found", sol)

		else:
			break

	
	'''
	CHECK FOR STABILITY OF SOLUTIONS FOUND
	'''
	sampleSols = []
	rotatedSols = {}
	stableSols = []
	unstableSols = []
	allSols = []
	for hyper in allHypers:
		exampleSoln = (hyper[:,0] + hyper[:,1])/2.0
		finalSoln = intervalUtils.newton(model,exampleSoln)
		#print "exampleSoln ", exampleSoln
		#print "finalSoln ", finalSoln
		stable = determineStability(finalSoln[1],model)
		if stable:
			stableSols.append(finalSoln[1])
		else:
			unstableSols.append(finalSoln[1])
		allSols.append(finalSoln[1])
		
		# Classify the solutions into equivalence classes
		if len(sampleSols) == 0:
			sampleSols.append(finalSoln[1])
			rotatedSols[0] = []
		else:
			foundSample = False
			for si in range(len(sampleSols)):
				sample = sampleSols[si]
				for ii in range(lenV):
					if abs(finalSoln[1][0] - sample[ii]) < 1e-8:
						rotatedSample = np.zeros_like(finalSoln[1])
						for ri in range(lenV):
							rotatedSample[ri] = sample[(ii+ri)%lenV]
						if np.less_equal(np.absolute(rotatedSample - finalSoln[1]), np.ones((lenV))*1e-8 ).all():
							foundSample = True
							rotatedSols[si].append(ii)
							break
				if foundSample:
					break

			if foundSample == False:
				sampleSols.append(finalSoln[1])
				rotatedSols[len(sampleSols)-1] = []

	for hi in range(len(sampleSols)):
		print ("equivalence class# ", hi)
		print ("main member ", sampleSols[hi])
		print ("number of other members ", len(rotatedSols[hi]))
		print ("other member rotationIndices: ")
		for mi in range(len(rotatedSols[hi])):
			print (rotatedSols[hi][mi])
		print ("")

	for hi in range(len(sampleSols)):
		if len(rotatedSols[hi]) > lenV - 1 or (len(rotatedSols[hi]) >= 1 and rotatedSols[hi][0] == 0):
			print ("problem equivalence class# ", hi)
			print ("main member ", sampleSols[hi])
			print ("num other Solutions ", len(rotatedSols[hi]))

	print ("")
	print ("numSolutions, ", len(allHypers))
	print ("num stable solutions ", len(stableSols))
	'''for si in range(len(stableSols)):
		print stableSols[si]'''
	#print "num unstable solutions ", len(unstableSols)
	'''for si in range(len(unstableSols)):
		print unstableSols[si]'''
	end = time.time()
	print ("time taken", end - start)


def rambusOscillator(a, numStages):
	global hyperNum
	hyperNum = 0

	model = tanhModel.TanhModel(modelParam = a, g_cc = 0.5, g_fwd = 1.0, numStages=numStages)
	
	#modelParam = [Vtp, Vtn, Vdd, Kn, Sn]
	#modelParam = [-0.25, 0.25, 1.0, 1.0, 1.0]
	#model = mosfetModel.MosfetModel(modelParam = modelParam, g_cc = 4.0, g_fwd = 1.0, numStages = numStages)
	
	startExp = time.time()
	lenV = numStages*2
	#exampleOrdering = []
	indexChoiceArray = []
	firstIndex = numStages - 1
	secondIndex = numStages*2 - 1
	for i in range(lenV):
		#exampleOrdering.append(0)
		#indexChoiceArray.append(i)
		if i%2 == 0:
			indexChoiceArray.append(firstIndex)
			firstIndex -= 1
		else:
			indexChoiceArray.append(secondIndex)
			secondIndex -= 1
	
	print ("indexChoiceArray", indexChoiceArray)
	boundMap = []
	parentHyper = []
	for i in range(lenV):
		boundMap.append({0:[-1.0,0.0],1:[0.0,1.0]})
		#boundMap.append({0:[0.0,0.5],1:[0.5,1.0]})
		parentHyper.append([-1.0, 1.0])
	parentHyper = np.array(parentHyper)
	#excludingBound = findExcludingBound(exampleOrdering,boundMap,model)
	#excludingBound = 0.1
	print ("boundMap ", boundMap)
	minBoundMap = 0
	maxBoundMap = 1
	rootCombinationNodes = intervalUtils.combinationWithTrees(lenV,[minBoundMap,maxBoundMap],indexChoiceArray)
	hyperBound = 0.1
	'''bisectionHypers = refineHyper(a, params, xs, ys, zs, [None, None, None, None], boundMap, hyperBound)
	print "bisectionHypers"
	print bisectionHypers'''

	'''hyperRectangle = np.zeros((lenV,2))
	hyperRectangle[0,:] = [0.0, 0.3]
	hyperRectangle[1,:] = [0.6, 1.0]
	hyperRectangle[2,:] = [0.6233348, 0.88285783]
	hyperRectangle[3,:] = [0.0, 0.3]
	hyperRectangle[4,:] = [0.6, 1.0]
	hyperRectangle[5,:] = [0.0, 0.6]
	hyperRectangle[6,:] = [0.10523252, 0.37844998]
	hyperRectangle[7,:] = [0.6, 1.0]

	feasibility = ifFeasibleHyper(hyperRectangle, hyperBound,model)
	#feasibility = intervalUtils.checkExistenceOfSolutionGS(model,hyperRectangle.transpose())
	print (feasibility)'''
	'''#exampleSoln = (hyperRectangle[:,0] + hyperRectangle[:,1])/2.0
	exampleSoln = np.array([0.83, 0.17, 0.83, 0.17])
	finalSoln = intervalUtils.newton(model,exampleSoln)
	print ("finalSoln", finalSoln)'''
	#print (model.currentFun(0.90732064, 0.09267936))
	'''ordering = [0,1,1,0,1,0,0,1]
	hypers = refineHyper(a, params, xs, ys, zs, ordering, boundMap, hyperBound)
	print "hypers"
	print hypers
	exampleSoln = (hypers[0][:,0] + hypers[0][:,1])/2.0
	finalSoln = intervalUtils.newton(a,params,exampleSoln, oscNum, getJacobian)
	print "finalSoln ", finalSoln'''
	parentTree = Tree()
	hyper_feasHyperString = np.array_str(parentHyper) + "\n ****\n" + np.array_str(parentHyper) + "\n"
	hyperId = str(hyperNum)
	hyperNum += 1
	parentTree.create_node(hyper_feasHyperString, hyperId)
	allHypers = []
	for i in range(len(rootCombinationNodes)):
		getFeasibleIntervalIndices(rootCombinationNodes[i],boundMap,hyperBound,model,allHypers, parentTree, hyperId)
	
	print ("allHypers")
	print (allHypers)
	sampleSols = []
	rotatedSols = {}
	stableSols = []
	unstableSols = []
	allSols = []
	for hyper in allHypers:
		exampleSoln = (hyper[:,0] + hyper[:,1])/2.0
		finalSoln = intervalUtils.newton(model,exampleSoln)
		#print "exampleSoln ", exampleSoln
		#print "finalSoln ", finalSoln
		stable = determineStability(finalSoln[1],model)
		if stable:
			stableSols.append(finalSoln[1])
		else:
			unstableSols.append(finalSoln[1])
		allSols.append(finalSoln[1])
		
		# Classify the solutions into equivalence classes
		if len(sampleSols) == 0:
			sampleSols.append(finalSoln[1])
			rotatedSols[0] = []
		else:
			foundSample = False
			for si in range(len(sampleSols)):
				sample = sampleSols[si]
				for ii in range(lenV):
					if abs(finalSoln[1][0] - sample[ii]) < 1e-8:
						rotatedSample = np.zeros_like(finalSoln[1])
						for ri in range(lenV):
							rotatedSample[ri] = sample[(ii+ri)%lenV]
						if np.less_equal(np.absolute(rotatedSample - finalSoln[1]), np.ones((lenV))*1e-8 ).all():
							foundSample = True
							rotatedSols[si].append(ii)
							break
				if foundSample:
					break

			if foundSample == False:
				sampleSols.append(finalSoln[1])
				rotatedSols[len(sampleSols)-1] = []

	for hi in range(len(sampleSols)):
		print ("equivalence class# ", hi)
		print ("main member ", sampleSols[hi])
		print ("number of other members ", len(rotatedSols[hi]))
		print ("other member rotationIndices: ")
		for mi in range(len(rotatedSols[hi])):
			print (rotatedSols[hi][mi])
		print ("")

	for hi in range(len(sampleSols)):
		if len(rotatedSols[hi]) > lenV - 1 or (len(rotatedSols[hi]) >= 1 and rotatedSols[hi][0] == 0):
			print ("problem equivalence class# ", hi)
			print ("main member ", sampleSols[hi])
			print ("num other Solutions ", len(rotatedSols[hi]))

	print ("")
	print ("numSolutions, ", len(allHypers))
	print ("num stable solutions ", len(stableSols))
	'''for si in range(len(stableSols)):
		print stableSols[si]'''
	#print "num unstable solutions ", len(unstableSols)
	'''for si in range(len(unstableSols)):
		print unstableSols[si]'''
	endExp = time.time()
	print ("TOTAL TIME ", endExp - startExp)
	#parentTree.show(line_type="ascii-em")


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
#rambusOscillator(-5.0,2)
z3Version(-5.0,4)

