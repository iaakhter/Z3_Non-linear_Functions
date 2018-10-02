import numpy as np
import time
import intervalUtils
from intervalBasics import *
from circuitModels import RambusTanh, RambusMosfet
from circuitModels import SchmittMosfet
from circuitModels import InverterTanh, InverterMosfet
from circuitModels import InverterLoopTanh, InverterLoopMosfet
import flyspeckProblems
import metiProblems
import dcUtils
import random
import math
import circuit


# A mechanism to bisect a hyperrectangle
# Find the dimension in hyper with the highest
# length and bisect the hyperrectangle at that dimension
def bisectMax(hyper, options=None):
	intervalLength = hyper[:,1] - hyper[:,0]
	bisectIndex = np.argmax(intervalLength)
	#print ("bisectIndex", bisectIndex)
	lHyper = np.copy(hyper)
	rHyper = np.copy(hyper)
	midVal = (hyper[bisectIndex][0] + hyper[bisectIndex][1])/2.0
	lHyper[bisectIndex][1] = midVal
	rHyper[bisectIndex][0] = midVal
	return [lHyper, rHyper]

def bisectNewton(hyper, model):
	bisectIndex, cutoffVal = findBisectingIndexProportion(model, hyper)
	if bisectIndex is None:
		lHyp, rHyp = bisectMax(hyper)
	else:
		lHyp, rHyp = bisectAtIndex(hyper, [bisectIndex, cutoffVal])
	return [lHyp, rHyp]

#options = [bisectIndex, cutoffVal]
def bisectAtIndex(hyper, options):
	bisectIndex, cutoffVal = options
	lHyper = np.copy(hyper)
	rHyper = np.copy(hyper)
	lHyper[bisectIndex][1] = cutoffVal
	rHyper[bisectIndex][0] = cutoffVal
	return [lHyper, rHyper]

def findBisectingIndexProportion(model, hyper):
	bisectIndex, cutoffVal = None, None
	hyperDist = hyper[:,1] - hyper[:,0]
	#trialSoln = hyper[:,0] + np.multiply(np.random.rand((hyper.shape[0])), hyperDist)
	trialSoln = hyper[:,0] + 0.5*hyperDist
	finalSoln = intervalUtils.newton(model, trialSoln)
	if finalSoln[0]:
		if np.all(finalSoln[1] >= hyper[:,0]) and np.all(finalSoln[1] <= hyper[:,1]):
			distFromLow = finalSoln[1] - hyper[:,0]
			distFromHigh = hyper[:,1] - finalSoln[1]
			maxDistIndexFromLo = np.argmax(distFromLow)
			maxDistIndexFromHi = np.argmax(distFromHigh)
			if distFromLow[maxDistIndexFromLo] > distFromHigh[maxDistIndexFromHi]:
				bisectIndex = maxDistIndexFromLo
				cutoffVal = finalSoln[1][bisectIndex] - 0.3*(finalSoln[1][bisectIndex] - hyper[bisectIndex][0])
			else:
				bisectIndex = maxDistIndexFromHi		
				cutoffVal = finalSoln[1][bisectIndex] + 0.3*(hyper[bisectIndex][1] - finalSoln[1][bisectIndex])

	return bisectIndex, cutoffVal			

# solver's main loop that uses LP
# @param uniqueHypers is a list of hyperrectangle containing unique solutions
#	found by solverLoop
# @param model indicates the problem we are trying to solve rambus/schmitt/metitarski
# @param statVars holds statistical information about the operations performed by the solver.
#	For example, number of bisections, number of Lp's performed
# @param volRedThreshold is indicates the stopping criterion for the loop of
#	Krawczyk and LP (implemented by the function ifFeasibleHyper) is applied
# @param bisectFun is a function that takes in a hyperrectangle and employes some mechanism
#	to bisect it	
# @param numSolutions indicates the number of solutions wanted by the user
# @param kAlpha is the threshold which indicates the stopping criterion for the Krawczyk loop
# @param hyperRectangle the initial hyperrectangle over which the search for solutions
#	is done by solverLoop. If this argument is None then the hyperrectangle defined
#	by the bounds of the model is used
def solverLoop(uniqueHypers, model, statVars=None, volRedThreshold=1.0, bisectFun=bisectNewton, numSolutions="all", kAlpha=1.0, hyperRectangle = None):
	if statVars is None:
		statVars = {}
		statVars.update({'numBisection':0, 'numLp':0, 'numK':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalKTime':0, 'totalLPTime':0, 'avgKTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})
	lenV = len(model.bounds)
	
	if hyperRectangle is None:
		hyperRectangle = np.zeros((lenV,2))

		for i in range(lenV):
			hyperRectangle[i,0] = model.bounds[i][0]
			hyperRectangle[i,1] = model.bounds[i][1]

	
	statVars['stringHyperList'].append(("i", intervalUtils.volume(hyperRectangle)))
	
	start = time.time()
	feas = intervalUtils.checkExistenceOfSolution(model,hyperRectangle.transpose(),kAlpha)
	end = time.time()
	statVars['totalKTime'] += end - start
	statVars['numK'] += 1
	
	statVars['stringHyperList'].append(("g", intervalUtils.volume(feas[1])))


	#stack containing hyperrectangles about which any decision
	#has not been made - about whether they contain unique solution
	#or no solution
	stackList = []
	if feas[1] is not None:
		stackList.append(feas[1])

	while len(stackList) > 0:
		#pop the hyperrectangle
		#print ("len(stackList)", len(stackList))
		hyperPopped = stackList.pop(-1)
		#print ("solver loop hyperPopped")
		#intervalUtils.printHyper(hyperPopped)
		
		#if the popped hyperrectangle is contained in a hyperrectangle
		#that is already known to contain a unique solution, then do not
		#consider this hyperrectangle for the next steps
		hyperAlreadyConsidered = False
		for hyper in uniqueHypers:
			if np.greater_equal(hyperPopped[:,0], hyper[:,0]).all() and np.less_equal(hyperPopped[:,1], hyper[:,1]).all():
				hyperAlreadyConsidered = True
				break
		if hyperAlreadyConsidered:
			continue

		#Apply the Krawczyk + Lp loop
		feasibility = ifFeasibleHyper(hyperPopped, statVars, volRedThreshold, model, kAlpha)
		
		#print ("feasibility", feasibility)
		if feasibility[0]:
			#If the Krawczyk + Lp loop indicate uniqueness, then add the hyperrectangle
			#to our list
			if numSolutions == "all" or len(uniqueHypers) < numSolutions:
				addToSolutions(model, uniqueHypers, feasibility[1], kAlpha)

		elif feasibility[0] == False and feasibility[1] is not None:
			#If the Krawczyk + Lp loop cannot make a decision about
			#the hyperrectangle, the do the bisect and kill loop - keep
			#bisecting as long atleast one half either contains a unique
			#solution or no solution. Otherwise, add the two halves to
			#the stackList to be processed again.
			hypForBisection = feasibility[1]
			while hypForBisection is not None:
				#print ("hypForBisection")
				#intervalUtils.printHyper(hypForBisection)
				lHyp, rHyp = bisectFun(hypForBisection, model)
				statVars['numBisection'] += 1
				statVars['stringHyperList'].append(("b", [intervalUtils.volume(lHyp), intervalUtils.volume(rHyp)]))
				#print ("lHyp")
				#intervalUtils.printHyper(lHyp)
				start = time.time()
				lFeas = intervalUtils.checkExistenceOfSolution(model, lHyp.transpose(), kAlpha)
				end = time.time()
				#print ("lFeas", lFeas)
				statVars['totalKTime'] += end - start
				statVars['numK'] += 1
				statVars['stringHyperList'].append(("g", intervalUtils.volume(lFeas[1])))
				#print ("rHyp")
				#intervalUtils.printHyper(rHyp)
				start = time.time()
				rFeas = intervalUtils.checkExistenceOfSolution(model, rHyp.transpose(), kAlpha)
				end = time.time()
				#print ("rFeas", rFeas)
				statVars['totalKTime'] += end - start

				statVars['numK'] += 1
				statVars['stringHyperList'].append(("g", intervalUtils.volume(rFeas[1])))
				if lFeas[0] or rFeas[0] or (lFeas[0] == False and lFeas[1] is None) or (rFeas[0] == False and rFeas[1] is None):
					if lFeas[0] and rFeas[0]:
						statVars['numDoubleKill'] += 1
					elif lFeas[0] == False and lFeas[1] is None and rFeas[0] == False and rFeas[1] is None:
						statVars['numDoubleKill'] += 1
					else:
						statVars['numSingleKill'] += 1
					
					if lFeas[0]:
						if numSolutions == "all" or len(uniqueHypers) < numSolutions:
							addToSolutions(model, uniqueHypers, lFeas[1], kAlpha)
					if rFeas[0]:
						if numSolutions == "all" or len(uniqueHypers) < numSolutions:
							addToSolutions(model, uniqueHypers, rFeas[1], kAlpha)

					if lFeas[0] == False and lFeas[1] is not None:
						hypForBisection = lFeas[1]
					elif rFeas[0] == False and rFeas[1] is not None:
						hypForBisection = rFeas[1]
					else:
						hypForBisection = None

				
				else:
					stackList.append(lFeas[1])
					stackList.append(rFeas[1])
					hypForBisection = None


# solver's main loop that doesn't use LP
# @param uniqueHypers is a list of hyperrectangle containing unique solutions
#	found by solverLoop
# @param model indicates the problem we are trying to solve rambus/schmitt/metitarski
# @param statVars holds statistical information about the operations performed by the solver.
#	For example, number of bisections, number of Lp's performed
# @param bisectFun is a function that takes in a hyperrectangle and employes some mechanism
#	to bisect it	
# @param numSolutions indicates the number of solutions wanted by the user
# @param kAlpha is the threshold which indicates the stopping criterion for the Krawczyk loop
# @param hyperRectangle the initial hyperrectangle over which the search for solutions
#	is done by solverLoop. If this argument is None then the hyperrectangle defined
#	by the bounds of the model is used
def solverLoopNoLp(uniqueHypers, model, statVars=None, bisectFun=bisectMax, numSolutions="all", kAlpha=1.0, hyperRectangle = None):
	if statVars is None:
		statVars = {}
		statVars.update({'numBisection':0, 'numLp':0, 'numK':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalKTime':0, 'totalLPTime':0, 'avgKTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})
	lenV = len(model.bounds)
	
	if hyperRectangle is None:
		hyperRectangle = np.zeros((lenV,2))

		for i in range(lenV):
			hyperRectangle[i,0] = model.bounds[i][0]
			hyperRectangle[i,1] = model.bounds[i][1]

	
	statVars['stringHyperList'].append(("i", intervalUtils.volume(hyperRectangle)))
	
	#stack containing hyperrectangles about which any decision
	#has not been made - about whether they contain unique solution
	#or no solution
	stackList = [hyperRectangle]

	while len(stackList) > 0:
		#pop the hyperrectangle
		#print ("len(stackList)", len(stackList))
		hyperPopped = stackList.pop(-1)
		#print ("solver loop hyperPopped")
		#intervalUtils.printHyper(hyperPopped)
		
		#if the popped hyperrectangle is contained in a hyperrectangle
		#that is already known to contain a unique solution, then do not
		#consider this hyperrectangle for the next steps
		hyperAlreadyConsidered = False
		for hyper in uniqueHypers:
			if np.greater_equal(hyperPopped[:,0], hyper[:,0]).all() and np.less_equal(hyperPopped[:,1], hyper[:,1]).all():
				hyperAlreadyConsidered = True
				break
		if hyperAlreadyConsidered:
			continue

		start = time.time()
		feasibility = intervalUtils.checkExistenceOfSolution(model, hyperPopped.transpose(), kAlpha)
		end = time.time()
		statVars['totalKTime'] += end - start
		statVars['numK'] += 1
		statVars['stringHyperList'].append(("g", intervalUtils.volume(feasibility[1])))
		
		#print ("feasibility", feasibility)
		if feasibility[0]:
			#If the Krawczyk loop indicate uniqueness, then add the hyperrectangle
			#to our list
			if numSolutions == "all" or len(uniqueHypers) < numSolutions:
				#print ("solution found")
				#print ("hyper")
				#intervalUtils.printHyper(hyperPopped)
				#print ("feas")
				#intervalUtils.printHyper(feasibility[1])
				addToSolutions(model, uniqueHypers, feasibility[1], kAlpha)

		elif feasibility[0] == False and feasibility[1] is not None:
			#If the Krawczyk loop cannot make a decision about
			#the hyperrectangle, bisect and add the two halves to
			#the stackList to be processed again.
			hypForBisection = feasibility[1]
			lHyp, rHyp = bisectFun(hypForBisection, model)
			statVars['numBisection'] += 1
			statVars['stringHyperList'].append(("b", [intervalUtils.volume(lHyp), intervalUtils.volume(rHyp)]))
			stackList.append(lHyp)
			stackList.append(rHyp)


# Apply Krawczyk and linear programming to refine the hyperrectangle
# @param hyperRectangle 
# @param statVars statVars holds statistical information about the operations performed by the solver.
#	For example, number of Lp's performed
# @param volRedTheshold indicates the stopping criterion for the loop of
#	Krawczyk and LP (implemented by this function). Basically keep repeating the loop
#	as long as the percentage of volume reduction of the hyperrectangle is atleast volRedThreshold. 
# @param model indicates the problem we are trying to solve
# @param kAlpha is the threshold which indicates the stopping criterion for the Krawczyk loop
# @return (True, hyper) if hyperRectangle contains a unique
# 	solution and hyper maybe smaller than hyperRectangle containing the solution
# @return (False, None) if hyperRectangle contains no solution
# @return (False, hyper) if hyperRectangle may contain more
# 	than 1 solution and hyper maybe smaller than hyperRectangle containing the solutions
def ifFeasibleHyper(hyperRectangle, statVars, volRedThreshold, model, kAlpha):
	lenV = hyperRectangle.shape[0]
	iterNum = 0
	while True:
		newHyperRectangle = np.copy(hyperRectangle)

		# Apply linear programming step
		start = time.time()
		feasible, newHyperRectangle, numTotalLp, numSuccessLp, numUnsuccessLp = model.linearConstraints(newHyperRectangle)
		end = time.time()
		statVars['totalLPTime'] += end - start
		statVars['numLpCalls'] += numTotalLp
		statVars['numSuccessLpCalls'] += numSuccessLp
		statVars['numUnsuccessLpCalls'] += numUnsuccessLp
		statVars['numLp'] += 1
		if feasible:
			vol = intervalUtils.volume(newHyperRectangle)
		else:
			vol = None
		statVars['stringHyperList'].append(("l", vol))
		'''print ("newHyperRectangle", newHyperRectangle)
		intervalUtils.printHyper(newHyperRectangle'''
		if feasible == False:
			return (False, None)

		for i in range(lenV):
			if newHyperRectangle[i,0] < hyperRectangle[i,0]:
				newHyperRectangle[i,0] = hyperRectangle[i,0]
			if newHyperRectangle[i,1] > hyperRectangle[i,1]:
				newHyperRectangle[i,1] = hyperRectangle[i,1]


		start = time.time()
		
		#Apply Krawczyk
		kResult = intervalUtils.checkExistenceOfSolution(model, newHyperRectangle.transpose(), kAlpha)
		end = time.time()
		statVars['totalKTime'] += (end - start)
		statVars['numK'] += 1
		statVars['stringHyperList'].append(("g", intervalUtils.volume(kResult[1])))
		
		#Unique solution or no solution
		if kResult[0] or kResult[1] is None:
			#print ("uniqueHyper", hyperRectangle)
			return kResult

		newHyperRectangle = kResult[1]			 

		hyperVol = intervalUtils.volume(hyperRectangle)

		newHyperVol = intervalUtils.volume(newHyperRectangle)

		propReduc = (hyperVol - newHyperVol)/hyperVol
		
		# If the proportion of volume reduction is not atleast
		# volRedThreshold then return
		if math.isnan(propReduc) or propReduc <= volRedThreshold:
			return (False, newHyperRectangle)
		hyperRectangle = newHyperRectangle
		iterNum+=1
	


# A function that adds a new solution to the list of existing hyperrectangles
# containing unique solutions if the new hyperrectangle does not contain
# the same solution as the solution in any of the existing hyperrectangles
# @param model indicates the problem we are trying to solve rambus/schmitt/metitarski
# @param allHypers list of hyperrectangles containing unique solutions
# @param solHyper new hyperrectangle containing unique solution
def addToSolutions(model, allHypers, solHyper, kAlpha):
	epsilon = 1e-12
	lenV = len(model.bounds)
	foundOverlap = False
	exampleVolt = (solHyper[:,0] + solHyper[:,1])/2.0
	soln = intervalUtils.newton(model,exampleVolt)
	
	if not(soln[0]):
		raise Exception("prototype.py addToSolutions: Something went wrong. Should contain a unique solution")
	
	for hi in range(len(allHypers)):
		oldHyper = allHypers[hi]

		#Check if solHyper overlaps with oldHyper
		if all(interval_intersect(solHyper[i], oldHyper[i]) is not None for i in range(lenV)):
			intersectHyper = np.zeros((lenV,2))
			for ui in range(lenV):
				intersectHyper[ui,:] = interval_intersect(solHyper[ui], oldHyper[ui])

			if np.all(soln[1] >= intersectHyper[:,0]) and np.all(soln[1] <= intersectHyper[:,1]):
				hyperAroundNewton = np.zeros((lenV, 2))
				for si in range(lenV):
					minDiff = min(abs(intersectHyper[si,1] - soln[1][si]), abs(soln[1][si] - intersectHyper[si,0]))
					hyperAroundNewton[si,0] = soln[1][si] - minDiff
					hyperAroundNewton[si,1] = soln[1][si] + minDiff
				feasibility = intervalUtils.checkExistenceOfSolution(model, hyperAroundNewton.transpose(), alpha = kAlpha)
				if feasibility[0]:
					foundOverlap = True
					break

	if not(foundOverlap):
		allHypers.append(solHyper)
		return True
	else:
		return False

def addToSolutionsMaximal(model, allHypers, solHyper):
	lenV = len(model.bounds)
	allHypersCopy = [hyper for hyper in allHypers]
	foundOverlap = False
	for hi in range(len(allHypers)):
		oldHyper = allHypers[hi]

		#Check if solHyper overlaps with oldHyper
		if all(interval_intersect(solHyper[i], oldHyper[i]) is not None for i in range(lenV)):
			unionHyper = np.zeros((lenV,2))
			for ui in range(lenV):
				unionHyper[ui,:] = interval_union(solHyper[ui], oldHyper[ui])

			feasibility = intervalUtils.checkExistenceOfSolution(model, unionHyper.transpose())
			if feasibility[0]:
				foundOverlap = True
				allHypers[hi] = unionHyper
				break

	if not(foundOverlap):
		trialSoln = (solHyper[:,0] + solHyper[:,1])/2.0
		finalSoln = intervalUtils.newton(model,trialSoln)
		hyperWithUniqueSoln = np.zeros((lenV,2))
		maxDiff = np.maximum(np.absolute(solHyper[:,1] - finalSoln[1]), np.absolute(finalSoln[1] - solHyper[:,0]))
		hyperWithUniqueSoln[:,0] = finalSoln[1] - maxDiff
		hyperWithUniqueSoln[:,1] = finalSoln[1] + maxDiff
		kResult = intervalUtils.checkExistenceOfSolution(model,hyperWithUniqueSoln.transpose())
		if not(kResult[0]):
			allHypers.append(solHyper)
			return True
		diff = maxDiff*4.0
		startingIndex = 0
		foundUniqueHyper = False
		while True:
			#print ("diff", diff)
			hyperWithUniqueSoln[:,0] = finalSoln[1] - maxDiff - diff
			hyperWithUniqueSoln[:,1] = finalSoln[1] + maxDiff + diff
			kResult = intervalUtils.checkExistenceOfSolution(model,hyperWithUniqueSoln.transpose())
			if kResult[0] == False and kResult[1] is not None:
				diff[startingIndex] = diff[startingIndex]/2.0
				startingIndex = (startingIndex + 1)%lenV
			elif kResult[0]:
				foundUniqueHyper = True
				#print ("Preprocess: found unique hyper ", hyperWithUniqueSoln)
				#print ("hyper before", solHyper)
				numFailures = 0
				allHypers.append(hyperWithUniqueSoln)
				break
		if not(foundUniqueHyper):
			allHypers.append(solHyper)
		return True
	else:
		return False


# Find the dc equilibrium points for the schmitt trigger for a specific
# input voltage
# @param modelType indicates the type of transistor model used for the schmitt
#	trigger. If modelType == "lcMosfet", use the long channel mosfet model.
#	If modelType == "scMosfet", use the short channel mosfet model
# @param inputVoltage the value of the specific input voltage for which 
# 	the dc equilibrium points are found
# @param statVars dictionary to hold statistics like number of bisections, number of Krawczyk calls
# @param numSolutions, number of dc equilibrium points we are looking for
# @param useLp flag to decide whether to use linear programming in our method or not
# @return a list of hyperrectangles containing unique dc equilibrium points
def schmittTrigger(modelType, inputVoltage, statVars, kAlpha = 1.0, bisectType="bisectNewton", numSolutions = "all", useLp = False):
	statVars.update({'numBisection':0, 'numLp':0, 'numK':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalKTime':0, 'totalLPTime':0, 'avgKTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})

	#load the schmitt trigger model
	if modelType == "lcMosfet":
		#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
		modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
		model = SchmittMosfet(modelType = modelType, modelParam = modelParam, inputVoltage = inputVoltage)
	elif modelType == "scMosfet":
		modelParam = [1.0] #Vdd
		model = SchmittMosfet(modelType = modelType, modelParam = modelParam, inputVoltage = inputVoltage)

	startExp = time.time()

	allHypers = []

	if bisectType == "bisectMax":
		bisectFun = bisectMax
	if bisectType == "bisectNewton":
		bisectFun = bisectNewton
	if useLp:
		volRedThreshold = 1.0
		solverLoop(uniqueHypers=allHypers, model=model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectFun, numSolutions=numSolutions, kAlpha=kAlpha)
	else:
		solverLoopNoLp(uniqueHypers=allHypers, model=model, statVars=statVars, bisectFun=bisectFun, numSolutions=numSolutions, kAlpha=kAlpha)

	#print ("allHypers")
	#print (allHypers)
	#print ("numSolutions", len(allHypers))

	
	#dcUtils.printSol(allHypers, model)
	endExp = time.time()
	#print ("TOTAL TIME ", endExp - startExp)
	if statVars['numLp'] != 0:
		statVars['avgLPTime'] = (statVars['totalLPTime']*1.0)/statVars['numLp']
	if statVars['numK'] != 0:
		statVars['avgKTime'] = (statVars['totalKTime']*1.0)/statVars['numK']

	#print ("numBisection", statVars['numBisection'], "numLp", statVars['numLp'], "numK", statVars['numK'],
	#	"numSingleKill", statVars['numSingleKill'], "numDoubleKill", statVars['numDoubleKill'])
	#print ("totalKTime", statVars['totalKTime'], "totalLPTime", statVars['totalLPTime'], "avgKTime", 
	#	statVars['avgKTime'], "avgLPTime", statVars['avgLPTime'])
	#print ("numLpCalls", statVars['numLpCalls'], "numSuccessLpCalls", statVars['numSuccessLpCalls'], "numUnsuccessLpCalls", statVars['numUnsuccessLpCalls'])
	return allHypers

# Find the dc equilibrium points for an inverter for a specific
# input voltage
# @param modelType indicates the type of transistor model used for the inverter. 
# If modelType == "lcMosfet", use the long channel mosfet model.
#	If modelType == "scMosfet", use the short channel mosfet model
# @param inputVoltage the value of the specific input voltage for which 
# 	the dc equilibrium points are found
# @param statVars dictionary to hold statistics like number of bisections, number of Krawczyk calls
# @param numSolutions, number of dc equilibrium points we are looking for
# @param useLp flag to decide whether to use linear programming in our method or not
# @return a list of hyperrectangles containing unique dc equilibrium points
def inverter(modelType, inputVoltage, statVars, kAlpha=1.0, bisectType="bisectNewton", numSolutions="all" , useLp=False):
	statVars.update({'numBisection':0, 'numLp':0, 'numK':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalKTime':0, 'totalLPTime':0, 'avgKTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})
	

	#load the inverter model
	if modelType == "tanh":
		modelParam = [-5.0, 0.0] # y = tanh(modelParam[0]*x + modelParam[1])
		model = InverterTanh(modelParam = modelParam, inputVoltage = inputVoltage)
	if modelType == "lcMosfet":
		#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
		modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
		model = InverterMosfet(modelType = modelType, modelParam = modelParam, inputVoltage = inputVoltage)
	if modelType == "scMosfet":
		modelParam = [1.0] #Vdd
		model = InverterMosfet(modelType = modelType, modelParam = modelParam, inputVoltage = inputVoltage)

	startExp = time.time()
	
	allHypers = []
	if bisectType == "bisectMax":
		bisectFun = bisectMax
	if bisectType == "bisectNewton":
		bisectFun = bisectNewton
	if useLp:
		volRedThreshold = 1.0
		solverLoop(uniqueHypers=allHypers, model=model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectFun, numSolutions=numSolutions, kAlpha=kAlpha)
	else:
		solverLoopNoLp(uniqueHypers=allHypers, model=model, statVars=statVars, bisectFun=bisectFun, numSolutions=numSolutions, kAlpha=kAlpha)
	
	#print ("allHypers")
	#print (allHypers)
	#print ("numSolutions", len(allHypers))
	
	endExp = time.time()
	#print ("TOTAL TIME ", endExp - startExp)

	if statVars['numLp'] != 0:
		statVars['avgLPTime'] = (statVars['totalLPTime']*1.0)/statVars['numLp']
	if statVars['numK'] != 0:
		statVars['avgKTime'] = (statVars['totalKTime']*1.0)/statVars['numK']
	
	#print ("numBisection", statVars['numBisection'], "numLp", statVars['numLp'], "numK", statVars['numK'],
	#	"numSingleKill", statVars['numSingleKill'], "numDoubleKill", statVars['numDoubleKill'])
	#print ("totalKTime", statVars['totalKTime'], "totalLPTime", statVars['totalLPTime'], "avgKTime", 
	#	statVars['avgKTime'], "avgLPTime", statVars['avgLPTime'])
	#print ("numLpCalls", statVars['numLpCalls'], "numSuccessLpCalls", statVars['numSuccessLpCalls'], "numUnsuccessLpCalls", statVars['numUnsuccessLpCalls'])
	return allHypers

def inverterLoop(modelType, numInverters, statVars, kAlpha=1.0, bisectType="bisectNewton", numSolutions="all" , useLp=False):
	statVars.update({'numBisection':0, 'numLp':0, 'numK':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalKTime':0, 'totalLPTime':0, 'avgKTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})
	

	#load the inverter model
	if modelType == "tanh":
		modelParam = [-5.0, 0.0] # y = tanh(modelParam[0]*x + modelParam[1])
		model = InverterLoopTanh(modelParam = modelParam, numInverters = numInverters)
	if modelType == "lcMosfet":
		#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
		modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
		model = InverterLoopMosfet(modelType = modelType, modelParam = modelParam, numInverters = numInverters)
	if modelType == "scMosfet":
		modelParam = [1.0] #Vdd
		model = InverterLoopMosfet(modelType = modelType, modelParam = modelParam, numInverters = numInverters)

	startExp = time.time()
	
	allHypers = []
	if bisectType == "bisectMax":
		bisectFun = bisectMax
	if bisectType == "bisectNewton":
		bisectFun = bisectNewton
	if useLp:
		volRedThreshold = 1.0
		solverLoop(uniqueHypers=allHypers, model=model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectFun, numSolutions=numSolutions, kAlpha=kAlpha)
	else:
		solverLoopNoLp(uniqueHypers=allHypers, model=model, statVars=statVars, bisectFun=bisectFun, numSolutions=numSolutions, kAlpha=kAlpha)
	
	#print ("allHypers")
	#print (allHypers)
	#print ("numSolutions", len(allHypers))
	
	endExp = time.time()
	#print ("TOTAL TIME ", endExp - startExp)

	if statVars['numLp'] != 0:
		statVars['avgLPTime'] = (statVars['totalLPTime']*1.0)/statVars['numLp']
	if statVars['numK'] != 0:
		statVars['avgKTime'] = (statVars['totalKTime']*1.0)/statVars['numK']
	
	#print ("numBisection", statVars['numBisection'], "numLp", statVars['numLp'], "numK", statVars['numK'],
	#	"numSingleKill", statVars['numSingleKill'], "numDoubleKill", statVars['numDoubleKill'])
	#print ("totalKTime", statVars['totalKTime'], "totalLPTime", statVars['totalLPTime'], "avgKTime", 
	#	statVars['avgKTime'], "avgLPTime", statVars['avgLPTime'])
	#print ("numLpCalls", statVars['numLpCalls'], "numSuccessLpCalls", statVars['numSuccessLpCalls'], "numUnsuccessLpCalls", statVars['numUnsuccessLpCalls'])
	return allHypers



# Find the dc equilibrium points for a rambus ring oscillator
# @param modelType indicates the type of inverter used in the rambus oscillator
# 	If modelType == "tanh", use the tanh inverter model.
#	If modelType == "lcMosfet", use transistor with two long channel mosfet models.
#	If modelType == "scMosfet", use transistor with two short channel mosfet models.
# @param numStages the number of stages in the rambus ring oscillator
# @param g_cc strength of the cross coupled inverter (as compared to that of the forward)
# @param statVars dictionary to hold statistics like number of bisections, number of Krawczyk calls
# @param numSolutions, number of dc equilibrium points we are looking for
# @param useLp flag to decide whether to use linear programming in our method or not
# @return a list of hyperrectangles containing unique dc equilibrium points
def rambusOscillator(modelType, numStages, g_cc, statVars, kAlpha=1.0, bisectType="bisectNewton", numSolutions="all", useLp=False):
	statVars.update({'numBisection':0, 'numLp':0, 'numK':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalKTime':0, 'totalLPTime':0, 'avgKTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})
	
	if modelType == "tanh":
		modelParam = [-5.0, 0.0] # y = tanh(modelParam[0]*x + modelParam[1])
		model = RambusTanh(modelParam = modelParam, g_cc = g_cc, g_fwd = 1.0, numStages=numStages)
	elif modelType == "lcMosfet":
		#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
		#modelParam = [-0.25, 0.25, 1.0, 1.0, -0.5, 1.0]
		#modelParam = [-0.4, 0.4, 1.8, 1.5, -0.5, 8/3.0]
		modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
		model = RambusMosfet(modelType = modelType, modelParam = modelParam, g_cc = g_cc, g_fwd = 1.0, numStages = numStages)	
	elif modelType == "scMosfet":
		modelParam = [1.0] #Vdd
		model = RambusMosfet(modelType = modelType, modelParam = modelParam, g_cc = g_cc, g_fwd = 1.0, numStages = numStages)

	startExp = time.time()
	
	allHypers = []

	hyperRectangle = np.array([[1.35, 1.8],
								[0.0, 0.45],
								[0.0, 0.45],
								[0.0, 0.45],
								[0.0, 0.45],
								[1.35, 1.8],
								[1.35, 1.8],
								[1.35, 1.8]])

	'''oldHyper = np.array([[1.7999999999999938, 1.8000000000000063],
							[0.8978237347799127, 1.1934092303556052],
							[0.8620895839164895, 0.9773303624346581],
							[1.754075960482887, 1.7773879210140902],
							[-8.433296759887354e-15, 8.433296776059946e-15],
							[0.38998925796262235, 0.7015147520267985],
							[0.7300534002279107, 0.8658528592868592],
							[0.01301797075972959, 0.0285804257510049]])
	newHyper = np.array([[1.7999999999935017, 1.8000000000065006],
							[0.9992885151580121, 1.0919496051353852],
							[0.8961014115826367, 0.943319044577329],
							[1.7063977960810166, 1.8250794892834268],
							[-2.3311758512155914e-12, 2.328446598938295e-12],
							[0.4957291506785126, 0.5957701643559664],
							[0.7365923400673265, 0.8593121406942096],
							[-0.052841301644077486, 0.09443471872412834]])
	oldHyperList = [oldHyper]

	addToSolutions(model, oldHyperList, newHyper, kAlpha)
	print ("oldHyperList")
	print (oldHyperList)'''

	'''hyperRectangle = np.array([[0.0, 1.8],
								[0.0, 0.9],
								[0.0, 0.9],
								[0.0, 1.8],
								[0.0, 0.9],
								[0.0, 0.9],
								[0.0, 0.9],
								[0.0, 0.9]])
	kResult = intervalUtils.checkExistenceOfSolution(model, oldHyper.transpose(),alpha=kAlpha)
	print ("kResult old", kResult[0])
	intervalUtils.printHyper(kResult[1])
	print ("current")
	intervalUtils.printHyper(model.f(kResult[1]))
	kResult = intervalUtils.checkExistenceOfSolution(model, newHyper.transpose(),alpha=kAlpha)
	print ("kResult new", kResult[0])
	intervalUtils.printHyper(kResult[1])
	print ("current")
	intervalUtils.printHyper(model.f(kResult[1]))'''
	if bisectType == "bisectMax":
		bisectFun = bisectMax
	if bisectType == "bisectNewton":
		bisectFun = bisectNewton
	if useLp:
		volRedThreshold = 1.0
		solverLoop(uniqueHypers=allHypers, model=model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectFun, numSolutions=numSolutions, kAlpha=kAlpha)
	else:
		solverLoopNoLp(uniqueHypers=allHypers, model=model, statVars=statVars, bisectFun=bisectFun, numSolutions=numSolutions, kAlpha=kAlpha)
	
	#print ("allHypers")
	#print (allHypers)
	#dcUtils.printSol(allHypers, model)
	#print ("numSolutions", len(allHypers))

	endExp = time.time()
	#print ("TOTAL TIME ", endExp - startExp)

	if statVars['numLp'] != 0:
		statVars['avgLPTime'] = (statVars['totalLPTime']*1.0)/statVars['numLp']
	if statVars['numK'] != 0:
		statVars['avgKTime'] = (statVars['totalKTime']*1.0)/statVars['numK']
	
	print ("numBisection", statVars['numBisection'], "numLp", statVars['numLp'], "numK", statVars['numK'],
		"numSingleKill", statVars['numSingleKill'], "numDoubleKill", statVars['numDoubleKill'])
	print ("totalKTime", statVars['totalKTime'], "totalLPTime", statVars['totalLPTime'], "avgKTime", 
		statVars['avgKTime'], "avgLPTime", statVars['avgLPTime'])
	#print ("numLpCalls", statVars['numLpCalls'], "numSuccessLpCalls", statVars['numSuccessLpCalls'], "numUnsuccessLpCalls", statVars['numUnsuccessLpCalls'])
	return allHypers

# Find valid intervals given inequalities of a single variable and the
# solutions for f(x) = 0
# @param allHypers list of hyperrectangles containing unique solution
#	to f(x) = 0
# @param model class representing the inequality we are trying to solve
def validSingleVariableInterval(allHypers, model):
	xBound = [model.bounds[0][0], model.bounds[0][1]]
	sampleSols, rotatedSols, stableSols, unstableSols = dcUtils.categorizeSolutions(allHypers,model)
	
	print ("numSolutions, ", len(allHypers))

	inequalityIntervals = {" > ": [], " < ": []}

	if len(allHypers) == 0:
		xRand = random.random()*(xBound[1] - xBound[0]) + xBound[0]
		xRandVal = model.f(np.array([xRand]))
		if xRandVal > 0:
			inequalityIntervals[" > "].append(xBound)
		else:
			inequalityIntervals[" < "].append(xBound)

	else:
		#print ("sampleSols", sampleSols)
		sampleSols.append(np.array([xBound[0]]))
		sampleSols.append(np.array([xBound[1]]))
		allSampleSols = np.sort(sampleSols, axis=None)
		for si in range(len(allSampleSols)-1):
			startInterval = allSampleSols[si]
			endInterval = allSampleSols[si + 1]
			
			xRand = random.random()*(endInterval - startInterval) + startInterval
			xRandVal = model.f(np.array([xRand]))
			#print ("startInterval", startInterval, "endInterval", endInterval)
			#print ("xRand", xRand, "xRandVal", xRandVal)
			if xRandVal > 0:
				inequalityIntervals[" > "].append([startInterval, endInterval])
			else:
				inequalityIntervals[" < "].append([startInterval, endInterval])



	if model.sign == " > ":
		#print ("valid intervals", inequalityIntervals[" > "])
		return inequalityIntervals[" > "]

	elif model.sign == " < ":
		#print ("valid intervals", inequalityIntervals[" < "])
		return inequalityIntervals[" < "]



# Find valid intervals for inequalitites - solving benchmark problems
# from flyspeck and metatarski project
# @param problemType indicates the type of inequality we are trying to solve
# 	If problemType == "flyspeck172", solve the flyspeck problem number 172.
#	If problemType == "meti25", solve the metittarski problem number 25.
#	If problemType == "meti18", solve the metittarski problem number 18.
# @param statVars dictionary to hold statistics like number of bisections, number of Krawczyk calls
# @param useLp flag to decide whether to use linear programming in our method or not
# @return a list of hyperrectangles containing valid intervals for the inequality
def singleVariableInequalities(problemType, statVars, kAlpha=1.0, bisectType="bisectNewton", useLp=False):
	statVars.update({'numBisection':0, 'numLp':0, 'numK':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalKTime':0, 'totalLPTime':0, 'avgKTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})

	model, xBound = None, None
	startExp = time.time()
	
	if problemType == "flyspeck172":
		#xBound = [-100.0, -0.001]
		xBound = [3.0, 64.0]
		model = flyspeckProblems.Flyspeck172(xBound[0], xBound[1], " > ")

	if problemType == "meti25":
		#xBound = [0.0,7*math.pi]
		xBound = [math.pi/3.0, (2*math.pi/3.0)]
		model = metiProblems.Meti25(xBound[0], xBound[1], " > ")

	if problemType == "meti18":
		#xBound = [-1.0, 2.0]
		xBound = [0.0, 100.0/201]
		model = metiProblems.Meti18(xBound[0], xBound[1], " > ")



	if bisectType == "bisectMax":
		bisectFun = bisectMax
	if bisectType == "bisectNewton":
		bisectFun = bisectNewton
	allHypers = []
	if useLp:
		volRedThreshold = 1.0
		solverLoop(uniqueHypers=allHypers, model=model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectFun, numSolutions="all", kAlpha=kAlpha)
	else:
		solverLoopNoLp(uniqueHypers=allHypers, model=model, statVars=statVars, bisectFun=bisectFun, numSolutions="all", kAlpha=kAlpha)
	print ("allHypers")
	print (allHypers)
	print ("numSolutions", len(allHypers))
	
	validIntervals = validSingleVariableInterval(allHypers, model)
	print ("validIntervals", validIntervals)
	endExp = time.time()
	#print ("time taken", endExp - startExp)

	if statVars['numLp'] != 0:
		statVars['avgLPTime'] = (statVars['totalLPTime']*1.0)/statVars['numLp']
	if statVars['numK'] != 0:
		statVars['avgKTime'] = (statVars['totalKTime']*1.0)/statVars['numK']
	return validIntervals



if __name__ == "__main__":
	statVars = {}
	start = time.time()
	#allHypers = schmittTrigger(modelType="scMosfet", inputVoltage = 0.0, statVars=statVars, numSolutions = "all")
	#allHypers = inverter(modelType="tanh", inputVoltage=1.0, statVars=statVars, numSolutions="all")
	#allHypers = inverterLoop(modelType="scMosfet", numInverters=1, statVars=statVars, numSolutions="all")
	allHypers = rambusOscillator(modelType="scMosfet", numStages=4, g_cc=4.0, statVars=statVars, kAlpha = 1.0, numSolutions="all", bisectType="bisectNewton")
	#allHypers = singleVariableInequalities(problemType="flyspeck172", statVars=statVars)
	#ownCircuit()
	end = time.time()
	'''print ("allHypers")
	for hyper in allHypers:
		print ("hyper")
		intervalUtils.printHyper(hyper)'''
	print ("numSolutions", len(allHypers))
	print ("time taken", end - start)
