import numpy as np
import time
import intervalUtils
from intervalBasics import *
from circuitModels import RambusTanh, RambusMosfet
from circuitModels import SchmittMosfet
from circuitModels import InverterMosfet
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
def solverLoop(uniqueHypers, model, statVars, volRedThreshold, bisectFun, numSolutions, kAlpha, hyperRectangle = None):
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
				addToSolutions(model, uniqueHypers, feasibility[1])

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
				lHyp, rHyp = bisectFun(hypForBisection)
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
							addToSolutions(model, uniqueHypers, lFeas[1])
					if rFeas[0]:
						if numSolutions == "all" or len(uniqueHypers) < numSolutions:
							addToSolutions(model, uniqueHypers, rFeas[1])

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
				addToSolutions(model, uniqueHypers, feasibility[1])

		elif feasibility[0] == False and feasibility[1] is not None:
			#If the Krawczyk loop cannot make a decision about
			#the hyperrectangle, bisect and add the two halves to
			#the stackList to be processed again.
			hypForBisection = feasibility[1]
			lHyp, rHyp = bisectFun(hypForBisection)
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
def addToSolutions(model, allHypers, solHyper):
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
def schmittTrigger(modelType, inputVoltage, statVars, numSolutions = "all", useLp = False):
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

	kAlpha = 1.0
	if useLp:
		volRedThreshold = 1.0
		solverLoop(uniqueHypers=allHypers, model=model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectMax, numSolutions=numSolutions, kAlpha=kAlpha)
	else:
		solverLoopNoLp(uniqueHypers=allHypers, model=model, statVars=statVars, bisectFun=bisectMax, numSolutions=numSolutions, kAlpha=kAlpha)

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
def inverter(modelType, inputVoltage, statVars, numSolutions="all" , useLp=False):
	global runOptions
	statVars.update({'numBisection':0, 'numLp':0, 'numK':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalKTime':0, 'totalLPTime':0, 'avgKTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})
	

	#load the inverter trigger model
	if modelType == "lcMosfet":
		#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
		modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
		model = InverterMosfet(modelType = modelType, modelParam = modelParam, inputVoltage = inputVoltage)
	if modelType == "scMosfet":
		modelParam = [1.0] #Vdd
		model = InverterMosfet(modelType = modelType, modelParam = modelParam, inputVoltage = inputVoltage)

	startExp = time.time()
	
	allHypers = []
	kAlpha = 1.0
	if useLp:
		volRedThreshold = 1.0
		solverLoop(uniqueHypers=allHypers, model=model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectMax, numSolutions=numSolutions, kAlpha=kAlpha)
	else:
		solverLoopNoLp(uniqueHypers=allHypers, model=model, statVars=statVars, bisectFun=bisectMax, numSolutions=numSolutions, kAlpha=kAlpha)
	
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

def ownCircuit():
	modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
	model = RambusLcMosfet(modelParam = modelParam, g_cc = 0.5, g_fwd = 1.0, numStages = 2)
	uniqueHypers = []
	solverLoopNoLp(uniqueHypers, model)	
	print ("uniqueHypers")
	print (uniqueHypers)


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
def rambusOscillator(modelType, numStages, g_cc, statVars, numSolutions="all", useLp=False):
	statVars.update({'numBisection':0, 'numLp':0, 'numK':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalKTime':0, 'totalLPTime':0, 'avgKTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})
	
	if modelType == "tanh":
		a = -5.0
		model = RambusTanh(modelParam = a, g_cc = g_cc, g_fwd = 1.0, numStages=numStages)
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

	kAlpha = 1.0
	if useLp:
		volRedThreshold = 1.0
		solverLoop(uniqueHypers=allHypers, model=model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectMax, numSolutions=numSolutions, kAlpha=kAlpha)
	else:
		solverLoopNoLp(uniqueHypers=allHypers, model=model, statVars=statVars, bisectFun=bisectMax, numSolutions=numSolutions, kAlpha=kAlpha)
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
	
	#print ("numBisection", statVars['numBisection'], "numLp", statVars['numLp'], "numK", statVars['numK'],
	#	"numSingleKill", statVars['numSingleKill'], "numDoubleKill", statVars['numDoubleKill'])
	#print ("totalKTime", statVars['totalKTime'], "totalLPTime", statVars['totalLPTime'], "avgKTime", 
	#	statVars['avgKTime'], "avgLPTime", statVars['avgLPTime'])
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
def singleVariableInequalities(problemType, statVars, useLp=False):
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



	kAlpha = 1.0
	allHypers = []
	if useLp:
		volRedThreshold = 1.0
		solverLoop(uniqueHypers=allHypers, model=model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectMax, numSolutions="all", kAlpha=kAlpha)
	else:
		solverLoopNoLp(uniqueHypers=allHypers, model=model, statVars=statVars, bisectFun=bisectMax, numSolutions="all", kAlpha=kAlpha)
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
	#allHypers = schmittTrigger(modelType="scMosfet", inputVoltage = 0.5, statVars=statVars, numSolutions = "all")
	#allHypers = inverter(modelType="scMosfet", inputVoltage=1.0, statVars=statVars, numSolutions="all")
	#allHypers = rambusOscillator(modelType="tanh", numStages=2, g_cc=0.5, statVars=statVars, numSolutions="all")
	#allHypers = singleVariableInequalities(problemType="flyspeck172", statVars=statVars)
	end = time.time()
	print ("allHypers", allHypers)
	print ("numSolutions", len(allHypers))
	print ("time taken", end - start)
