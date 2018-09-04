import numpy as np
import time
import intervalUtils
from intervalBasics import *
from circuitModels import RambusTanh, RambusMosfetMark, RambusStMosfet
from circuitModels import SchmittMosfetMark, SchmittStMosfet
from circuitModels import InverterStMosfet, InverterMosfet
import flyspeckProblems
import metiProblems
import rambusUtils as rUtils
import random
import math
import circuit
import itertools


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


# Divide bigHyper into smaller spaces such that
# smallHyper is one of the smaller spaces
# Make sure that none of the smaller hypers overlap
# one another
def separateHyperSpace(bigHyper, smallHyper):
	lenV = bigHyper.shape[0]
	startHyper = np.copy(bigHyper)
	smallerHyperList = []

	allRanges = []
	for i in range(lenV):
		allRanges.append([[bigHyper[i,0], smallHyper[i,0]], 
							[smallHyper[i,0], smallHyper[i,1]], 
							[smallHyper[i,1], bigHyper[i,1]]])

	numDiv = [0,1,2]
	perms = list(itertools.product(numDiv, repeat=lenV))
	print ("len(perms)", len(perms))

	for pi in range(len(perms)):
		if pi!= len(perms)//2:
			perm = perms[pi]
			#print ("perm", perm)
			hyper = np.zeros((lenV,2))
			for i in range(lenV):
				#print ("perm at i", perm[i])
				hyper[i,:] = allRanges[i][perm[i]]
			smallerHyperList.append(hyper)

	"""for i in range(lenV):
		rangei = allRanges[i]
	for i in range(lenV):
		indexOfSeparation = i
		'''hyper1 = np.copy(startHyper)
		hyper1[indexOfSeparation,1] = smallHyper[indexOfSeparation, 0]
		hyper2 = np.copy(startHyper)
		hyper2[indexOfSeparation,0] = smallHyper[indexOfSeparation, 1]
		smallerHyperList.append(hyper1)
		smallerHyperList.append(hyper2)
		startHyper[indexOfSeparation,:] = smallHyper[indexOfSeparation,:]'''"""

	print ("numSmallerHypers", len(smallerHyperList))
	return smallerHyperList

# Try to find one solution in hyper using Newton's method
def findNewtonSol(model, uniqueHypers, hyper, numTrials = 100):
	lenV = hyper.shape[0]
	hyperRange = hyper[:,1] - hyper[:,0]
	
	#print ("numTrials in newtons preprocessing", numTrials)
	start = time.time()
	numFailures = 0
	newSol = False
	for n in range(numTrials):
		if newSol:
			break
		numFailures += 1
		trialSoln = np.multiply(np.random.rand((lenV)), hyperRange) + hyper[:,0]
		finalSoln = intervalUtils.newton(model,trialSoln)
		if finalSoln[0] and np.greater_equal(finalSoln[1], hyper[:,0]).all() and np.less_equal(finalSoln[1], hyper[:,1]).all():
			solAlreadyFound = False
			for oldSol in uniqueHypers:
				#print ("finalSoln[1]", finalSoln[1])
				#print ("oldSol", oldSol[0])
				if np.all(finalSoln[1] >= oldSol[0][:,0]) and np.all(finalSoln[1] <= oldSol[0][:,1]):
					solAlreadyFound = True
					break
			if solAlreadyFound:
				continue
			hyperWithUniqueSoln = np.zeros((lenV,2))
			diff = np.ones((lenV))*0.01
			startingIndex = 0
			while True:
				#print ("diff", diff)
				hyperWithUniqueSoln[:,0] = finalSoln[1] - diff
				hyperWithUniqueSoln[:,1] = finalSoln[1] + diff
				kResult = intervalUtils.checkExistenceOfSolution(model,hyperWithUniqueSoln.transpose())
				if kResult[0] == False and kResult[1] is not None:
					diff[startingIndex] = diff[startingIndex]/2.0
					startingIndex = (startingIndex + 1)%lenV
				else:
					end = time.time()
					#print ("Preprocess: found unique hyper ", hyperWithUniqueSoln, "in", end - start, "after", numFailures, "failures")
					start = time.time()
					numFailures = 0
					hyperSol = [hyperWithUniqueSoln, "1dim"]
					newSol = addToSolutions(model, uniqueHypers, hyperSol)
					break
	return newSol
	



def solverLoop2(uniqueHypers, model, statVars, volRedThreshold, bisectFun, numSolutions = "all", useLp=True, kAlpha = 1.0):
	lenV = len(model.bounds)
	hyperRectangle = np.zeros((lenV, 2))
	for i in range(lenV):
		hyperRectangle[i,0] = model.bounds[i][0]
		hyperRectangle[i,1] = model.bounds[i][1]

	
	stackHyper = [hyperRectangle]
	while len(stackHyper) > 0:
		hyperPopped = stackHyper.pop()
		print ("hyperPopped", hyperPopped)
		feas = intervalUtils.checkExistenceOfSolution(model,hyperPopped.transpose(),kAlpha)
		if feas[1] is None:
			continue
		foundNewSol = findNewtonSol(model, uniqueHypers, hyperPopped, numTrials = 10)
		print ("foundNewSol", foundNewSol)
		#print ("uniqueHypers", uniqueHypers)
		if foundNewSol:
			#print ("found newton sol")
			#print ("newtonSol", uniqueHypers[-1])
			smallerHyperList = separateHyperSpace(hyperPopped, np.copy(uniqueHypers[-1][0]))
			#print (smallerHyperList)
			#print ("hyperRectangle")
			stackHyper += smallerHyperList
		else:
			solverLoop(uniqueHypers = uniqueHypers, model = model, statVars = statVars, 
				volRedThreshold= volRedThreshold, bisectFun = bisectFun, 
				numSolutions = numSolutions, useLp=useLp, kAlpha = kAlpha, hyperRectangle = np.copy(hyperPopped))




#solver's main loop
# @param uniqueHypers is a list of hyperrectangle containing unique solutions
#	found by solverLoop
# @param model indicates the problem we are trying to solve rambus/schmitt/metitarski
# @param statVars holds statistical information about the operations performed by the solver.
#	For example, number of bisections, number of Lp's performed
# @param volRedThreshold is indicates the stopping criterion for the loop of
#	Krawczyk and LP (implemented by the function ifFeasibleHyper) is applied
# @param bisectFun is a function that takes in a hyperrectangle and employes some mechanism
#	to besict it	
# @param numSolutions indicates the number of solutions wanted by the user
# @param useLp is a flag to indicate if LP is going to be used by the solver
# @param kAlpha is the threshold which indicates the stopping criterion for the Krawczyk loop
# @param hyperRectangle the initial hyperrectangle over which the search for solutions
#	is done by solverLoop. If this argument is None then the hyperrectangle defined
#	by the bounds of the model is used
def solverLoop(uniqueHypers, model, statVars, volRedThreshold, bisectFun, numSolutions = "all", useLp=True, kAlpha = 1.0, hyperRectangle = None):
	lenV = len(model.bounds)
	
	if hyperRectangle is None:
		hyperRectangle = np.zeros((lenV,2))

		for i in range(lenV):
			hyperRectangle[i,0] = model.bounds[i][0]
			hyperRectangle[i,1] = model.bounds[i][1]

	#print ("solver loop hyper", hyperRectangle)
	
	statVars['stringHyperList'].append(("i", intervalUtils.volume(hyperRectangle)))
	
	start = time.time()
	feas = intervalUtils.checkExistenceOfSolution(model,hyperRectangle.transpose(),kAlpha)
	#print ("feas", feas)
	end = time.time()
	statVars['totalGSTime'] += end - start
	statVars['numGs'] += 1
	
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
			if np.greater_equal(hyperPopped[:,0], hyper[0][:,0]).all() and np.less_equal(hyperPopped[:,1], hyper[0][:,1]).all():
				hyperAlreadyConsidered = True
				break
		if hyperAlreadyConsidered:
			continue

		#Apply the Krawczyk + Lp loop
		feasibility = ifFeasibleHyper(hyperPopped, statVars, volRedThreshold, model, kAlpha, useLp)
		
		#print ("feasibility", feasibility)
		if feasibility[0]:
			#If the Krawczyk + Lp loop indicate uniqueness, then add the hyperrectangle
			#to our list
			if numSolutions == "all" or len(uniqueHypers) < numSolutions:
				addToSolutions(model, uniqueHypers, feasibility[1:])

		elif feasibility[0] == False and feasibility[1] is not None:
			#If the Krawczyk + Lp loop cannot make a decision about
			#the hyperrectangle, the do the bisect and kill loop - keep
			#bisecting as long atleast one half either contains a unique
			#solution or no solution. Otherwise, add the two halves to
			#the stackList to be processed again.
			hypForBisection = feasibility[1]
			'''hypForBisection = np.array([[0.9, 1.35], 
								[1.35, 1.8], 
								[1.35, 1.8], 
								[0.0, 0.45], 
								[1.35, 1.8], 
								[0.0, 0.9], 
								[0.0, 0.9], 
								[0.0, 0.9]])'''
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
				statVars['totalGSTime'] += end - start
				statVars['numGs'] += 1
				statVars['stringHyperList'].append(("g", intervalUtils.volume(lFeas[1])))
				#print ("rHyp")
				#intervalUtils.printHyper(rHyp)
				start = time.time()
				rFeas = intervalUtils.checkExistenceOfSolution(model, rHyp.transpose(), kAlpha)
				end = time.time()
				#print ("rFeas", rFeas)
				statVars['totalGSTime'] += end - start

				statVars['numGs'] += 1
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
							addToSolutions(model, uniqueHypers, lFeas[1:])
					if rFeas[0]:
						if numSolutions == "all" or len(uniqueHypers) < numSolutions:
							addToSolutions(model, uniqueHypers, rFeas[1:])

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


# Apply Krawczyk and linear programming to refine the hyperrectangle
# @param hyperRectangle 
# @param statVars statVars holds statistical information about the operations performed by the solver.
#	For example, number of Lp's performed
# @param volRedTheshold indicates the stopping criterion for the loop of
#	Krawczyk and LP (implemented by this function). Basically keep repeating the loop
#	as long as the percentage of volume reduction of the hyperrectangle is atleast volRedThreshold. 
# @param model indicates the problem we are trying to solve
# @param kAlpha is the threshold which indicates the stopping criterion for the Krawczyk loop
# @param useLp is a flag to indicate if LP is going to be used
# @return (True, hyper) if hyperRectangle contains a unique
# 	solution and hyper maybe smaller than hyperRectangle containing the solution
# @return (False, None) if hyperRectangle contains no solution
# @return (False, hyper) if hyperRectangle may contain more
# 	than 1 solution and hyper maybe smaller than hyperRectangle containing the solutions
def ifFeasibleHyper(hyperRectangle, statVars, volRedThreshold, model, kAlpha = 1.0, useLp = True):
	lenV = hyperRectangle.shape[0]
	iterNum = 0
	while True:
		newHyperRectangle = np.copy(hyperRectangle)

		# Apply linear programming if the useLp flag is turned on
		if useLp:
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
			for i in range(lenV):
				print (newHyperRectangle[i,0], newHyperRectangle[i,1])'''
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
		statVars['totalGSTime'] += (end - start)
		statVars['numGs'] += 1
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
	


def addToSolutions(model, allHypers, solHyper):
	#print ("allHypers")
	#print (allHypers)
	lenV = len(model.bounds)
	allHypersCopy = [hyper for hyper in allHypers]
	foundOverlap = False
	for hi in range(len(allHypers)):
		oldHyper = allHypers[hi]

		#print ("CHECKING OVERLAP")
		#print ("solHyper")
		#printHyper(solHyper[0])
		#print ("oldHyper")
		#printHyper(oldHyper[0])
		#Check if solHyper overlaps with oldHyper
		if all(interval_intersect(solHyper[0][i], oldHyper[0][i]) is not None for i in range(lenV)):
			#print ("FOUND OVERLAP")
			unionHyper = np.zeros((lenV,2))
			for ui in range(lenV):
				unionHyper[ui,:] = interval_union(solHyper[0][ui], oldHyper[0][ui])

			feasibility = intervalUtils.checkExistenceOfSolution(model, unionHyper.transpose())
			if feasibility[0]:
				#print ("solHyper", solHyper)
				#print ("oldHyper", oldHyper)
				#print ("unionHyper", unionHyper)
				foundOverlap = True
				allHypers[hi][0] = unionHyper
				break

	if not(foundOverlap):
		#print ("adding", solHyper)
		allHypers.append(solHyper)
		return True
	else:
		return False


def schmittTrigger(modelType, inputVoltage, volRedThreshold, statVars, numSolutions = "all", useLp = True):
	statVars.update({'numBisection':0, 'numLp':0, 'numGs':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalGSTime':0, 'totalLPTime':0, 'avgGSTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})
	stringHyperList = []

	if modelType == "mosfet":
		#load the schmitt trigger model
		#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
		modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
		model = SchmittMosfetMark(modelParam = modelParam, inputVoltage = inputVoltage)
	elif modelType == "stMosfet":
		modelParam = [1.0] #Vdd
		model = SchmittStMosfet(modelParam = modelParam, inputVoltage = inputVoltage)

	startExp = time.time()

	allHypers = []

	solverLoop(allHypers, model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectMax, numSolutions=numSolutions, useLp=useLp)
	
	print ("allHypers")
	print (allHypers)
	print ("numSolutions", len(allHypers))


	#Debugging
	#newtonSol = intervalUtils.newton(model,np.array([0.9999998921840166, 0.8, 0.9999998921840166]))
	#print ("newtonSol", newtonSol[1][0], newtonSol[1][1], newtonSol[1][2])
	#print ("fVal at newtonSol", model.f(np.array([1.799993521250602, 1.5192899890822074, 1.799993521250602])))
	
	'''hyper = np.array([[0.999999892182, 0.999999892184],
 					[0.839655709862, 0.839655709864],
					[0.999999892182, 0.999999892184]])

	print ("fVal at hyper")
	print (model.f(hyper))
	feasibility = ifFeasibleHyper(hyper, statVars, volRedThreshold,model, kAlpha = 0.5)
	print ("feas", feasibility[0])
	intervalUtils.printHyper(feasibility[1])'''

	
	# categorize solutions found
	sampleSols, rotatedSols, stableSols, unstableSols = rUtils.categorizeSolutions(allHypers,model)

	for hi in range(len(sampleSols)):
		print ("equivalence class# ", hi)
		print ("main member ", sampleSols[hi])
		print ("check current ", model.f(sampleSols[hi]))
		print ("number of other members ", len(rotatedSols[hi]))
		print ("other member rotationIndices: ")
		for mi in range(len(rotatedSols[hi])):
			print (rotatedSols[hi][mi])
		print ("")

	'''for hi in range(len(sampleSols)):
		if len(rotatedSols[hi]) > lenV - 1 or (len(rotatedSols[hi]) >= 1 and rotatedSols[hi][0] == 0):
			print ("problem equivalence class# ", hi)
			print ("main member ", sampleSols[hi])
			print ("num other Solutions ", len(rotatedSols[hi]))

	print ("")
	print ("numSolutions, ", len(allHypers))
	print ("num stable solutions ", len(stableSols))'''
	'''for si in range(len(stableSols)):
		print stableSols[si]'''
	#print "num unstable solutions ", len(unstableSols)
	'''for si in range(len(unstableSols)):
		print unstableSols[si]'''
	endExp = time.time()
	#print ("TOTAL TIME ", endExp - startExp)
	if statVars['numLp'] != 0:
		statVars['avgLPTime'] = (statVars['totalLPTime']*1.0)/statVars['numLp']
	if statVars['numGs'] != 0:
		statVars['avgGSTime'] = (statVars['totalGSTime']*1.0)/statVars['numGs']

	# test val
	#fVal = model.f(np.array([[-1.07149845e-23,  1.07149845e-23], [-6.42907949e-24,  6.42907949e-24], [-0.1,  4.97086508e-01]]))
	#print ("fVal", fVal)
	print ("numBisection", statVars['numBisection'], "numLp", statVars['numLp'], "numGs", statVars['numGs'],
		"numSingleKill", statVars['numSingleKill'], "numDoubleKill", statVars['numDoubleKill'])
	print ("totalGSTime", statVars['totalGSTime'], "totalLPTime", statVars['totalLPTime'], "avgGSTime", 
		statVars['avgGSTime'], "avgLPTime", statVars['avgLPTime'])
	print ("numLpCalls", statVars['numLpCalls'], "numSuccessLpCalls", statVars['numSuccessLpCalls'], "numUnsuccessLpCalls", statVars['numUnsuccessLpCalls'])
	return allHypers

def inverter(modelType, inputVoltage, volRedThreshold, statVars, numSolutions="all" , useLp=True, kAlpha = 1.0):
	global runOptions
	statVars.update({'numBisection':0, 'numLp':0, 'numGs':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalGSTime':0, 'totalLPTime':0, 'avgGSTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})
	

	if modelType == "mosfet":
		#load the schmitt trigger model
		#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
		modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
		model = InverterMosfet(modelParam = modelParam, inputVoltage = inputVoltage)
	if modelType == "stMosfet":
		modelParam = [1.0] #Vdd
		model = InverterStMosfet(modelParam = modelParam, inputVoltage = inputVoltage)

	startExp = time.time()
	
	allHypers = []

	solverLoop(allHypers, model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectMax, numSolutions=numSolutions, useLp=useLp, kAlpha=kAlpha)
	
	print ("allHypers")
	print (allHypers)
	print ("numSolutions", len(allHypers))
	
	endExp = time.time()
	print ("TOTAL TIME ", endExp - startExp)

	if statVars['numLp'] != 0:
		statVars['avgLPTime'] = (statVars['totalLPTime']*1.0)/statVars['numLp']
	if statVars['numGs'] != 0:
		statVars['avgGSTime'] = (statVars['totalGSTime']*1.0)/statVars['numGs']
	
	print ("numBisection", statVars['numBisection'], "numLp", statVars['numLp'], "numGs", statVars['numGs'],
		"numSingleKill", statVars['numSingleKill'], "numDoubleKill", statVars['numDoubleKill'])
	print ("totalGSTime", statVars['totalGSTime'], "totalLPTime", statVars['totalLPTime'], "avgGSTime", 
		statVars['avgGSTime'], "avgLPTime", statVars['avgLPTime'])
	print ("numLpCalls", statVars['numLpCalls'], "numSuccessLpCalls", statVars['numSuccessLpCalls'], "numUnsuccessLpCalls", statVars['numUnsuccessLpCalls'])
	return allHypers


def rambusOscillator(modelType, numStages, g_cc, volRedThreshold, statVars, numSolutions="all" , useLp=True, kAlpha = 1.0):
	global runOptions
	statVars.update({'numBisection':0, 'numLp':0, 'numGs':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalGSTime':0, 'totalLPTime':0, 'avgGSTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})
	
	if modelType == "tanh":
		a = -5.0
		model = RambusTanh(modelParam = a, g_cc = g_cc, g_fwd = 1.0, numStages=numStages)
	elif modelType == "mosfet":
		#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
		#modelParam = [-0.25, 0.25, 1.0, 1.0, -0.5, 1.0]
		#modelParam = [-0.4, 0.4, 1.8, 1.5, -0.5, 8/3.0]
		modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
		model = RambusMosfetMark(modelParam = modelParam, g_cc = g_cc, g_fwd = 1.0, numStages = numStages)	
	elif modelType == "stMosfet":
		modelParam = [1.0] #Vdd
		model = RambusStMosfet(modelParam = modelParam, g_cc = g_cc, g_fwd = 1.0, numStages = numStages)

	startExp = time.time()
	
	'''hyper = np.array([[0.9, 1.8],
					 [0.0, 0.9],
					 [0.9, 1.8],
					 [0.9, 1.8],
					 [0.9, 1.8],
					 [0.0, 0.9],
					 [0.0, 0.9],
					 [0.9, 1.8],
					 [0.0, 0.9],
					 [0.0, 0.9],
					 [0.0, 0.9],
					 [0.9, 1.8]])
	hyper = np.array([[1.3938656300017775, 1.5111248219290043],
					 [0.0, 0.0008326014376641796],
					 [1.7999999921739855, 1.8],
					 [1.35, 1.59347548330315],
					 [1.5786804198588806, 1.8],
					 [0.0, 0.0019502672081610944],
					 [0.12106371086921032, 0.21635265886498917],
					 [1.799990388012524, 1.8],
					 [0.0, 1.8616507200815385e-09],
					 [0.03991796820270773, 0.2974984015314918],
					 [0.0, 0.32592815141774656],
					 [1.7995591458535327, 1.8]])

	feas = ifFeasibleHyper(hyper, statVars, volRedThreshold,model, useLp=False)
	print ("feas", feas)'''

	'''sol = np.array([0.04369848, 0.56992351, 0.80192295, 0.83295818,
			1.77673741, 1.03976852, 1.74566377, 1.03117171,
			0.91452938, 0.86123781, 0.01632339, 0.55631079])

	print ("model.f", model.f(sol))'''
	
	allHypers = []

	solverLoop(allHypers, model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectMax, numSolutions=numSolutions, useLp=useLp, kAlpha=kAlpha)
	
	print ("allHypers")
	print (allHypers)
	'''print ("numSolutions", len(allHypers))
	
	# categorize solutions found
	sampleSols, rotatedSols, stableSols, unstableSols = rUtils.categorizeSolutions(allHypers,model)

	for hi in range(len(sampleSols)):
		print ("equivalence class# ", hi)
		print ("main member ", sampleSols[hi])
		print ("check current ", model.f(sampleSols[hi]))
		print ("number of other members ", len(rotatedSols[hi]))
		print ("other member rotationIndices: ")
		for mi in range(len(rotatedSols[hi])):
			print (rotatedSols[hi][mi])
		print ("")'''

	'''for hi in range(len(sampleSols)):
		if len(rotatedSols[hi]) > lenV - 1 or (len(rotatedSols[hi]) >= 1 and rotatedSols[hi][0] == 0):
			print ("problem equivalence class# ", hi)
			print ("main member ", sampleSols[hi])
			print ("num other Solutions ", len(rotatedSols[hi]))

	print ("")
	print ("numSolutions, ", len(allHypers))
	print ("num stable solutions ", len(stableSols))'''
	'''for si in range(len(stableSols)):
		print stableSols[si]'''
	#print "num unstable solutions ", len(unstableSols)
	'''for si in range(len(unstableSols)):
		print unstableSols[si]'''
	endExp = time.time()
	#print ("TOTAL TIME ", endExp - startExp)

	'''if statVars['numLp'] != 0:
		statVars['avgLPTime'] = (statVars['totalLPTime']*1.0)/statVars['numLp']
	if statVars['numGs'] != 0:
		statVars['avgGSTime'] = (statVars['totalGSTime']*1.0)/statVars['numGs']
	
	print ("numBisection", statVars['numBisection'], "numLp", statVars['numLp'], "numGs", statVars['numGs'],
		"numSingleKill", statVars['numSingleKill'], "numDoubleKill", statVars['numDoubleKill'])
	print ("totalGSTime", statVars['totalGSTime'], "totalLPTime", statVars['totalLPTime'], "avgGSTime", 
		statVars['avgGSTime'], "avgLPTime", statVars['avgLPTime'])
	print ("numLpCalls", statVars['numLpCalls'], "numSuccessLpCalls", statVars['numSuccessLpCalls'], "numUnsuccessLpCalls", statVars['numUnsuccessLpCalls'])'''
	return allHypers

def validSingleVariableInterval(allHypers, model):
	xBound = [model.bounds[0][0], model.bounds[0][1]]
	sampleSols, rotatedSols, stableSols, unstableSols = rUtils.categorizeSolutions(allHypers,model)

	for hi in range(len(sampleSols)):
		print ("equivalence class# ", hi)
		print ("main member ", sampleSols[hi])
		print ("number of other members ", len(rotatedSols[hi]))
		print ("other member rotationIndices: ")
		for mi in range(len(rotatedSols[hi])):
			print (rotatedSols[hi][mi])
		print ("")


	print ("")
	print ("numSolutions, ", len(allHypers))

	inequalityIntervals = {" > ": [], " < ": []}

	if len(sampleSols) == 0:
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



def singleVariableInequalities(problemType, volRedThreshold, statVars, useLp=True, kAlpha=1.0):
	global runOptions
	statVars.update({'numBisection':0, 'numLp':0, 'numGs':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalGSTime':0, 'totalLPTime':0, 'avgGSTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})

	model, xBound = None, None
	#numSolutions = 1
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


	'''newtonSol = intervalUtils.newton(model,np.array([0.8]))
	print ("newtonSol", newtonSol)
	#[[ -5.31508875e-09   5.31508875e-09]]
	#feasibility = ifFeasibleHyper(np.array([[-0.03333433, 4.1887911 ]]), volRedThreshold,model)
	feasibility = ifFeasibleHyper(np.array([[ xBound[0], xBound[1]]]), volRedThreshold,model)
	print ("feasibility", feasibility)'''

	allHypers = []
	solverLoop(allHypers, model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectMax, numSolutions = "all", useLp=useLp, kAlpha=kAlpha)
	print ("allHypers")
	print (allHypers)
	print ("numSolutions", len(allHypers))
	
	validIntervals = validSingleVariableInterval(allHypers, model)
	print ("validIntervals", validIntervals)
	endExp = time.time()
	#print ("time taken", endExp - startExp)

	if statVars['numLp'] != 0:
		statVars['avgLPTime'] = (statVars['totalLPTime']*1.0)/statVars['numLp']
	if statVars['numGs'] != 0:
		statVars['avgGSTime'] = (statVars['totalGSTime']*1.0)/statVars['numGs']
	return validIntervals



if __name__ == "__main__":
	statVars = {}
	start = time.time()
	#allHypers = rambusOscillator(modelType="mosfet", numStages=2, g_cc=0.5, volRedThreshold=1.0, statVars=statVars, numSolutions="all" , useLp=False)
	#print ("numSolutions", len(allHypers))
	#print ("allHypers", allHypers)
	schmittTrigger(modelType="stMosfet", inputVoltage = 1.0, volRedThreshold = 1.0, statVars = statVars, numSolutions = "all", useLp = False)
	#singleVariableInequalities(problemType="meti25", volRedThreshold=1.0, statVars=statVars, useLp=True)
	#inverter(modelType="mosfet", inputVoltage=1.7, volRedThreshold=1.0, statVars=statVars, numSolutions="all" , useLp=False)
	end = time.time()
	print ("time taken", end - start)
