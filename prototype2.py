import numpy as np
import time
import intervalUtils
from intervalBasics import *
import rambusTanh
import rambusMosfet
import rambusStMosfet
import flyspeckProblems
import metiProblems
import schmittMosfet
import rambusUtils as rUtils
import random
import math
import circuit

def printHyper(hyper):
	for i in range(hyper.shape[0]):
		print (hyper[i,0], hyper[i,1])


def bisectFunOrdering(hyper):
	orderingArray = options[0]
	orderingIndex = options[1]
	bisectIndex = orderingArray[orderingIndex]
	lHyper = np.copy(hyper)
	rHyper = np.copy(hyper)
	midVal = (hyper[bisectIndex][0] + hyper[bisectIndex][1])/2.0
	lHyper[bisectIndex][1] = midVal
	rHyper[bisectIndex][0] = midVal
	return [lHyper, rHyper]

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


#solver's main loop
#hyperrectangles containing unique solutions are added in uniqueHypers. 
#	If Newton's preprocessing step was used then uniqueHypers also containing
#	maximal hyperrectangles containig the solution found by Newton's method
#solutionsFoundSoFar lists the solutions in uniqueHypers.
#model indicates the problem we are trying to solve - rambus/schmitt/metitarski
#volRedThreshold is the threshold which indicates how many times the loop of
#	Gauss-Seidel and LP is applied
#bisectFun is the bisection mechanism by which hyperrectangles are bisected
#orderingArray is the order of indices considered during bisection. This is not
#	None if bisectFun expects an ordering
#useLp indicates if we need to use LP
def solverLoop(uniqueHypers, model, statVars, volRedThreshold, bisectFun, numSolutions = "all", useLp=True, kAlpha = 1.0):
	lenV = len(model.bounds)
	hyperRectangle = np.zeros((lenV,2))

	for i in range(lenV):
		hyperRectangle[i,0] = model.bounds[i][0]
		hyperRectangle[i,1] = model.bounds[i][1]

	statVars['stringHyperList'].append(("i", intervalUtils.volume(hyperRectangle)))
	
	start = time.time()
	feas = intervalUtils.checkExistenceOfSolution(model,hyperRectangle.transpose(),kAlpha)
	end = time.time()
	statVars['totalGSTime'] += end - start
	statVars['numGs'] += 1
	
	statVars['stringHyperList'].append(("g", intervalUtils.volume(feas[1])))


	#stack containing hyperrectangles about which any decision
	#has not been made - about whether they contain unique solution
	#or no solution
	stackList = [feas[1]]

	while len(stackList) > 0:
		#pop the hyperrectangle
		hyperPopped = stackList.pop(-1)
		print ("hyperPopped")
		printHyper(hyperPopped)
		
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

		#Apply the Gauss-Seidel + Lp loop
		feasibility = ifFeasibleHyper(hyperPopped, statVars, volRedThreshold, model, kAlpha, useLp)
		
		print ("feasibility", feasibility)
		if feasibility[0]:
			#If the Gauss_Seidel + Lp loop indicate uniqueness, then add the hyperrectangle
			#to our list
			if numSolutions == "all" or len(uniqueHypers) < numSolutions:
				addToSolutions(model, uniqueHypers, feasibility[1:])

		elif feasibility[0] == False and feasibility[1] is not None:
			#If the Gauss-Seidel + Lp loop cannot make a decision about
			#the hyperrectangle, the do the bisect and kill loop - keep
			#bisecting as long atleast one half either contains a unique
			#solution or no solution. Otherwise, add the two halves to
			#the stackList to be processed again.
			hypForBisection = feasibility[1]
			while hypForBisection is not None:
				print ("hypForBisection")
				printHyper(hypForBisection)
				lHyp, rHyp = bisectFun(hypForBisection)
				statVars['numBisection'] += 1
				statVars['stringHyperList'].append(("b", [intervalUtils.volume(lHyp), intervalUtils.volume(rHyp)]))
				start = time.time()
				lFeas = intervalUtils.checkExistenceOfSolution(model, lHyp.transpose(), kAlpha)
				end = time.time()
				statVars['totalGSTime'] += end - start
				statVars['numGs'] += 1
				statVars['stringHyperList'].append(("g", intervalUtils.volume(lFeas[1])))
				start = time.time()
				rFeas = intervalUtils.checkExistenceOfSolution(model, rHyp.transpose(), kAlpha)
				end = time.time()
				statVars['totalGSTime'] += end - start

				statVars['numGs'] += 1
				statVars['stringHyperList'].append(("g", intervalUtils.volume(rFeas[1])))
				print ("lHyp")
				printHyper(lHyp)
				print ("lFeas", lFeas)
				print ("rHyp")
				printHyper(rHyp)
				print ("rFeas", rFeas)
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


# Apply gauss seidel and linear programming to refine the hyperrectangle
# Keep repeating as long as the percentage of volume reduction of the
# hyperrectangle is atleast volRedThreshold. model indicates the problem
# we are trying to solve
# return (True, hyper) if hyperRectangle contains a unique
# solution and hyper maybe smaller than hyperRectangle containing the solution
# return (False, None) if hyperRectangle contains no solution
# return (False, hyper) if hyperRectangle may contain more
# than 1 solution and hyper maybe smaller than hyperRectangle containing the solutions
def ifFeasibleHyper(hyperRectangle, statVars, volRedThreshold, model, kAlpha = 1.0, useLp = True):
	lenV = hyperRectangle.shape[0]
	iterNum = 0
	while True:
		newHyperRectangle = np.copy(hyperRectangle)

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
			print ("newHyperRectangle", newHyperRectangle)
			for i in range(lenV):
				print (newHyperRectangle[i,0], newHyperRectangle[i,1])
			if feasible == False:
				return (False, None)

			for i in range(lenV):
				if newHyperRectangle[i,0] < hyperRectangle[i,0]:
					newHyperRectangle[i,0] = hyperRectangle[i,0]
				if newHyperRectangle[i,1] > hyperRectangle[i,1]:
					newHyperRectangle[i,1] = hyperRectangle[i,1]


		start = time.time()
		#First apply Krawczyk
		kResult = intervalUtils.checkExistenceOfSolution(model, newHyperRectangle.transpose(), kAlpha)
		end = time.time()
		statVars['totalGSTime'] += (end - start)
		statVars['numGs'] += 1
		statVars['stringHyperList'].append(("g", intervalUtils.volume(kResult[1])))
		#If gauss-seidel interval says there is unique solution
		#in hyperrectangle or no solution, then return
		
		#Unique solution or no solution
		if kResult[0] or kResult[1] is None:
			#print ("uniqueHyper", hyperRectangle)
			return kResult

		#If Gauss-Seidel cannot make a decision apply linear programming
		newHyperRectangle = kResult[1]			 

		hyperVol = intervalUtils.volume(hyperRectangle)

		newHyperVol = intervalUtils.volume(newHyperRectangle)

		propReduc = (hyperVol - newHyperVol)/hyperVol
		
		# If the proportion of volume reduction is not atleast
		# volRedThreshold then return
		if math.isnan(propReduc) or propReduc < volRedThreshold:
			#return (False, kResult[1])
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

		#Check if solHyper overlaps with oldHyper
		if all(interval_intersect(solHyper[0][i], oldHyper[0][i]) is not None for i in range(lenV)):
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


def schmittTrigger(inputVoltage, volRedThreshold, statVars, numSolutions = "all", useLp = True):
	statVars.update({'numBisection':0, 'numLp':0, 'numGs':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalGSTime':0, 'totalLPTime':0, 'avgGSTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})
	stringHyperList = []

	#load the schmitt trigger model
	#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
	modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
	model = schmittMosfet.SchmittMosfetMark(modelParam = modelParam, inputVoltage = inputVoltage)

	startExp = time.time()

	allHypers = []

	solverLoop(allHypers, model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectMax, numSolutions=numSolutions, useLp=useLp)
	
	print ("allHypers")
	print (allHypers)
	print ("numSolutions", len(allHypers))


	#Debugging
	'''newtonSol = intervalUtils.newton(model,np.array([1.8, 1.4, 1.8]))
	print ("newtonSol", newtonSol)'''

	'''hyper = np.array([[0.0, 0.19752105925203098],
 					[0.0, 0.1202708866316566],
					[0.0, 0.225]])

	feasibility = ifFeasibleHyper(hyper, statVars, volRedThreshold,model)
	print ("feas", feasibility)'''

	
	# categorize solutions found
	'''sampleSols, rotatedSols, stableSols, unstableSols = rUtils.categorizeSolutions(allHypers,model)'''

	'''for hi in range(len(sampleSols)):
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

def rambusOscillator(modelType, numStages, g_cc, volRedThreshold, statVars, numSolutions="all" , useLp=True, kAlpha = 1.0):
	global runOptions
	statVars.update({'numBisection':0, 'numLp':0, 'numGs':0, 'numSingleKill':0, 'numDoubleKill':0,
					'totalGSTime':0, 'totalLPTime':0, 'avgGSTime':0, 'avgLPTime':0, 'stringHyperList':[],
					'numLpCalls':0, 'numSuccessLpCalls':0, 'numUnsuccessLpCalls':0})
	
	if modelType == "tanh":
		a = -5.0
		model = rambusTanh.RambusTanh(modelParam = a, g_cc = g_cc, g_fwd = 1.0, numStages=numStages)
	elif modelType == "mosfet":
		#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
		#modelParam = [-0.25, 0.25, 1.0, 1.0, -0.5, 1.0]
		#modelParam = [-0.4, 0.4, 1.8, 1.5, -0.5, 8/3.0]
		modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
		model = rambusMosfet.RambusMosfetMark(modelParam = modelParam, g_cc = g_cc, g_fwd = 1.0, numStages = numStages)	
	elif modelType == "stMosfet":
		modelParam = [1.8] #Vdd
		model = rambusStMosfet.RambusStMosfet(modelParam = modelParam, g_cc = g_cc, g_fwd = 1.0, numStages = numStages)

	startExp = time.time()
	
	'''hyper = np.array([[-0.37800106, -0.02183056],
					 [-0.78603865, -0.57121116],
					 [ 0.99546833,  0.99977379],
					 [ 0.07034466,  0.07034743],
					 [ 0.02168568,  0.73358026],
					 [ 0.39912437,  0.78861493],
					 [-0.99977743, -0.99253623],
					 [-0.07034689, -0.07034519]])

	feas = ifFeasibleHyper(hyper, statVars, volRedThreshold,model)
	print ("feas", feas)'''

	'''sol = np.array([0.04369848, 0.56992351, 0.80192295, 0.83295818,
			1.77673741, 1.03976852, 1.74566377, 1.03117171,
			0.91452938, 0.86123781, 0.01632339, 0.55631079])

	print ("model.f", model.f(sol))'''
	
	allHypers = []

	solverLoop(allHypers, model, statVars=statVars, volRedThreshold=volRedThreshold, bisectFun=bisectMax, numSolutions=numSolutions, useLp=useLp, kAlpha=kAlpha)
	
	#print ("allHypers")
	#print (allHypers)
	print ("numSolutions", len(allHypers))
	
	# categorize solutions found
	sampleSols, rotatedSols, stableSols, unstableSols = rUtils.categorizeSolutions(allHypers,model)

	for hi in range(len(sampleSols)):
		print ("equivalence class# ", hi)
		print ("main member ", sampleSols[hi])
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
	
	print ("numBisection", statVars['numBisection'], "numLp", statVars['numLp'], "numGs", statVars['numGs'],
		"numSingleKill", statVars['numSingleKill'], "numDoubleKill", statVars['numDoubleKill'])
	print ("totalGSTime", statVars['totalGSTime'], "totalLPTime", statVars['totalLPTime'], "avgGSTime", 
		statVars['avgGSTime'], "avgLPTime", statVars['avgLPTime'])
	print ("numLpCalls", statVars['numLpCalls'], "numSuccessLpCalls", statVars['numSuccessLpCalls'], "numUnsuccessLpCalls", statVars['numUnsuccessLpCalls'])
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
	#logging.basicConfig(level=logging.DEBUG)
	start = time.time()
	allHypers = rambusOscillator(modelType="mosfet", numStages=2, g_cc=4.0, volRedThreshold=1.0, statVars=statVars, numSolutions="all" , useLp=False)
	#print ("allHypers", allHypers)
	#schmittTrigger(inputVoltage = 1.0, volRedThreshold = 1.0, statVars = statVars, numSolutions = "all", useLp = True)
	#singleVariableInequalities(problemType="flyspeck172", volRedThreshold=1.0, statVars=statVars, useLp=True)
	#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
	#print ("numSolutions", len(allHypers))
	#separateHyperSpaceTest()
	end = time.time()
	print ("time taken", end - start)
