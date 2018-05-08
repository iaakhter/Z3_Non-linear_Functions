import numpy as np
import time
import copy
import intervalUtils
import tanhModel
import mosfetModel
import example1
import metiProblems
import schmittMosfet
import rambusUtils as rUtils
import resource
import random
import math


def bisectFunOrdering(hyper, orderingArray, orderingIndex):
	bisectIndex = orderingArray[orderingIndex]
	lHyper = np.copy(hyper)
	rHyper = np.copy(hyper)
	midVal = (hyper[bisectIndex][0] + hyper[bisectIndex][1])/2.0
	lHyper[bisectIndex][1] = midVal
	rHyper[bisectIndex][0] = midVal
	return [lHyper, rHyper]

def bisectMax(hyper, orderingArray=None, orderingIndex=None):
	intervalLength = hyper[:,1] - hyper[:,0]
	bisectIndex = np.argmax(intervalLength)
	lHyper = np.copy(hyper)
	rHyper = np.copy(hyper)
	midVal = (hyper[bisectIndex][0] + hyper[bisectIndex][1])/2.0
	lHyper[bisectIndex][1] = midVal
	rHyper[bisectIndex][0] = midVal
	return [lHyper, rHyper]


def volume(hyperRectangle):
	if hyperRectangle is None:
		return None
	vol = 1
	for i in range(hyperRectangle.shape[0]):
		vol *= (hyperRectangle[i,1] - hyperRectangle[i,0])
	return vol

def solverLoop(uniqueHypers, model, volRedThreshold, bisectFun, orderingArray=None, useLp=True):
	global numBisection, numLp, numGs, numSingleKill, numDoubleKill
	global stringHyperList
	lenV = len(model.boundMap)
	hyperRectangle = np.zeros((lenV,2))

	for i in range(lenV):
		hyperRectangle[i,0] = model.boundMap[i][0][0]
		hyperRectangle[i,1] = model.boundMap[i][1][1]

	stringHyperList.append(("i", volume(hyperRectangle)))

	stackList = []
	stackList.append(hyperRectangle)
	while len(stackList) > 0:
		hyperPopped = stackList.pop(-1)
		#print ("hyperPopped", hyperPopped)

		hyperAlreadyConsidered = False
		for hyper in uniqueHypers:
			if np.greater_equal(hyperPopped[:,0], hyper[:,0]).all() and np.less_equal(hyperPopped[:,1], hyper[:,1]).all():
				hyperAlreadyConsidered = True
				break
		if hyperAlreadyConsidered:
			continue

		feasibility = ifFeasibleHyper(hyperPopped, volRedThreshold, model, useLp)
		#print ("feasibility", feasibility)
		if feasibility[0]:
			addToSolutions(model, uniqueHypers, feasibility[1])

		elif feasibility[0] == False and feasibility[1] is not None:
			hypForBisection = feasibility[1]
			orderIndex = 0
			while hypForBisection is not None:
				#print ("hypForBisection", hypForBisection)
				lHyp, rHyp = bisectFun(hypForBisection, orderingArray, orderIndex)
				numBisection += 1
				stringHyperList.append(("b", [volume(lHyp), volume(rHyp)]))
				#print ("lHyp", lHyp)
				orderIndex = (orderIndex + 1)%lenV
				lFeas = intervalUtils.checkExistenceOfSolutionGS(model,lHyp.transpose())
				numGs += 1
				stringHyperList.append(("g", volume(lFeas[1])))
				#print ("lFeas", lFeas)
				#print ("rHyp", rHyp)
				rFeas = intervalUtils.checkExistenceOfSolutionGS(model,rHyp.transpose())
				numGs += 1
				stringHyperList.append(("g", volume(rFeas[1])))
				#print ("rFeas", rFeas)
				if lFeas[0] or rFeas[0] or (lFeas[0] == False and lFeas[1] is None) or (rFeas[0] == False and rFeas[1] is None):
					if lFeas[0] and rFeas[0]:
						numDoubleKill += 1
					elif lFeas[0] == False and lFeas[1] is None and rFeas[0] == False and rFeas[1] is None:
						numDoubleKill += 1
					else:
						numSingleKill += 1
					
					if lFeas[0]:
						#print ("addedHyper", lHyp)
						addToSolutions(model, uniqueHypers, lFeas[1])
					if rFeas[0]:
						#print ("addedHyper", rHyp)
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

	return uniqueHypers




def ifFeasibleHyper(hyperRectangle, volRedThreshold,model,useLp=True):
	global numBisection, numLp, numGs, numSingleKill, numDoubleKill
	global stringHyperList
	lenV = hyperRectangle.shape[0]
	#print ("hyperRectangle")
	#print (hyperRectangle)
	iterNum = 0
	while True:
		#print ("hyperRectangle")
		#print (hyperRectangle)
		kResult = intervalUtils.checkExistenceOfSolutionGS(model,hyperRectangle.transpose())
		numGs += 1
		stringHyperList.append(("g", volume(kResult[1])))
		#print ("kResult")
		#print (kResult)
		if kResult[0]:
			#print ("uniqueHyper", hyperRectangle)
			return (True, kResult[1])

		if kResult[0] == False and kResult[1] is None:
			#print ("K operator not feasible", hyperRectangle)
			return (False, None)
		#print "kResult"
		#print kResult
		#print ("hyperRectangle 2")
		#print (hyperRectangle)
		
		newHyperRectangle = kResult[1]	

		if useLp:
			#print ("startlp")
			feasible, newHyperRectangle = model.linearConstraints(newHyperRectangle)
			#print ("endlp")
			numLp += 1
			if feasible:
				vol = volume(newHyperRectangle)
			else:
				vol = None
			stringHyperList.append(("l", vol))
			#print ("newHyperRectangle ", newHyperRectangle)
			if feasible == False:
				#print ("LP not feasible", hyperRectangle)
				return (False, None)

			for i in range(lenV):
				if newHyperRectangle[i,0] < hyperRectangle[i,0]:
					newHyperRectangle[i,0] = hyperRectangle[i,0]
				if newHyperRectangle[i,1] > hyperRectangle[i,1]:
					newHyperRectangle[i,1] = hyperRectangle[i,1]
		 

		hyperVol = 1.0
		for i in range(lenV):
			hyperVol *= (hyperRectangle[i,1] - hyperRectangle[i,0])
		#hyperVol = hyperVol**(1.0/lenV)

		newHyperVol = 1.0
		for i in range(lenV):
			newHyperVol *= (newHyperRectangle[i,1] - newHyperRectangle[i,0])
		#newHyperVol = newHyperVol**(1.0/lenV)

		propReduc = (hyperVol - newHyperVol)/hyperVol
		#print ("propReduc", propReduc)
		
		#if np.less_equal(newHyperRectangle[:,1] - newHyperRectangle[:,0],hyperBound*np.ones((lenV))).all() or np.less_equal(np.absolute(newHyperRectangle - hyperRectangle),1e-4*np.ones((lenV,2))).all():
		if math.isnan(propReduc) or propReduc < volRedThreshold:
			if kResult[0] == False and kResult[1] is not None:
				#return (False, kResult[1])
				return (False, newHyperRectangle)
		hyperRectangle = newHyperRectangle
		iterNum+=1
		#print ("here?")
		#return
	

# check if a solution already exists in 
# the list of solutions and if not, add it
# to the list of solutions
def addToSolutions(model, allHypers, solHyper):
	exampleSoln = (solHyper[:,0] + solHyper[:,1])/2.0
	finalSoln = intervalUtils.newton(model,exampleSoln)
	if finalSoln[0]:
		solExists = False
		for existingHypers in allHypers:
			exampleSolnInHyper = (existingHypers[:,0] + existingHypers[:,1])/2.0
			finalSolnInHyper = intervalUtils.newton(model, exampleSolnInHyper)
			if finalSolnInHyper[0]:
				if np.less_equal(np.absolute(finalSolnInHyper[1] - finalSoln[1]), np.ones(finalSoln[1].shape)*1e-14).all():
					# solution already exists
					solExists = True
					break
		if not(solExists):
			allHypers.append(solHyper)





def findAndIgnoreNewtonSoln(model, minVal, maxVal, numSolutions, numTrials = 10):
	lenV = len(model.boundMap)
	allHypers = []
	solutionsFoundSoFar = []
	boundMap = model.boundMap
	overallHyper = np.zeros((lenV,2))
	for i in range(lenV):
		overallHyper[i,0] = boundMap[i][0][0]
		overallHyper[i,1] = boundMap[i][1][1]
	#print ("numTrials in newtons preprocessing", numTrials)
	start = time.time()
	numFailures = 0
	for n in range(numTrials):
		numFailures += 1
		if len(allHypers) == numSolutions:
			break
		trialSoln = np.random.uniform(minVal, maxVal, (lenV))
		finalSoln = intervalUtils.newton(model,trialSoln)
		if finalSoln[0] and np.greater_equal(finalSoln[1], overallHyper[:,0]).all() and np.less_equal(finalSoln[1], overallHyper[:,1]).all():
			alreadyFound = False
			for exSol in solutionsFoundSoFar:
				diff = np.absolute(finalSoln[1] - exSol)
				if np.less_equal(diff, np.ones(diff.shape)*1e-14).all():
					alreadyFound = True
					break
			if not(alreadyFound):
				solutionsFoundSoFar.append(finalSoln[1])
				hyperWithUniqueSoln = np.zeros((lenV,2))
				diff = np.ones((lenV))*0.4
				startingIndex = 0
				while True:
					#print ("diff", diff)
					hyperWithUniqueSoln[:,0] = finalSoln[1] - diff
					hyperWithUniqueSoln[:,1] = finalSoln[1] + diff
					kResult = intervalUtils.checkExistenceOfSolutionGS(model,hyperWithUniqueSoln.transpose())
					if kResult[0] == False and kResult[1] is not None:
						diff[startingIndex] = diff[startingIndex]/2.0
						startingIndex = (startingIndex + 1)%lenV
					else:
						end = time.time()
						#print ("Preprocess: found unique hyper ", hyperWithUniqueSoln, "in", end - start, "after", numFailures, "failures")
						start = time.time()
						numFailures = 0
						allHypers.append(hyperWithUniqueSoln)
						if model.solver is not None:
							model.ignoreHyperInZ3(hyperWithUniqueSoln)
						#model.linearConstraints(hyperWithUniqueSoln)
						break
	#print ("total num hypers found in ", numTrials, " trials is ", len(allHypers))
	return (allHypers, solutionsFoundSoFar)

def schmittTrigger(inputVoltage, lpThreshold, numSolutions = "all", newtonHypers = True, useLp = True):
	#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
	modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
	model = schmittMosfet.SchmittMosfet(modelParam = modelParam, inputVoltage = inputVoltage)

	startExp = time.time()
	lenV = 3
	indexChoiceArray = []
	for i in range(lenV):
		indexChoiceArray.append(i)
	print ("indexChoiceArray", indexChoiceArray)
	boundMap = model.boundMap
	print ("boundMap ", boundMap)

	volRedThreshold = 0.3
	numStages = 1

	allHypers = []
	if newtonHypers is not None:
		allHypers, solutionsFoundSoFar = findAndIgnoreNewtonSoln(model, boundMap[0][0][0], boundMap[0][1][1], numSolutions=numSolutions, numTrials = numStages*100)
		newtonHypers = np.copy(allHypers)

	solverLoop(allHypers, model, lpThreshold, bisectFun=bisectMax, orderingArray=indexChoiceArray, useLp=useLp)
	print ("allHypers")
	print (allHypers)
	print ("numSolutions", len(allHypers))
	
	# categorize solutions found
	'''sampleSols, rotatedSols, stableSols, unstableSols = rUtils.categorizeSolutions(allHypers,model)

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
	print ("num stable solutions ", len(stableSols))'''
	'''for si in range(len(stableSols)):
		print stableSols[si]'''
	#print "num unstable solutions ", len(unstableSols)
	'''for si in range(len(unstableSols)):
		print unstableSols[si]'''
	endExp = time.time()
	#print ("TOTAL TIME ", endExp - startExp)
	return allHypers

def rambusOscillator(modelType, numStages, g_cc, lpThreshold, numSolutions="all" , newtonHypers=True, useLp=True):
	global numBisection, numLp, numGs, numSingleKill, numDoubleKill
	global stringHyperList
	numBisection, numLp, numGs, numSingleKill, numDoubleKill = 0, 0, 0, 0, 0
	stringHyperList = []
	a = -5.0
	model = tanhModel.TanhModel(modelParam = a, g_cc = g_cc, g_fwd = 1.0, numStages=numStages)
	
	if modelType == "mosfet":
		#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
		#modelParam = [-0.25, 0.25, 1.0, 1.0, -0.5, 1.0]
		#modelParam = [-0.4, 0.4, 1.8, 1.5, -0.5, 8/3.0]
		modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
		model = mosfetModel.MosfetModel(modelParam = modelParam, g_cc = g_cc, g_fwd = 1.0, numStages = numStages)
	
	startExp = time.time()
	lenV = numStages*2
	indexChoiceArray = []
	firstIndex = numStages - 1
	secondIndex = numStages*2 - 1
	for i in range(lenV):
		#indexChoiceArray.append(i)
		if i%2 == 0:
			indexChoiceArray.append(firstIndex)
			firstIndex -= 1
		else:
			indexChoiceArray.append(secondIndex)
			secondIndex -= 1
	#print ("indexChoiceArray", indexChoiceArray)
	boundMap = model.boundMap
	#print ("boundMap ", boundMap)
	

	allHypers = []
	if newtonHypers is not None:
		allHypers, solutionsFoundSoFar = findAndIgnoreNewtonSoln(model, boundMap[0][0][0], boundMap[0][1][1], numSolutions=numSolutions, numTrials = numStages*100)
		newtonHypers = np.copy(allHypers)

	solverLoop(allHypers, model, lpThreshold, bisectFun=bisectMax, orderingArray=indexChoiceArray, useLp=useLp)
	#print ("allHypers")
	#print (allHypers)
	print ("numSolutions", len(allHypers))
	
	# categorize solutions found
	'''sampleSols, rotatedSols, stableSols, unstableSols = rUtils.categorizeSolutions(allHypers,model)

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
	print ("num stable solutions ", len(stableSols))'''
	'''for si in range(len(stableSols)):
		print stableSols[si]'''
	#print "num unstable solutions ", len(unstableSols)
	'''for si in range(len(unstableSols)):
		print unstableSols[si]'''
	endExp = time.time()
	#print ("TOTAL TIME ", endExp - startExp)
	#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
	globalVars = [numBisection, numLp, numGs, numSingleKill, numDoubleKill, stringHyperList]
	return allHypers, globalVars

def validSingleVariableInterval(allHypers, model):
	xBound = [model.boundMap[0][0][0], model.boundMap[0][1][1]]
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
		xRandVal = model.oscNum(xRand)[2]
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
			xRandVal = model.oscNum(xRand)[2]
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



def singleVariableInequalities(problemType, volRedThreshold, numSolutions="all", newtonHypers=True, useLp=True):
	model, xBound = None, None
	#numSolutions = 1
	startExp = time.time()
	
	if problemType == "dRealExample":
		#xBound = [-100.0, -0.001]
		xBound = [3.0, 64.0]
		model = example1.Example1(xBound[0], xBound[1], " > ")

	if problemType == "meti25":
		xBound = [math.pi/3.0, (2*math.pi/3.0)]
		model = metiProblems.Meti25(xBound[0], xBound[1], " > ")

	if problemType == "meti18":
		xBound = [0.0, 100.0/201]
		model = metiProblems.Meti18(xBound[0], xBound[1], " > ")

	if problemType == "meti10":
		xBound = [0.5, 1.06155141]
		model = metiProblems.Meti10(xBound[0], xBound[1], " < ")

	lenV = 1
	indexChoiceArray = [0]

	'''newtonSol = intervalUtils.newton(model,np.array([0.8]))
	print ("newtonSol", newtonSol)
	#[[ -5.31508875e-09   5.31508875e-09]]
	#feasibility = ifFeasibleHyper(np.array([[-0.03333433, 4.1887911 ]]), volRedThreshold,model)
	feasibility = ifFeasibleHyper(np.array([[ xBound[0], xBound[1]]]), volRedThreshold,model)
	print ("feasibility", feasibility)'''

	allHypers = []
	if newtonHypers is not None:
		allHypers, solutionsFoundSoFar = findAndIgnoreNewtonSoln(model, xBound[0], xBound[1], numSolutions=numSolutions, numTrials = 100)
		newtonHypers = np.copy(allHypers)

	solverLoop(allHypers, model, volRedThreshold, bisectFun=bisectMax, orderingArray=indexChoiceArray,useLp=useLp)
	print ("allHypers")
	print (allHypers)
	print ("numSolutions", len(allHypers))
	
	validIntervals = validSingleVariableInterval(allHypers, model)
	print ("validIntervals", validIntervals)
	endExp = time.time()
	#print ("time taken", endExp - startExp)


if __name__ == "__main__":
	allHypers, globalVars = rambusOscillator(modelType="tanh", numStages=4, g_cc=4.0, lpThreshold=0.3, numSolutions="all" , newtonHypers=True, useLp=True)
	print ("allHypers", allHypers)
	print ("numSolutions", len(allHypers))
	#schmittTrigger(inputVoltage = 1.0, numSolutions = "all", newtonHypers = None)
	#singleVariableInequalities(problemType="meti10", volRedThreshold=0.3, numSolutions="all", newtonHypers=True, useLp=True)
	#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
