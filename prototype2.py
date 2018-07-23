import numpy as np
import time
import copy
import intervalUtils
from intervalBasics import *
import rambusTanh
import rambusMosfet
import flyspeckProblems
import metiProblems
import schmittMosfet
import rambusUtils as rUtils
import random
import math
import circuit
import pickle
from solverAnalysis import *


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
	smallerHyperList = [smallHyper]
	for i in range(lenV):
		indexOfSeparation = i
		hyper1 = np.copy(startHyper)
		hyper1[indexOfSeparation,1] = smallHyper[indexOfSeparation, 0]
		hyper2 = np.copy(startHyper)
		hyper2[indexOfSeparation,0] = smallHyper[indexOfSeparation, 1]
		smallerHyperList.append(hyper1)
		smallerHyperList.append(hyper2)
		startHyper[indexOfSeparation,:] = smallHyper[indexOfSeparation,:]

	return smallerHyperList

# Try to find one solution in hyper using Newton's method
def findNewtonSol(model, hyper, numTrials = 100):
	lenV = len(model.bounds)
	allHypers = []
	solutionsFoundSoFar = []
	hyperRange = hyper[:,1] - hyper[:,0]
	
	#print ("numTrials in newtons preprocessing", numTrials)
	start = time.time()
	numFailures = 0
	for n in range(numTrials):
		if len(allHypers) == 1:
			break
		numFailures += 1
		trialSoln = np.multiply(np.random.rand((lenV)), hyperRange) + hyper[:,0]
		finalSoln = intervalUtils.newton(model,trialSoln)
		if finalSoln[0] and np.greater_equal(finalSoln[1], hyper[:,0]).all() and np.less_equal(finalSoln[1], hyper[:,1]).all():
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
					kResult = intervalUtils.checkExistenceOfSolution(model,hyperWithUniqueSoln.transpose())
					if kResult[0] == False and kResult[1] is not None:
						diff[startingIndex] = diff[startingIndex]/2.0
						startingIndex = (startingIndex + 1)%lenV
					else:
						end = time.time()
						#print ("Preprocess: found unique hyper ", hyperWithUniqueSoln, "in", end - start, "after", numFailures, "failures")
						start = time.time()
						numFailures = 0
						allHypers.append(hyperWithUniqueSoln)
						break
	#print ("total num hypers found in ", numTrials, " trials is ", len(allHypers))
	return (allHypers, solutionsFoundSoFar)


# Try another algorithm for the solver where we use Newton's method
def solverLoop2(uniqueHypers, model, volRedThreshold, bisectFun, numSolutions = "all", numTrials = 100, orderingArray=None, useLp = True):
	global numBisection, numLp, numGs, numSingleKill, numDoubleKill
	global stringHyperList
	global totalLPTime, totalGSTime
	lenV = len(model.bounds)
	hyperRectangle = np.zeros((lenV,2))

	for i in range(lenV):
		hyperRectangle[i,0] = model.bounds[i][0]
		hyperRectangle[i,1] = model.bounds[i][1]

	stringHyperList.append(("i", intervalUtils.volume(hyperRectangle)))

	start = time.time()
	feas = intervalUtils.checkExistenceOfSolution(model,hyperRectangle.transpose())
	end = time.time()
	totalGSTime += end - start
	numGs += 1
	if feas[0] or feas[1] is None:
		return uniqueHypers
	stringHyperList.append(("g", intervalUtils.volume(feas[1])))

	#stack containing hyperrectangles about which any decision
	#has not been made - about whether they contain unique solution
	#or no solution
	stackList = []
	stackList.append(feas[1])
	numNewtonUsed = 0
	while len(stackList) > 0:
		print ("len stack", len(stackList))

		#pop the hyperrectangle
		hyperPopped = stackList.pop(-1)
		print ("hyperPopped", hyperPopped)

		#Apply the Gauss-Seidel + Lp loop
		feasibility = ifFeasibleHyper(hyperPopped, volRedThreshold, model, useLp)
		#print ("feasibility", feasibility)
		
		if feasibility[0]:
			#If the Gauss_Seidel + Lp loop indicate uniqueness, then add the hyperrectangle
			#to our list
			if numSolutions == "all" or len(uniqueHypers) < numSolutions:
				addToSolutions(model, uniqueHypers, feasibility[1])
			else:
				return uniqueHypers

		elif feasibility[0] == False and feasibility[1] is not None:
			# For now let findNewtonSols return at max 1 solution
			newtonHypers, newtonSols = findNewtonSol(model, feasibility[1], numTrials)
			for hyper in newtonHypers:
				if numSolutions == "all" or len(uniqueHypers) < numSolutions:
					addToSolutions(model, uniqueHypers, hyper)
				else:
					return uniqueHypers

				hypersToCheck = separateHyperSpace(feasibility[1], hyper)[1:]

				stackList += hypersToCheck

			if len(newtonHypers) > 0:
				numNewtonUsed += 1

			if len(newtonHypers) == 0:
				#If the Gauss-Seidel + Lp loop cannot make a decision about
				#the hyperrectangle, the do the bisect and kill loop - keep
				#bisecting as long atleast one half either contains a unique
				#solution or no solution. Otherwise, add the two halves to
				#the stackList to be processed again.
				hypForBisection = feasibility[1]
				#print (hypForBisection[:,1] - hypForBisection[:,0])
				orderIndex = 0
				while hypForBisection is not None:
					#print ("hypForBisection", hypForBisection)

					lHyp, rHyp = bisectFun(hypForBisection, orderingArray, orderIndex)
					numBisection += 1
					stringHyperList.append(("b", [intervalUtils.volume(lHyp), intervalUtils.volume(rHyp)]))
					#print ("lHyp", lHyp)
					orderIndex = (orderIndex + 1)%lenV
					start = time.time()
					lFeas = intervalUtils.checkExistenceOfSolution(model,lHyp.transpose())
					end = time.time()
					totalGSTime += end - start
					numGs += 1
					stringHyperList.append(("g", intervalUtils.volume(lFeas[1])))
					#print ("lFeas", lFeas)
					#print ("rHyp", rHyp)
					start = time.time()
					rFeas = intervalUtils.checkExistenceOfSolution(model,rHyp.transpose())
					end = time.time()
					totalGSTime += end - start

					numGs += 1
					stringHyperList.append(("g", intervalUtils.volume(rFeas[1])))
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
							if numSolutions == "all" or len(uniqueHypers) < numSolutions:
								addToSolutions(model, uniqueHypers, lFeas[1])
							else:
								return uniqueHypers
						if rFeas[0]:
							#print ("addedHyper", rHyp)
							if numSolutions == "all" or len(uniqueHypers) < numSolutions:
								addToSolutions(model, uniqueHypers, rFeas[1])
							else:
								return uniqueHypers

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

	print ("numNewtonUsed", numNewtonUsed)
	return uniqueHypers



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
def solverLoop(uniqueHypers, model, volRedThreshold, bisectFun, numSolutions = "all", orderingArray=None, useLp=True, kAlpha = 1.0):
	global numBisection, numLp, numGs, numSingleKill, numDoubleKill
	global stringHyperList
	global totalLPTime, totalGSTime
	lenV = len(model.bounds)
	hyperRectangle = np.zeros((lenV,2))

	for i in range(lenV):
		hyperRectangle[i,0] = model.bounds[i][0]
		hyperRectangle[i,1] = model.bounds[i][1]

	stringHyperList.append(("i", intervalUtils.volume(hyperRectangle)))
	rootHyper = HyperInfo(np.copy(hyperRectangle))
	
	start = time.time()
	feas = intervalUtils.checkExistenceOfSolution(model,hyperRectangle.transpose(),kAlpha)
	end = time.time()
	totalGSTime += end - start
	numGs += 1
	rootHyper.operation = "k"
	if feas[1] is not None:
		rootHyper.nextChild.append(HyperInfo(np.copy(feas[1])))
	else:
		rootHyper.nextChild.append(HyperInfo(None))

	if feas[0] or feas[1] is None:
		return rootHyper
	stringHyperList.append(("g", intervalUtils.volume(feas[1])))


	#stack containing hyperrectangles about which any decision
	#has not been made - about whether they contain unique solution
	#or no solution
	stackList = [feas[1]]
	allHyperInfos = [rootHyper.nextChild[0]]
	indicesPopped = [0]

	while len(stackList) > 0:
		#print ("len stack", len(stackList))

		#pop the hyperrectangle
		hyperPopped = stackList.pop(-1)
		#print ("hyperPopped", hyperPopped)
		print ("hyperPopped")
		for i in range(lenV):
			print (hyperPopped[i][0], hyperPopped[i][1])

		indexPopped = indicesPopped.pop(-1)
		#print ("indexPopped", indexPopped)
		#print ("allHyperInfos[indexPopped]", allHyperInfos[indexPopped])
		
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
		feasibility = ifFeasibleHyper(hyperPopped, volRedThreshold, model, kAlpha, useLp, allHyperInfos, indexPopped)
		allHyperInfos[indexPopped].operation = "k"
		nextChild = None
		
		if feasibility[0]:
			#If the Gauss_Seidel + Lp loop indicate uniqueness, then add the hyperrectangle
			#to our list
			allHyperInfos[indexPopped].operation += "unique"
			nextChild = HyperInfo(np.copy(feasibility[1]))
			allHyperInfos[indexPopped].nextChild.append(nextChild)
			if numSolutions == "all" or len(uniqueHypers) < numSolutions:
				#print ("feasibility[1:]", feasibility[1:])
				addToSolutions(model, uniqueHypers, feasibility[1:])
			else:
				return rootHyper

		elif feasibility[0] == False and feasibility[1] is not None:
			#If the Gauss-Seidel + Lp loop cannot make a decision about
			#the hyperrectangle, the do the bisect and kill loop - keep
			#bisecting as long atleast one half either contains a unique
			#solution or no solution. Otherwise, add the two halves to
			#the stackList to be processed again.
			child = HyperInfo(np.copy(feasibility[1]))
			#print ("nextChild numChildren", len(child.nextChild))
			allHyperInfos[indexPopped].nextChild.append(child)
			#print ("allHyperInfos[indexPopped]", allHyperInfos[indexPopped])
			hypForBisection = feasibility[1]
			#print (hypForBisection[:,1] - hypForBisection[:,0])
			orderIndex = 0
			curChild = child
			while hypForBisection is not None:
				#print ("hypForBisection", hypForBisection)
				curChild.operation = "b"
				lHyp, rHyp = bisectFun(hypForBisection, orderingArray, orderIndex)
				curChild.nextChild.append(HyperInfo(np.copy(lHyp)))
				curChild.nextChild.append(HyperInfo(np.copy(rHyp)))
				#print ("hypForBisection", hypForBisection)
				#print ("lHyp", lHyp)
				#print ("rHyp", rHyp)
				#print ("curChild", curChild)
				#print ("curChild.nextChild[0]", curChild.nextChild[0])
				#print ("curChild.nextChild[1]", curChild.nextChild[1])
				#print ("allHyperInfos[indexPopped].nextChild[0].nextChild[0]", allHyperInfos[indexPopped].nextChild[0].nextChild[0])
				#print ("allHyperInfos[indexPopped].nextChild[0].nextChild[1]", allHyperInfos[indexPopped].nextChild[0].nextChild[1])
				#print ("allHyperInfos[indexPopped].nextChild[0]", allHyperInfos[indexPopped].nextChild[0])
				numBisection += 1
				stringHyperList.append(("b", [intervalUtils.volume(lHyp), intervalUtils.volume(rHyp)]))
				#print ("lHyp", lHyp)
				orderIndex = (orderIndex + 1)%lenV
				start = time.time()
				lFeas = intervalUtils.checkExistenceOfSolution(model, lHyp.transpose(), kAlpha)
				end = time.time()
				totalGSTime += end - start
				numGs += 1
				stringHyperList.append(("g", intervalUtils.volume(lFeas[1])))
				#print ("lFeas", lFeas)
				#print ("rHyp", rHyp)
				start = time.time()
				rFeas = intervalUtils.checkExistenceOfSolution(model, rHyp.transpose(), kAlpha)
				end = time.time()
				totalGSTime += end - start

				#print ("curChild", curChild)
				#print ("curChild.nextChild", curChild.nextChild)
				curChild.nextChild[0].operation = "k"
				curChild.nextChild[1].operation = "k"
				if lFeas[0] or lFeas[1] is not None:
					leftInfo = HyperInfo(np.copy(lFeas[1]))
					if lFeas[0]:
						curChild.nextChild[0].operation = "kunique"
					curChild.nextChild[0].nextChild.append(leftInfo)
				else:
					curChild.nextChild[0].operation = "knoSol"

				if rFeas[0] or rFeas[1] is not None:
					rightInfo = HyperInfo(np.copy(rFeas[1]))
					if rFeas[0]:
						curChild.nextChild[1].operation = "kunique"
					curChild.nextChild[1].nextChild.append(rightInfo)
				else:
					curChild.nextChild[1].operation = "knoSol"

				numGs += 1
				stringHyperList.append(("g", intervalUtils.volume(rFeas[1])))
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
						if numSolutions == "all" or len(uniqueHypers) < numSolutions:
							#print ("lFeas[1:]", lFeas[1:])
							addToSolutions(model, uniqueHypers, lFeas[1:])
						else:
							return allHyperInfos[0]
					if rFeas[0]:
						#print ("addedHyper", rHyp)
						if numSolutions == "all" or len(uniqueHypers) < numSolutions:
							#print ("rFeas", rFeas[1:])
							addToSolutions(model, uniqueHypers, rFeas[1:])
						else:
							return rootHyper

					if lFeas[0] == False and lFeas[1] is not None:
						hypForBisection = lFeas[1]
						curChild = curChild.nextChild[0].nextChild[0]
					elif rFeas[0] == False and rFeas[1] is not None:
						hypForBisection = rFeas[1]
						curChild = curChild.nextChild[1].nextChild[0]
					else:
						hypForBisection = None

				else:
					stackList.append(lFeas[1])
					allHyperInfos.append(curChild.nextChild[0].nextChild[0])
					indicesPopped.append(len(allHyperInfos)-1)
					stackList.append(rFeas[1])
					allHyperInfos.append(curChild.nextChild[1].nextChild[0])
					indicesPopped.append(len(allHyperInfos)-1)
					hypForBisection = None
		else:
			allHyperInfos[indexPopped].operation += "noSol"

	return rootHyper




# Apply gauss seidel and linear programming to refine the hyperrectangle
# Keep repeating as long as the percentage of volume reduction of the
# hyperrectangle is atleast volRedThreshold. model indicates the problem
# we are trying to solve
# return (True, hyper) if hyperRectangle contains a unique
# solution and hyper maybe smaller than hyperRectangle containing the solution
# return (False, None) if hyperRectangle contains no solution
# return (False, hyper) if hyperRectangle may contain more
# than 1 solution and hyper maybe smaller than hyperRectangle containing the solutions
def ifFeasibleHyper(hyperRectangle, volRedThreshold, model, kAlpha = 1.0, useLp = True,allHyperInfos = None,indexPopped = None):
	global numBisection, numLp, numGs, numSingleKill, numDoubleKill
	global stringHyperList
	global totalLPTime, totalGSTime
	global numLpCalls, numSuccessLpCalls, numUnsuccessLpCalls
	global numSaddle, numAnyRegion
	lenV = hyperRectangle.shape[0]
	#print ("hyperRectangle")
	#print (hyperRectangle)
	iterNum = 0
	while True:
		#print ("iterNum", iterNum)
		print ("hyperRectangle")
		print (hyperRectangle)

		newHyperRectangle = np.copy(hyperRectangle)

		if useLp:
			#print ("startlp")
			numTotalLp, numSuccessLp, numUnsuccessLp, nSaddle, nAnyRegion = 0, 0, 0, 0, 0
			start = time.time()
			feasible, newHyperRectangle, numTotalLp, numSuccessLp, numUnsuccessLp, nSaddle, nAnyRegion = model.linearConstraints(newHyperRectangle)
			#feasible, newHyperRectangle, numTotalLp, numSuccessLp, numUnsuccessLp = model.linearConstraints(newHyperRectangle)
			end = time.time()
			numSaddle += nSaddle
			numAnyRegion += nAnyRegion
			totalLPTime += end - start
			numLpCalls += numTotalLp
			numSuccessLpCalls += numSuccessLp
			numUnsuccessLpCalls += numUnsuccessLp
			#print ("endlp")
			numLp += 1
			if feasible:
				vol = intervalUtils.volume(newHyperRectangle)
			else:
				vol = None
			stringHyperList.append(("l", vol))
			#print ("newHyperRectangle ", newHyperRectangle)
			#print ("feasible", feasible)
			print ("newHyperRectangle")
			for i in range(lenV):
				print (newHyperRectangle[i][0], newHyperRectangle[i][1])
			#print ("newHyperRectangle ", newHyperRectangle)
			if feasible == False:
				#print ("LP not feasible", hyperRectangle)
				return (False, None)

			for i in range(lenV):
				if newHyperRectangle[i,0] < hyperRectangle[i,0]:
					newHyperRectangle[i,0] = hyperRectangle[i,0]
				if newHyperRectangle[i,1] > hyperRectangle[i,1]:
					newHyperRectangle[i,1] = hyperRectangle[i,1]

			if indexPopped is not None:
				allHyperInfos[indexPopped].operation = "l"
				allHyperInfos[indexPopped].nextChild.append(HyperInfo(newHyperRectangle))
				allHyperInfos[indexPopped] = allHyperInfos[indexPopped].nextChild[0]

		start = time.time()
		#First apply gauss seidel
		#print ("applying gauss-seidel")
		kResult = intervalUtils.checkExistenceOfSolution(model, newHyperRectangle.transpose(), kAlpha)
		end = time.time()
		totalGSTime += (end - start)
		numGs += 1
		stringHyperList.append(("g", intervalUtils.volume(kResult[1])))
		#print ("kResult")
		#print (kResult)
		
		#If gauss-seidel interval says there is unique solution
		#in hyperrectangle or no solution, then return
		
		#Unique solution or no solution
		if kResult[0] or kResult[1] is None:
			#print ("uniqueHyper", hyperRectangle)
			return kResult

		#print "kResult"
		#print kResult
		
		#If Gauss-Seidel cannot make a decision apply linear programming
		newHyperRectangle = kResult[1]			 

		hyperVol = intervalUtils.volume(hyperRectangle)
		#hyperVol = hyperVol**(1.0/lenV)

		newHyperVol = intervalUtils.volume(newHyperRectangle)
		#newHyperVol = newHyperVol**(1.0/lenV)

		propReduc = (hyperVol - newHyperVol)/hyperVol
		#print ("propReduc", propReduc)
		
		# If the proportion of volume reduction is not atleast
		# volRedThreshold then return
		if math.isnan(propReduc) or propReduc < volRedThreshold:
			#return (False, kResult[1])
			return (False, newHyperRectangle)
		hyperRectangle = newHyperRectangle
		iterNum+=1
	


def addToSolutions(model, allHypers, solHyper):
	#print ("solHyper[0]", solHyper[0][0])
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

			#print ("unionHyper", unionHyper)
			feasibility = intervalUtils.checkExistenceOfSolution(model, unionHyper.transpose())
			if feasibility[0]:
				foundOverlap = True
				allHypers[hi][0] = unionHyper
				break

	if not(foundOverlap):
		allHypers.append(solHyper)

def findAndIgnoreNewtonSoln(model, minVal, maxVal, numSolutions, numTrials = 10):
	lenV = len(model.bounds)
	allHypers = []
	solutionsFoundSoFar = []
	bounds = model.bounds
	overallHyper = np.zeros((lenV,2))
	for i in range(lenV):
		overallHyper[i,0] = bounds[i][0]
		overallHyper[i,1] = bounds[i][1]
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
					kResult = intervalUtils.checkExistenceOfSolution(model,hyperWithUniqueSoln.transpose())
					if kResult[0] == False and kResult[1] is not None:
						diff[startingIndex] = diff[startingIndex]/2.0
						startingIndex = (startingIndex + 1)%lenV
					else:
						end = time.time()
						#print ("Preprocess: found unique hyper ", hyperWithUniqueSoln, "in", end - start, "after", numFailures, "failures")
						start = time.time()
						numFailures = 0
						allHypers.append(hyperWithUniqueSoln)
						break
	#print ("total num hypers found in ", numTrials, " trials is ", len(allHypers))
	return (allHypers, solutionsFoundSoFar)


def schmittTrigger(inputVoltage, lpThreshold, numSolutions = "all", newtonHypers = True, useLp = True):
	global numBisection, numLp, numGs, numSingleKill, numDoubleKill
	global stringHyperList
	global totalLPTime, totalGSTime
	global numLpCalls, numSuccessLpCalls, numUnsuccessLpCalls
	global numSaddle, numAnyRegion
	numBisection, numLp, numGs, numSingleKill, numDoubleKill = 0, 0, 0, 0, 0
	totalLPTime, totalGSTime = 0, 0
	numLpCalls, numSuccessLpCalls, numUnsuccessLpCalls = 0, 0, 0
	numSaddle, numAnyRegion = 0, 0
	stringHyperList = []

	#load the schmitt trigger model
	#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
	modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
	#model = schmittMosfet.SchmittMosfet(modelParam = modelParam, inputVoltage = inputVoltage)
	model = schmittMosfet.SchmittMosfetMark(modelParam = modelParam, inputVoltage = inputVoltage)

	# test val
	fVal = model.f(np.array([0.0, 0.0, 0.2]))
	print ("fVal", fVal[0], fVal[1], fVal[2])

	startExp = time.time()
	lenV = 3
	
	#in case the user wants to specify the ordering of bisection
	#by default this is a useless variable
	indexChoiceArray = []
	for i in range(lenV):
		indexChoiceArray.append(i)
	print ("indexChoiceArray", indexChoiceArray)
	bounds = model.bounds
	print ("bounds ", bounds)

	numStages = 1

	#Apply Newton's preprocessing step
	allHypers = []
	solutionsFoundSoFar = []
	'''if newtonHypers:
		allHypers, solutionsFoundSoFar = findAndIgnoreNewtonSoln(model, bounds[0][0], bounds[0][1], numSolutions=numSolutions, numTrials = numStages*100)
		newtonHypers = np.copy(allHypers)

	#the solver's main loop
	solverLoop(allHypers, model, lpThreshold, bisectFun=bisectMax, numSolutions=numSolutions, orderingArray=indexChoiceArray, useLp=useLp)
	print ("allHypers")
	print (allHypers)
	print ("numSolutions", len(allHypers))'''


	#Debugging
	'''newtonSol = intervalUtils.newton(model,np.array([1.8, 1.4, 1.8]))
	print ("newtonSol", newtonSol)'''

	'''hyper = np.array([[ 1.575 - 1e-9 ,  1.8 + 1e-9],
       [ 1.39999908 - 1e-9  ,  1.8 + 1e-9],
       [ 1.61538929 - 1e-9 , 1.8 + 1e-9]])

	feasibility = ifFeasibleHyper(hyper, lpThreshold,model,useLp=useLp)
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
	if numLp == 0:
		avgLPTime = 0.0
	else:
		avgLPTime = (totalLPTime*1.0)/numLp
	if numGs == 0:
		avgGSTime = 0.0
	else:
		avgGSTime = (totalGSTime*1.0)/numGs
	globalVars = [numBisection, numLp, numGs, numSingleKill, numDoubleKill, totalLPTime, totalGSTime, avgLPTime, avgGSTime, stringHyperList]
	return allHypers, globalVars

def rambusOscillator(modelType, numStages, g_cc, lpThreshold, numSolutions="all" , newtonHypers=True, useLp=True, kAlpha = 1.0):
	global numBisection, numLp, numGs, numSingleKill, numDoubleKill
	global stringHyperList
	global totalGSTime, totalLPTime
	global numLpCalls, numSuccessLpCalls, numUnsuccessLpCalls
	global numSaddle, numAnyRegion
	numBisection, numLp, numGs, numSingleKill, numDoubleKill = 0, 0, 0, 0, 0
	totalGSTime, totalLPTime = 0, 0
	numLpCalls, numSuccessLpCalls, numUnsuccessLpCalls = 0, 0, 0
	numSaddle, numAnyRegion = 0, 0
	stringHyperList = []
	a = -5.0
	model = rambusTanh.RambusTanh(modelParam = a, g_cc = g_cc, g_fwd = 1.0, numStages=numStages)
	
	if modelType == "mosfet":
		#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
		#modelParam = [-0.25, 0.25, 1.0, 1.0, -0.5, 1.0]
		#modelParam = [-0.4, 0.4, 1.8, 1.5, -0.5, 8/3.0]
		modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
		model = rambusMosfet.RambusMosfetMark(modelParam = modelParam, g_cc = g_cc, g_fwd = 1.0, numStages = numStages)
	
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
	bounds = model.bounds
	#print ("bounds ", bounds)
	
	'''hyper = np.array([[ 0.9,  1.8 ],
       [ 0.0,  1.8],
       [ 0.0,  1.8],
       [ 0.0,  1.8]])

	feas = ifFeasibleHyper(hyper, lpThreshold,model)
	print ("feas", feas)'''

	'''sol = np.array([0.04369848, 0.56992351, 0.80192295, 0.83295818,
			1.77673741, 1.03976852, 1.74566377, 1.03117171,
			0.91452938, 0.86123781, 0.01632339, 0.55631079])

	print ("model.f", model.f(sol))'''
	
	allHypers = []
	solutionsFoundSoFar = []
	if newtonHypers:
		allHypers, solutionsFoundSoFar = findAndIgnoreNewtonSoln(model, bounds[0][0], bounds[0][1], numSolutions=numSolutions, numTrials = numStages*100)
		newtonHypers = np.copy(allHypers)

	hyperInfo = solverLoop(allHypers, model, lpThreshold, bisectFun=bisectMax, numSolutions=numSolutions, orderingArray=indexChoiceArray, useLp=useLp, kAlpha=kAlpha)
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

	#print ("hyperInfo")
	#print (hyperInfo)
	#printHyperInfo(hyperInfo)
	filename = "rambus_"+modelType + "_numStages_" + str(numStages) + "_gcc_" + str(g_cc) + "_useLp_" + str(useLp) + ".pkl"
	theFile = open(filename, "wb")
	pickle.dump(hyperInfo, theFile)
	theFile.close()'''

	'''rootPaths = []
	costHyperInfo(root = hyperInfo, paths = rootPaths, pathIndex = 0)
	shortestPath = None
	pathLength = float("inf")
	for path in rootPaths:
		print ("path", path)
		if len(path) < pathLength:
			shortestPath = path
			pathLength = len(path)
	print ("shortestPath", shortestPath)'''
	'''for i in range(len(hyperInfo.nextChild)):
		print ("child #", i, hyperInfo.nextChild[i])'''

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
	#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
	if numLp == 0:
		avgLPTime = 0.0
	else:
		avgLPTime = (totalLPTime*1.0)/numLp
	avgGSTime = (totalGSTime*1.0)/numGs
	globalVars = [numBisection, numLp, numGs, numSingleKill, numDoubleKill, totalLPTime, totalGSTime, avgLPTime, avgGSTime, stringHyperList]
	print ("numBisection", numBisection, "numLp", numLp, "numGs", numGs)
	print ("avgLPTime", avgLPTime)
	print ("numSingleKill", numSingleKill, "numDoubleKill", numDoubleKill)
	print ("numLpCalls", numLpCalls, "numSuccessLpCalls", numSuccessLpCalls, "numUnsuccessLpCalls", numUnsuccessLpCalls)
	print ("numAnyRegions", numAnyRegion, "numSaddles", numSaddle)
	return allHypers, globalVars

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



def singleVariableInequalities(problemType, volRedThreshold, newtonHypers=False, useLp=True):
	global numBisection, numLp, numGs, numSingleKill, numDoubleKill
	global stringHyperList
	global totalGSTime, totalLPTime
	numBisection, numLp, numGs, numSingleKill, numDoubleKill = 0, 0, 0, 0, 0
	totalGSTime, totalLPTime = 0, 0
	stringHyperList = []

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


	lenV = 1
	indexChoiceArray = [0]

	'''newtonSol = intervalUtils.newton(model,np.array([0.8]))
	print ("newtonSol", newtonSol)
	#[[ -5.31508875e-09   5.31508875e-09]]
	#feasibility = ifFeasibleHyper(np.array([[-0.03333433, 4.1887911 ]]), volRedThreshold,model)
	feasibility = ifFeasibleHyper(np.array([[ xBound[0], xBound[1]]]), volRedThreshold,model)
	print ("feasibility", feasibility)'''

	allHypers = []
	solutionsFoundSoFar = []
	if newtonHypers:
		allHypers, solutionsFoundSoFar = findAndIgnoreNewtonSoln(model, xBound[0], xBound[1], numSolutions=numSolutions, numTrials = 100)
		newtonHypers = np.copy(allHypers)

	solverLoop(allHypers, model, volRedThreshold, bisectFun=bisectMax, numSolutions = "all", orderingArray=indexChoiceArray,useLp=useLp)
	print ("allHypers")
	print (allHypers)
	print ("numSolutions", len(allHypers))
	
	validIntervals = validSingleVariableInterval(allHypers, model)
	print ("validIntervals", validIntervals)
	endExp = time.time()
	#print ("time taken", endExp - startExp)

	if numLp == 0:
		avgLPTime = 0.0
	else:
		avgLPTime = (totalLPTime*1.0)/numLp
	avgGSTime = (totalGSTime*1.0)/numGs
	globalVars = [numBisection, numLp, numGs, numSingleKill, numDoubleKill, totalLPTime, totalGSTime, avgLPTime, avgGSTime, stringHyperList]
	return validIntervals, globalVars

def separateHyperSpaceTest():
	lenV = 3
	biggerHyper = np.zeros((lenV,2))
	smallerHyper = np.zeros((lenV,2))
	for i in range(lenV):
		biggerHyper[i,:] = [-1.0, 1.0]
		smallerHyper[i,:] = [-0.25, 0.5]

	smallerHyperList = separateHyperSpace(biggerHyper, smallerHyper)
	print ("number of total smallHypers", len(smallerHyperList))
	volSmallHypers = 0.0
	for smallHyper in smallerHyperList:
		volSmallHypers += intervalUtils.volume(smallHyper)
		print ("smallHyper")
		print (smallHyper) 

	print ("volume of biggerHyper", intervalUtils.volume(biggerHyper))
	print ("total volume of smallerHypers", volSmallHypers)


if __name__ == "__main__":
	start = time.time()
	#allHypers, globalVars = rambusOscillator(modelType="mosfet", numStages=2, g_cc=4.0, lpThreshold=1.0, numSolutions="all" , newtonHypers=None, useLp=True)
	#print ("allHypers", allHypers)
	schmittTrigger(inputVoltage = 1.8, lpThreshold = 1.0, numSolutions = "all", newtonHypers = False, useLp = False)
	#singleVariableInequalities(problemType="meti18", volRedThreshold=1.0, newtonHypers=False, useLp=True)
	#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
	#print ("numSolutions", len(allHypers))
	#separateHyperSpaceTest()
	end = time.time()
	print ("time taken", end - start)
