import numpy as np
import time
import copy
import intervalUtils
import rambusTanh
import rambusMosfet
import flyspeckProblems
import metiProblems
import schmittMosfet
import rambusUtils as rUtils
import random
import math
import circuit


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
def solverLoop(uniqueHypers, solutionsFoundSoFar, model, volRedThreshold, bisectFun, orderingArray=None, useLp=True):
	global numBisection, numLp, numGs, numSingleKill, numDoubleKill
	global stringHyperList
	global totalLPTime, totalGSTime
	lenV = len(model.bounds)
	hyperRectangle = np.zeros((lenV,2))

	for i in range(lenV):
		hyperRectangle[i,0] = model.bounds[i][0]
		hyperRectangle[i,1] = model.bounds[i][1]

	'''hyperRectangle = np.array([[ 0.  ,  0.45],
       [ 0.45,  0.9 ],
       [ 0.45,  0.9 ],
       [ 0.45,  0.9 ],
       [ 1.35,  1.8 ],
       [ 0.9 ,  1.35],
       [ 0.9 ,  1.8 ],
       [ 0.9 ,  1.8 ],
       [ 0.9 ,  1.8 ],
       [ 0.  ,  0.9 ],
       [ 0.  ,  0.9 ],
       [ 0.  ,  0.9 ]])'''

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
	while len(stackList) > 0:
		#print ("len stack", len(stackList))

		#pop the hyperrectangle
		hyperPopped = stackList.pop(-1)
		'''print ("hyperPopped")
		#print ("hyperPopped", hyperPopped)
		for i in range(lenV):
			print (hyperPopped[i][0], hyperPopped[i][1])
		print ("hyperPopped", hyperPopped)'''
		
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

		#Apply the Gauss-Seidel + Lp loop
		feasibility = ifFeasibleHyper(hyperPopped, volRedThreshold, model, useLp)
		#print ("feasibility", feasibility)
		
		if feasibility[0]:
			#If the Gauss_Seidel + Lp loop indicate uniqueness, then add the hyperrectangle
			#to our list
			addToSolutions(model, uniqueHypers, solutionsFoundSoFar, feasibility[1])

		elif feasibility[0] == False and feasibility[1] is not None:
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
				'''if np.less_equal(hypForBisection[:,1] - hypForBisection[:,0], np.ones((lenV))*0.001).all():
					feasibility = ifFeasibleHyper(hypForBisection, volRedThreshold, model, useLp)
					if feasibility[0]:
						addToSolutions(model, uniqueHypers, solutionsFoundSoFar, feasibility[1])
						break
					elif feasibility[0] == False and feasibility[1] is None:
						break'''
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
						addToSolutions(model, uniqueHypers, solutionsFoundSoFar, lFeas[1])
					if rFeas[0]:
						#print ("addedHyper", rHyp)
						addToSolutions(model, uniqueHypers, solutionsFoundSoFar, rFeas[1])

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




# Apply gauss seidel and linear programming to refine the hyperrectangle
# Keep repeating as long as the percentage of volume reduction of the
# hyperrectangle is atleast volRedThreshold. model indicates the problem
# we are trying to solve
# return (True, hyper) if hyperRectangle contains a unique
# solution and hyper maybe smaller than hyperRectangle containing the solution
# return (False, None) if hyperRectangle contains no solution
# return (False, hyper) if hyperRectangle may contain more
# than 1 solution and hyper maybe smaller than hyperRectangle containing the solutions
def ifFeasibleHyper(hyperRectangle, volRedThreshold,model,useLp=True):
	global numBisection, numLp, numGs, numSingleKill, numDoubleKill
	global stringHyperList
	global totalLPTime, totalGSTime
	lenV = hyperRectangle.shape[0]
	#print ("hyperRectangle")
	#print (hyperRectangle)
	iterNum = 0
	while True:
		#print ("iterNum", iterNum)
		#print ("hyperRectangle")
		#print (hyperRectangle)

		newHyperRectangle = np.copy(hyperRectangle)

		if useLp:
			#print ("startlp")
			start = time.time()
			feasible, newHyperRectangle = model.linearConstraints(newHyperRectangle)
			end = time.time()
			totalLPTime += end - start
			#print ("endlp")
			numLp += 1
			if feasible:
				vol = intervalUtils.volume(newHyperRectangle)
			else:
				vol = None
			stringHyperList.append(("l", vol))
			#print ("newHyperRectangle ", newHyperRectangle)
			#print ("feasible", feasible)
			'''print ("newHyperRectangle")
			for i in range(lenV):
				print (newHyperRectangle[i][0], newHyperRectangle[i][1])'''
			#print ("newHyperRectangle ", newHyperRectangle)
			if feasible == False:
				#print ("LP not feasible", hyperRectangle)
				return (False, None)

			for i in range(lenV):
				if newHyperRectangle[i,0] < hyperRectangle[i,0]:
					newHyperRectangle[i,0] = hyperRectangle[i,0]
				if newHyperRectangle[i,1] > hyperRectangle[i,1]:
					newHyperRectangle[i,1] = hyperRectangle[i,1]

		start = time.time()
		#First apply gauss seidel
		#print ("applying gauss-seidel")
		kResult = intervalUtils.checkExistenceOfSolution(model,newHyperRectangle.transpose())
		end = time.time()
		totalGSTime += (end - start)
		numGs += 1
		stringHyperList.append(("g", intervalUtils.volume(kResult[1])))
		#print ("kResult")
		#print (kResult)
		
		#If gauss-seidel interval says there is unique solution
		#in hyperrectangle or no solution, then return
		
		#Unique solution
		if kResult[0]:
			#print ("uniqueHyper", hyperRectangle)
			return (True, kResult[1])

		#No Solution
		if kResult[0] == False and kResult[1] is None:
			#print ("K operator not feasible", hyperRectangle)
			return (False, None)
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
	

# check if a solution already exists in 
# the list of solutions and if not, add it
# to the list of solutions
def addToSolutions(model, allHypers, solutionsFoundSoFar, solHyper):
	exampleSoln = (solHyper[:,0] + solHyper[:,1])/2.0
	finalSoln = intervalUtils.newton(model,exampleSoln)
	if finalSoln[0]:
		lenV = len(finalSoln[1])
		solExists = False
		for existingSol in solutionsFoundSoFar:
			if np.less_equal(np.absolute(existingSol - finalSoln[1]), np.ones(finalSoln[1].shape)*1e-14).all():
				# solution already exists
				solExists = True
				break
		if not(solExists):
			solutionsFoundSoFar.append(finalSoln[1])
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
						if model.solver is not None:
							model.ignoreHyperInZ3(hyperWithUniqueSoln)
						break
	#print ("total num hypers found in ", numTrials, " trials is ", len(allHypers))
	return (allHypers, solutionsFoundSoFar)


def schmittTrigger(inputVoltage, lpThreshold, numSolutions = "all", newtonHypers = True, useLp = True):
	global numBisection, numLp, numGs, numSingleKill, numDoubleKill
	global stringHyperList
	global totalLPTime, totalGSTime
	numBisection, numLp, numGs, numSingleKill, numDoubleKill = 0, 0, 0, 0, 0
	totalLPTime, totalGSTime = 0, 0
	stringHyperList = []

	#load the schmitt trigger model
	#modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn]
	modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
	#model = schmittMosfet.SchmittMosfet(modelParam = modelParam, inputVoltage = inputVoltage)
	model = schmittMosfet.SchmittMosfetMark(modelParam = modelParam, inputVoltage = inputVoltage)

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
	if newtonHypers:
		allHypers, solutionsFoundSoFar = findAndIgnoreNewtonSoln(model, bounds[0][0], bounds[0][1], numSolutions=numSolutions, numTrials = numStages*100)
		newtonHypers = np.copy(allHypers)

	#the solver's main loop
	solverLoop(allHypers, solutionsFoundSoFar, model, lpThreshold, bisectFun=bisectMax, orderingArray=indexChoiceArray, useLp=useLp)
	print ("allHypers")
	print (allHypers)
	print ("numSolutions", len(allHypers))


	#Debugging
	'''newtonSol = intervalUtils.newton(model,np.array([0.0, 0.0, 0.415]))
	print ("newtonSol", newtonSol)

	hyper = np.array([[ 0.9 ,  1.35],
       [ 0.  ,  0.45],
       [ 0.9 ,  1.8 ]])

	feasibility = ifFeasibleHyper(hyper, lpThreshold,model)
	print ("feas", feasibility)'''

	
	# categorize solutions found
	sampleSols, rotatedSols, stableSols, unstableSols = rUtils.categorizeSolutions(allHypers,model)

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
	avgGSTime = (totalGSTime*1.0)/numGs
	globalVars = [numBisection, numLp, numGs, numSingleKill, numDoubleKill, totalLPTime, totalGSTime, avgLPTime, avgGSTime, stringHyperList]
	return allHypers, globalVars

def rambusOscillator(modelType, numStages, g_cc, lpThreshold, numSolutions="all" , newtonHypers=True, useLp=True):
	global numBisection, numLp, numGs, numSingleKill, numDoubleKill
	global stringHyperList
	global totalGSTime, totalLPTime
	numBisection, numLp, numGs, numSingleKill, numDoubleKill = 0, 0, 0, 0, 0
	totalGSTime, totalLPTime = 0, 0
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

	'''hyper = np. array([[ 0.033104943613681291, 0.054206798951633238 ],
       [ 0.51972564406263211, 0.57419060711979319],
       [ 0.77279573980872096, 0.83203199010367601],
       [ 0.82437266623951189, 0.84174582030759826],
       [ 1.7751290274902156, 1.7783228407959957],
       [ 1.0049761474100403, 1.0739281094755144],
       [ 1.7339918205341063, 1.7575937176641356],
       [ 0.98574297803212851, 1.0730225275030481],
       [ 0.90000000000000002, 0.93114500285085533  ],
       [ 0.84821701993957588, 0.87401903180675899],
       [ 0.015248184200376956, 0.017414119527335674],
       [ 0.51959119530035003, 0.59360505677576625]])'''

	'''hyper = np. array([[ 0.033104943613681291, 0.054206798951633238 ],
       [ 0.51972564406263211, 0.57419060711979319],
       [ 0.77279573980872096, 0.83203199010367601],
       [ 0.82437266623951189, 0.84174582030759826],
       [ 1.7751290274902156, 1.7783228407959957],
       [ 1.0049761474100403, 1.0739281094755144],
       [ 1.7339918205341063, 1.7575937176641356],
       [ 0.98574297803212851, 1.0730225275030481],
       [ 0.90000000000000002, 0.93114500285085533  ],
       [ 0.84821701993957588, 0.87401903180675899],
       [ 0.015248184200376956, 0.017414119527335674],
       [ 0.51959119530035003, 0.59360505677576625]])'''
	
	'''hyper = np.array([[ 0.03310494,  0.0542068 ],
       [ 0.51972564,  0.57419061],
       [ 0.77279574,  0.83203199],
       [ 0.82437267,  0.84174582],
       [ 1.77512903,  1.77832284],
       [ 1.00497615,  1.07392811],
       [ 1.73399182,  1.75759372],
       [ 0.98574298,  1.07302253],
       [ 0.9       ,  0.931145  ],
       [ 0.84821702,  0.87401903],
       [ 0.01524818,  0.01741412],
       [ 0.5195912 ,  0.59360506]])'''

	'''feas = ifFeasibleHyper(hyper, lpThreshold,model)
	print ("feas", feas)

	sol = np.array([0.04369848, 0.56992351, 0.80192295, 0.83295818,
			1.77673741, 1.03976852, 1.74566377, 1.03117171,
			0.91452938, 0.86123781, 0.01632339, 0.55631079])

	print ("model.f", model.f(sol))'''
	
	allHypers = []
	solutionsFoundSoFar = []
	if newtonHypers:
		allHypers, solutionsFoundSoFar = findAndIgnoreNewtonSoln(model, bounds[0][0], bounds[0][1], numSolutions=numSolutions, numTrials = numStages*100)
		newtonHypers = np.copy(allHypers)

	solverLoop(allHypers, solutionsFoundSoFar, model, lpThreshold, bisectFun=bisectMax, orderingArray=indexChoiceArray, useLp=useLp)
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
	#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
	if numLp == 0:
		avgLPTime = 0.0
	else:
		avgLPTime = (totalLPTime*1.0)/numLp
	avgGSTime = (totalGSTime*1.0)/numGs
	globalVars = [numBisection, numLp, numGs, numSingleKill, numDoubleKill, totalLPTime, totalGSTime, avgLPTime, avgGSTime, stringHyperList]
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



def singleVariableInequalities(problemType, volRedThreshold, numSolutions="all", newtonHypers=True, useLp=True):
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

	solverLoop(allHypers, solutionsFoundSoFar, model, volRedThreshold, bisectFun=bisectMax, orderingArray=indexChoiceArray,useLp=useLp)
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


if __name__ == "__main__":
	start = time.time()
	#allHypers, globalVars = rambusOscillator(modelType="mosfet", numStages=6, g_cc=4.0, lpThreshold=1.0, numSolutions="all" , newtonHypers=None, useLp=True)
	#print ("allHypers", allHypers)
	#schmittTrigger(inputVoltage = 1.0, lpThreshold = 0.05, numSolutions = "all", newtonHypers = False, useLp = True)
	singleVariableInequalities(problemType="meti18", volRedThreshold=1.0, numSolutions="all", newtonHypers=False, useLp=True)
	#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
	#print ("numSolutions", len(allHypers))
	end = time.time()
	print ("time taken", end - start)
