import pickle
from intervalUtils import *
import time
from prototype2 import addToSolutions

class HyperInfo:
	def __init__(self, hyper, operation = None):
		self.hyper = hyper
		self.operation = operation
		self.nextChild = []
		#print ("self.nextChild", self.nextChild)

	def __str__(self):
		if self.hyper is None:
			return "None"
		if self.operation is None:
			return "hyper " + str(self.hyper)
		stringVal = "hyper " + str(self.hyper) + ", operation " + self.operation + ", numChildren " + str(len(self.nextChild)) 
		return stringVal


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
def solverLoopTrace(uniqueHypers, model, statVars, volRedThreshold, bisectFun, numSolutions = "all", useLp=True, kAlpha = 1.0):
	lenV = len(model.bounds)
	hyperRectangle = np.zeros((lenV,2))

	for i in range(lenV):
		hyperRectangle[i,0] = model.bounds[i][0]
		hyperRectangle[i,1] = model.bounds[i][1]

	statVars['stringHyperList'].append(("i", volume(hyperRectangle)))
	rootHyper = HyperInfo(np.copy(hyperRectangle))
	
	start = time.time()
	feas = checkExistenceOfSolution(model,hyperRectangle.transpose(),kAlpha)
	end = time.time()
	statVars['totalGSTime'] += end - start
	statVars['numGs'] += 1
	rootHyper.operation = "k"
	if feas[1] is not None:
		rootHyper.nextChild.append(HyperInfo(np.copy(feas[1])))
	else:
		rootHyper.nextChild.append(HyperInfo(None))

	if feas[0] or feas[1] is None:
		return rootHyper
	statVars['stringHyperList'].append(("g", volume(feas[1])))


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
		feasibility = ifFeasibleHyperTrace(hyperPopped, statVars, volRedThreshold, model, kAlpha, useLp, allHyperInfos, indexPopped)
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
			curChild = child
			while hypForBisection is not None:
				#print ("hypForBisection", hypForBisection)
				curChild.operation = "b"
				lHyp, rHyp = bisectFun(hypForBisection, None)
				curChild.nextChild.append(HyperInfo(np.copy(lHyp)))
				curChild.nextChild.append(HyperInfo(np.copy(rHyp)))
				statVars['numBisection'] += 1
				statVars['stringHyperList'].append(("b", [volume(lHyp), volume(rHyp)]))
				#print ("lHyp", lHyp)
				start = time.time()
				lFeas = checkExistenceOfSolution(model, lHyp.transpose(), kAlpha)
				end = time.time()
				statVars['totalGSTime'] += end - start
				statVars['numGs'] += 1
				statVars['stringHyperList'].append(("g", volume(lFeas[1])))
				#print ("lFeas", lFeas)
				#print ("rHyp", rHyp)
				start = time.time()
				rFeas = checkExistenceOfSolution(model, rHyp.transpose(), kAlpha)
				end = time.time()
				statVars['totalGSTime'] += end - start

		
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

				statVars['numGs'] += 1
				statVars['stringHyperList'].append(("g", volume(rFeas[1])))
				#print ("rFeas", rFeas)
				if lFeas[0] or rFeas[0] or (lFeas[0] == False and lFeas[1] is None) or (rFeas[0] == False and rFeas[1] is None):
					if lFeas[0] and rFeas[0]:
						statVars['numDoubleKill'] += 1
					elif lFeas[0] == False and lFeas[1] is None and rFeas[0] == False and rFeas[1] is None:
						statVars['numDoubleKill'] += 1
					else:
						statVars['numSingleKill'] += 1
					
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
def ifFeasibleHyperTrace(hyperRectangle, statVars, volRedThreshold, model, kAlpha = 1.0, useLp = True,allHyperInfos = None,indexPopped = None):
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
			start = time.time()
			feasible, newHyperRectangle, numTotalLp, numSuccessLp, numUnsuccessLp = model.linearConstraints(newHyperRectangle)
			end = time.time()
			statVars['totalLPTime'] += end - start
			statVars['numLpCalls'] += numTotalLp
			statVars['numSuccessLpCalls'] += numSuccessLp
			statVars['numUnsuccessLpCalls'] += numUnsuccessLp
			#print ("endlp")
			statVars['numLp'] += 1
			if feasible:
				vol = intervalUtils.volume(newHyperRectangle)
			else:
				vol = None
			startVars['stringHyperList'].append(("l", vol))
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
		kResult = checkExistenceOfSolution(model, newHyperRectangle.transpose(), kAlpha)
		end = time.time()
		statVars['totalGSTime'] += (end - start)
		statVars['numGs'] += 1
		statVars['stringHyperList'].append(("g", volume(kResult[1])))
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

		hyperVol = volume(hyperRectangle)
		#hyperVol = hyperVol**(1.0/lenV)

		newHyperVol = volume(newHyperRectangle)
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
	



def costHyperInfo(root, paths, pathIndex):
	if root is None:
		return

	if pathIndex == len(paths):
		paths.append([])
	#print ("pathIndex", pathIndex)
	#print ("paths", paths)
	#print ("root", root)

	if root.operation is not None:
		paths[pathIndex].append(root.operation)
	#print ("paths after rootAdded")
	#print (paths)
	previousPath = paths[pathIndex].copy()
	#print ("previousPath before", previousPath)
	if root.operation == "knoSol" or root.operation == "kunique":
		previousPath = previousPath[:-1]

	#print ("previousPath after", previousPath)

	#print ("len(root.nextChild)", len(root.nextChild))
	if len(root.nextChild) == 1:
		#print ("nextIteration child", root.nextChild[0])
		#print ("nextIteration pathIndex", pathIndex)
		#print ("nextIteration paths", paths)
		costHyperInfo(root.nextChild[0], paths, pathIndex)
	elif len(root.nextChild) == 2:
		#print ("nextIteration left child", root.nextChild[0])
		#print ("nextIteration pathIndex", pathIndex)
		#print ("nextIteration paths", paths)
		costHyperInfo(root.nextChild[0], paths, pathIndex)
		paths.append(previousPath)
		#print ("nextIteration right child", root.nextChild[1])
		#print ("nextIteration pathIndex", len(paths)-1)
		#print ("nextIteration paths", paths)
		costHyperInfo(root.nextChild[1], paths, len(paths)-1)


def printHyperInfo(root):
	if root is None:
		return

	queueList = []
	queueList.append((root, 1))

	while len(queueList) > 0:
		node, nodeCount = queueList[0]
		#print ("node")
		#print (node)
		#print ("nodeCount", nodeCount)
		
		while nodeCount > 0:
			#print ("start while")
			'''print ("node info")
			print (node.hyper)
			print (node.operation)
			print (len(node.nextChild))'''
			node, _ = queueList[0]
			print (node)
			del queueList[0]

			for child in node.nextChild:
				#print ("child", child)
				#print ("len(node.nextChild)", len(node.nextChild))
				queueList.append((child, len(node.nextChild)))

			nodeCount-=1
		print ("")

def findHyperInfoWithVolRed(root, volRed):
	if root is None:
		return
	relHyperInfos = []

	queueList = []
	queueList.append((root, 1))
	parentWithSmallestVolRed = None
	smallestVolRed = float("inf")

	while len(queueList) > 0:
		node, nodeCount = queueList[0]
		#print ("node")
		#print (node)
		#print ("nodeCount", nodeCount)
		
		while nodeCount > 0:
			#print ("start while")
			node, _ = queueList[0]
			del queueList[0]

			for child in node.nextChild:
				#print ("child", child)
				#print ("len(node.nextChild)", len(node.nextChild))
				parentHyper = node.hyper
				childHyper = child.hyper

				parentVol = volume(parentHyper)
				childVol = volume(childHyper)
				propVolRed = (parentVol - childVol)/parentVol

				if parentVol > 0 and propVolRed >= volRed and node.operation == "l":
					relHyperInfos.append((node, child))
					if propVolRed < smallestVolRed:
						parentWithSmallestVolRed = parentVol
						smallestVolRed = propVolRed

				queueList.append((child, len(node.nextChild)))

			nodeCount-=1
	return relHyperInfos, parentWithSmallestVolRed, smallestVolRed



if __name__ == "__main__":
	filename = "rambus_mosfet_numStages_4_gcc_4.0_useLp_True.pkl"
	theFile = open(filename, "rb")
	hyperInfo = pickle.load(theFile)
	theFile.close()
	volRed = 0.9
	relHyperInfos, parentWithSmallestVolRed, smallestVolRed = findHyperInfoWithVolRed(hyperInfo, volRed = volRed)
	for item in relHyperInfos:
		print ("parent child with vol reduction grater than", volRed)
		print ("parent", item[0], "volume", volume(item[0].hyper))
		print ("child", item[1], "volume", volume(item[1].hyper))
	print  ("parentWithSmallestVolRed", parentWithSmallestVolRed, "smallestVolRed", smallestVolRed)
	print ("numValid entries", len(relHyperInfos))


