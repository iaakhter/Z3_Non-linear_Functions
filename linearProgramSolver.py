import numpy as np
import copy

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

# Applies the simplex on a standard form LP matrix
# the first row is the objective function
# the first element in the first row must be positive
def simplex(standardMat, artificialIndices):
	numRows = standardMat.shape[0]
	numCols = standardMat.shape[1]
	count = 0
	while len(np.where(standardMat[0] < 0)[0]) > 0:
		print "mat "
		print standardMat
		print ""
		enteringVariable = np.where(standardMat[0,:] < 0)[0][0]
		ratios = np.divide(standardMat[:,numCols - 1],standardMat[:,enteringVariable])
		ratioSortedIndices = np.argsort(ratios)
		for ind in ratioSortedIndices:
			if ind != 0 and ratios[ind] > 0:
				pivot = ind
				break
		print "entering variable ", enteringVariable
		print "ratios ", ratios
		print "ratioIndices", ratioSortedIndices
		print "pivot ", pivot
		standardMat = gauss_jordan(standardMat,pivot,enteringVariable)
		print "mat after "
		print standardMat
		print ""
		count+=1
		'''if count == 2:
			return'''
	sols = np.zeros((numCols - 1))
	for i in range(len(sols)):
		if np.sum(standardMat[:,i]==0) == numRows - 1:
			nonzeroIndex = np.where(standardMat[:,i] != 0)[0][0]
			if standardMat[nonzeroIndex][i] == 1:
				sols[i] = standardMat[nonzeroIndex][numCols-1]
	return sols

def dualSimplex(theMat):
	normalizedMat = copy.deepcopy(theMat)
	numRows = normalizedMat.shape[0]
	numCols = normalizedMat.shape[1]
	numNegsIn1stRow = np.sum(normalizedMat[0,:numCols-1] < 0)
	numPosInLastColumn = np.sum(normalizedMat[1:,numCols-1] > 0)

	if numNegsIn1stRow != 0:
		print "can't apply dual simplex to this matrix"
		return None, None
	count = 0
	while np.sum(normalizedMat[1:,numCols-1] < 0) > 0:
		#print "mat "
		#print normalizedMat
		#print ""
		enteringVariable, pivot = None, None
		minInCol = np.argsort(normalizedMat[:,numCols-1])
		for i in range(len(minInCol)):
			minInd = minInCol[i]
			foundEnteringVariable = False
			if minInd != 0 and normalizedMat[minInd,numCols-1] < 0:
				pivot = minInd
				#print "pivot ", pivot
				ratios = np.divide(-normalizedMat[0,:numCols - 1],normalizedMat[pivot,:numCols-1])
				ratioSortedIndices = np.argsort(ratios)
				'''print "ratios ", ratios
				print "ratioIndices", ratioSortedIndices'''
				for ind in ratioSortedIndices:
					if ind != 0 and normalizedMat[pivot,ind] < 0:
						enteringVariable = ind
						foundEnteringVariable = True
						break
				if foundEnteringVariable:
					break
		if foundEnteringVariable == False:
			print "No feasible solution"
			return None, None

		#print "pivot ", pivot
		#print "enteringVariable ", enteringVariable
		#print "entering variable ", enteringVariable
		normalizedMat = gauss_jordan(normalizedMat,pivot,enteringVariable)
		#print "mat after "
		#print normalizedMat
		#print ""
		count+=1
		'''if count == 2:
			return'''
	sols = np.zeros((numCols - 1))
	for i in range(len(sols)):
		if np.sum(normalizedMat[:,i]==0) == numRows - 1:
			nonzeroIndex = np.where(normalizedMat[:,i] != 0)[0][0]
			if normalizedMat[nonzeroIndex][i] == 1:
				sols[i] = normalizedMat[nonzeroIndex][numCols-1]
	return normalizedMat, sols


def gauss_jordan(mat, pivot, enteringVariable):
	mat[pivot] = mat[pivot]/mat[pivot][enteringVariable]
	for i in range(mat.shape[0]):
		if i!= pivot:
			mat[i] = mat[i] - mat[i][enteringVariable]*mat[pivot]
	return mat

# Assumptions: All decision variables are >= 0
def normalize(stringConstraint):
	#split the strings appropriately to make it easy for parsing
	splittedConstraint = stringConstraint.split("\n")
	finalSplitConst = []
	for s in splittedConstraint:
		splitted = s.split(" ")
		finalSplitConst.append(splitted)

	# get all the variables from the last constraint added
	# assumes that the last constraint always constraints the individual
	# decision variables. e.g. x1 >= 0, x2 >= 0 and so on
	variables = {}
	variableConstraints = finalSplitConst[len(finalSplitConst)-1]
	countVariables = 1
	
	for j in range(len(variableConstraints)-1):
		if variableConstraints[j+1] == ">=":
			variables[variableConstraints[j]] = countVariables
			countVariables += 1

	print "variables"
	print variables
	countExtraVariables = 1 # because of the objective function
	countExtraConstraints = 0
	for i in range(1,len(finalSplitConst)-1):
		const = finalSplitConst[i]
		if const[len(const)-2] == ">=" or const[len(const)-2] == "<=":
			countExtraVariables+=1
		elif const[len(const)-2] == "==":
			countExtraVariables+=2
			countExtraConstraints +=1

	mat = np.zeros((len(finalSplitConst)-1 + countExtraConstraints,countVariables + countExtraVariables))
	#print "mat.shape ", mat.shape
	mat[0,0] = 1.0
	slackVarNum = countVariables
	#print "finalSplitConst"
	#print finalSplitConst

	# fill in the matrix
	matIndex = 0
	for i in range(len(finalSplitConst)-1):
		const = finalSplitConst[i]
		# deal with min max issue in the objectibe function
		if i== 0:
			maximize = True
			if const[0] == "min":
				maximize = False
			for j in range(1,len(const)-1):
				if is_number(const[j]):
					if maximize:
						if type(variables[const[j+1]]) != list:
							mat[matIndex][variables[const[j+1]]] = -float(const[j])
					else:
						if type(variables[const[j+1]]) != list:
							mat[matIndex][variables[const[j+1]]] = float(const[j])
		else:
			lessThan = True
			greaterThan = False
			if const[len(const)-2] == ">=":
				lessThan = False
				greaterThan = True
			elif const[len(const)-2] == "==":
				lessThan = False
				greaterThan = False
			#print "const, ", const
			#print "len(const) ", len(const)
			for j in range(len(const)):
				# if equality constraint, convert it to two less than equality
				# constraint
				if is_number(const[j]):
					if lessThan == False and greaterThan == False and j < len(const)-1:
						if type(variables[const[j+1]]) != list:
							mat[matIndex][variables[const[j+1]]] = float(const[j])
							mat[matIndex+1][variables[const[j+1]]] = -float(const[j])
					
					elif lessThan == False and greaterThan == False and j == len(const)-1:
						mat[matIndex][mat.shape[1]-1] = float(const[j])
						mat[matIndex+1][mat.shape[1]-1] = -float(const[j])
					
					elif lessThan == False and j < len(const)-1:
						if type(variables[const[j+1]]) != list:
							mat[matIndex][variables[const[j+1]]] = -float(const[j])
					
					elif lessThan == False and j == len(const)-1:
						mat[matIndex][mat.shape[1]-1] = -float(const[j])
					
					elif lessThan == True and j < len(const)-1:
						if type(variables[const[j+1]]) != list:
							mat[matIndex][variables[const[j+1]]] = float(const[j])
					
					elif lessThan == True and j == len(const)-1:
						mat[matIndex][mat.shape[1]-1] = float(const[j])

			if lessThan == False and greaterThan == False:
				mat[matIndex][slackVarNum] = 1.0
				slackVarNum += 1
				mat[matIndex+1][slackVarNum] = 1.0
				slackVarNum += 1
				matIndex += 1
			else:
				mat[i][slackVarNum] = 1.0
				slackVarNum += 1
		matIndex += 1
	return mat


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
	return a/(np.cosh(a*val)*np.cosh(a*val))
	#return (-4.0*a)/((exp(a*val) + exp(-a*val)) * (exp(a*val) + exp(-a*val)))

def convertTriangleBoundsToConstraints(a, Vin, Vout, Vlow, Vhigh):
	#return a string
	constraint = ""
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

	if a > 0:
		if Vlow >= 0 and Vhigh >=0:
			constraint = "min 1 "+Vout+"\n"
			constraint += "1 " + Vout+" + "+str(-dThird) + " " + Vin +" >= "+str(cThird)+"\n"
			constraint += "1 " + Vout+" + "+str(-dLow) + " " + Vin +" <= "+str(cLow)+"\n"
			constraint += "1 " + Vout+" + "+str(-dHigh) + " " + Vin +" <= "+str(cHigh)+"\n"
			if Vlow != 0:
				constraint += "1 " + Vin + " >= "+str(Vlow)+"\n"
			constraint += "1 " + Vin + " <= "+str(Vhigh)+"\n"

		elif Vlow <=0 and Vhigh <=0:
			constraint = "max -1 "+Vout+"\n"
			constraint += "-1 " + Vout + " + " +str(dThird)+" "+Vin+" <= "+str(cThird)+"\n"
			constraint += "-1 " + Vout + " + " +str(dLow)+" "+Vin+" >= "+str(cLow)+"\n"
			constraint += "-1 " + Vout + " + " +str(dHigh)+" "+Vin+" >= "+str(cHigh)+"\n"
			constraint += "-1 " + Vin + " >= "+str(Vlow)+"\n"
			if Vhigh != 0:
				constraint += "-1 " + Vin + " <= "+str(Vhigh)+"\n"
	
	elif a < 0:
		if Vlow <= 0 and Vhigh <=0:
			constraint = "min 1 "+Vout+"\n"
			constraint += "1 " + Vout+" + "+str(dThird) + " " + Vin +" >= "+str(cThird)+"\n"
			constraint += "1 " + Vout+" + "+str(dLow) + " " + Vin +" <= "+str(cLow)+"\n"
			constraint += "1 " + Vout+" + "+str(dHigh) + " " + Vin +" <= "+str(cHigh)+"\n"
			constraint += "-1 " + Vin + " >= "+str(Vlow)+"\n"
			if Vhigh != 0:
				constraint += "-1 " + Vin + " <= "+str(Vhigh)+"\n"
		
		elif Vlow >=0 and Vhigh >=0:
			constraint = "max -1 "+Vout+"\n"
			constraint += "-1 " + Vout + " + "+str(-dThird)+" "+Vin+" <= "+str(cThird)+"\n"
			constraint += "-1 " + Vout + " + "+str(-dLow)+" "+Vin+" >= "+str(cLow)+"\n"
			constraint += "-1 " + Vout + " + "+str(-dHigh)+" "+Vin+" >= "+str(cHigh)+"\n"
			constraint += "1 " + Vin + " >= "+str(Vlow)+"\n"
			if Vhigh != 0:
				constraint += "1 " + Vin + " <= "+str(Vhigh)+"\n"
	return constraint

def convertTrapezoidBoundsToConstraints(a, Vin, Vout, Vlow, Vhigh):
	constraint = ""
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

	if a > 0:
		if Vlow <= 0 and Vhigh <= 0:
			constraint = "max -1 "+Vout+"\n"
			constraint += "-1 "+Vout+" + "+str(dLow)+" "+Vin+" >= "+str(cLow)+"\n"
			constraint += "-1 "+Vout+" >= "+"-1\n"
			constraint += "-1 "+Vout+" <= "+str(tanhFunVlow)+"\n"
			constraint += "-1 "+Vin+" <= "+str(Vlow)+"\n"
		elif Vlow >= 0 and Vhigh >=0:
			constraint = "min 1 "+Vout+"\n"
			constraint += "1 "+Vout+" + "+str(-dHigh)+" "+Vin+" <= "+str(cHigh)+"\n"
			constraint += "1 "+Vout+" <= "+"1\n"
			constraint += "1 "+Vout+" >= "+str(tanhFunVhigh)+"\n"
			constraint += "1 "+Vin+" >= "+str(Vhigh)+"\n"

	elif a < 0:
		if Vlow <= 0 and Vhigh <= 0:
			constraint = "min 1 "+Vout+"\n"
			constraint += "1 "+Vout+" + "+str(dLow)+" "+Vin+" <= "+str(cLow)+"\n"
			constraint += "1 "+Vout+" <= "+"1\n"
			constraint += "1 "+Vout+" >= "+str(tanhFunVlow)+"\n"
			constraint += "-1 "+Vin+" <= "+str(Vlow)+"\n"
		elif Vlow >= 0 and Vhigh >=0:
			constraint = "max -1 "+Vout+"\n"
			constraint += "-1 "+Vout+" + "+str(-dHigh)+" "+Vin+" >= "+str(cHigh)+"\n"
			constraint += "-1 "+Vout+" >= "+"-1\n"
			constraint += "-1 "+Vout+" <= "+str(tanhFunVhigh)+"\n"
			constraint += "1 "+Vin+" >= "+str(Vhigh)+"\n"

	return constraint
	

def fun1Constraints(Vin, Vout):
	a = 1
	params = [0.3,0.1]
	solutions = []
	mats = []

	Vlow = 0.0
	Vhigh = 0.5
	overallConstraint = ""
	if Vlow >= 0 and Vhigh >= 0:
		overallConstraint += "1 "+Vout+" + "+str(-params[0])+" "+Vin+" == "+str(params[1])+"\n"
	elif Vlow <= 0 and Vhigh <= 0:
		overallConstraint += "-1 "+Vout+" + "+str(params[0])+" "+Vin+" == "+str(params[1])+"\n"
	overallConstraint += Vout + " >= 0 " + Vin + " >= 0"
	testSol = -3.66
	testSol = None
	
	triConstraint = convertTriangleBoundsToConstraints(a, Vin, Vout, Vlow, Vhigh)
	triConstraint += overallConstraint
	print "triConstraint1"
	print triConstraint
	mat = normalize(triConstraint)
	print "mat from constraints"
	print mat
	mat,soln = dualSimplex(mat)
	print "solutions ", soln
	print ""
	if soln is not None:
		if Vlow >= 0 and Vhigh >= 0:
			solutions.append(soln[2])
			if testSol is not None:
				sillySyntax(mat, testSol)
		elif Vlow <= 0 and Vhigh <= 0:
			solutions.append(-soln[2])
			if testSol is not None:
				sillySyntax(mat, testSol)
		mats.append(mat)
	
	trapConstraint = convertTrapezoidBoundsToConstraints(a, Vin, Vout, Vlow, Vhigh)
	trapConstraint += overallConstraint
	print "trapConstraint1"
	print trapConstraint
	mat = normalize(trapConstraint)
	mat,soln = dualSimplex(mat)
	print "solutions ", soln
	print ""
	if soln is not None:
		if Vlow >= 0 and Vhigh >= 0:
			solutions.append(soln[2])
			if testSol is not None:
				sillySyntax(mat, testSol)
		elif Vlow <= 0 and Vhigh <= 0:
			solutions.append(-soln[2])
			if testSol is not None:
				sillySyntax(mat, testSol)
		mats.append(mat)

	Vlow = -0.5
	Vhigh = 0.0
	overallConstraint = ""
	if Vlow >= 0 and Vhigh >= 0:
		overallConstraint += "1 "+Vout+" + "+str(-params[0])+" "+Vin+" == "+str(params[1])+"\n"
	elif Vlow <= 0 and Vhigh <= 0:
		overallConstraint += "-1 "+Vout+" + "+str(params[0])+" "+Vin+" == "+str(params[1])+"\n"
	overallConstraint += Vout + " >= 0 " + Vin + " >= 0"
	
	triConstraint = convertTriangleBoundsToConstraints(a, Vin, Vout, Vlow, Vhigh)
	triConstraint += overallConstraint
	print "triConstraint2"
	print triConstraint
	mat = normalize(triConstraint)
	mat,soln = dualSimplex(mat)
	print "solutions ", soln
	print ""
	if soln is not None:
		if Vlow >= 0 and Vhigh >= 0:
			solutions.append(soln[2])
			if testSol is not None:
				sillySyntax(mat, testSol)
		elif Vlow <= 0 and Vhigh <= 0:
			solutions.append(-soln[2])
			if testSol is not None:
				sillySyntax(mat, testSol)
		mats.append(mat)
	
	trapConstraint = convertTrapezoidBoundsToConstraints(a, Vin, Vout, Vlow, Vhigh)
	trapConstraint += overallConstraint
	print "trapConstraint2"
	print trapConstraint
	mat = normalize(trapConstraint)
	mat,soln = dualSimplex(mat)
	print "solutions ", soln
	print ""
	if soln is not None:
		if Vlow >= 0 and Vhigh >= 0:
			solutions.append(soln[2])
			if testSol is not None:
				sillySyntax(mat, testSol)
		elif Vlow <= 0 and Vhigh <= 0:
			solutions.append(-soln[2])
			if testSol is not None:
				sillySyntax(mat, testSol)
		mats.append(mat)
	
	return mats,solutions

# check if solution is feasible in mat
def sillySyntax(mat, sol):
	print "Checking feasibility of solution ", sol
	newMat = np.zeros((mat.shape[0]+2,mat.shape[1]+2))
	newMat[0:mat.shape[0],0:mat.shape[1]-1] = mat[:,0:mat.shape[1]-1]
	newMat[0:mat.shape[0],newMat.shape[1]-1] = mat[:,mat.shape[1]-1]
	if sol >= 0:
		newMat[mat.shape[0]][2] = 1.0
		newMat[mat.shape[0]][mat.shape[1]-1] = 1.0
		newMat[mat.shape[0]][newMat.shape[1]-1] = sol

		newMat[mat.shape[0]+1][2] = -1.0
		newMat[mat.shape[0]+1][mat.shape[1]] = 1.0
		newMat[mat.shape[0]+1][newMat.shape[1]-1] = -sol
	else:
		newMat[mat.shape[0]][2] = -1.0
		newMat[mat.shape[0]][mat.shape[1]-1] = 1.0
		newMat[mat.shape[0]][newMat.shape[1]-1] = sol

		newMat[mat.shape[0]+1][2] = 1.0
		newMat[mat.shape[0]+1][mat.shape[1]] = 1.0
		newMat[mat.shape[0]+1][newMat.shape[1]-1] = -sol

	'''print "mat "
	print mat
	print "newMat before"
	print newMat'''

	nonzeroIndex = np.where(mat[:,2]!=0)[0][0]
	newMat[mat.shape[0]] = newMat[mat.shape[0]] - (newMat[mat.shape[0]][2]/mat[nonzeroIndex][2])*newMat[nonzeroIndex]
	newMat[mat.shape[0]+1] = newMat[mat.shape[0]+1] - (newMat[mat.shape[0]+1][2]/mat[nonzeroIndex][2])*newMat[nonzeroIndex]

	#print "newMat after"
	#print newMat

	mat,soln = dualSimplex(newMat)
	if soln is None:
		print sol, " solution is not feasible\n"
	else:
		print sol, " solution is feasible\n"
	return

def findHyper(distances):
	Vin = "x0"
	Vout = "y0"
	mats,solutions = fun1Constraints(Vin, Vout)
	newConstraint = ""
	finalSolutions = {}
	for sol in solutions:
		finalSolutions[sol] = True
	count = 0
	hypers = []
	while len(solutions) > 0:
		newSolutions = []
		newMats = []
		for i in range(len(solutions)):
			soln = solutions[i]
			print "solution Number ", i, ": ", soln
			lowVal = soln - distances
			highVal = soln + distances
			start = 0
			hypers.append([lowVal,highVal])
			for j in range(0,len(mats)):
				print "mat number ", j
				mat = mats[j]
				newMat = np.zeros((mat.shape[0]+1,mat.shape[1]+1))
				newMat[0:mat.shape[0],0:mat.shape[1]-1] = mat[:,0:mat.shape[1]-1]
				newMat[0:mat.shape[0],newMat.shape[1]-1] = mat[:,mat.shape[1]-1]
				if lowVal >= 0:
					newMat[mat.shape[0]][2] = 1.0
				elif lowVal <= 0:
					newMat[mat.shape[0]][2] = -1.0
				newMat[mat.shape[0]][mat.shape[1]-1] = 1.0
				newMat[mat.shape[0]][newMat.shape[1]-1] = lowVal
				#print "mat"
				#print mat
				print "lowVal ", lowVal
				#print "newMat"
				#print newMat

				nonzeroIndex = np.where(mat[:,2]!=0)[0][0]
				newMat[mat.shape[0]] = newMat[mat.shape[0]] - (newMat[mat.shape[0]][2]/mat[nonzeroIndex][2])*newMat[nonzeroIndex]
				finalMat,sols = dualSimplex(newMat)
				if sols is not None:
					if -sols[2] not in finalSolutions and -sols[2] <= lowVal:
						finalSolutions[-sols[2]] = True
						newSolutions.append(-sols[2])
						print "found new solution ", -sols[2]

					if sols[2] not in finalSolutions and sols[2] <= lowVal:
						finalSolutions[sols[2]] = True
						newSolutions.append(sols[2])
						print "found new solution ", sols[2]
					newMats.append(newMat)
				else:
					print "no new solution found"

				newMat = np.zeros((mat.shape[0]+1,mat.shape[1]+1))
				newMat[0:mat.shape[0],0:mat.shape[1]-1] = mat[:,0:mat.shape[1]-1]
				newMat[0:mat.shape[0],newMat.shape[1]-1] = mat[:,mat.shape[1]-1]
				# need to figure out this better constraint
				if highVal >= 0:
					newMat[mat.shape[0]][2] = -1.0
				elif highVal <= 0:
					newMat[mat.shape[0]][2] = 1.0
				newMat[mat.shape[0]][mat.shape[1]-1] = 1.0
				newMat[mat.shape[0]][newMat.shape[1]-1] = -highVal

				#print "mat"
				#print mat
				print "highVal ", highVal
				#print "newMat"
				#print newMat
			
				newMat[mat.shape[0]] = newMat[mat.shape[0]] - (newMat[mat.shape[0]][2]/mat[nonzeroIndex][2])*newMat[nonzeroIndex]
				finalMat,sols = dualSimplex(newMat)
				if sols is not None:
					if -sols[2] not in finalSolutions and -sols[2] >= highVal:
						finalSolutions[-sols[2]] = True
						newSolutions.append(-sols[2])
						print "found new solution ", -sols[2]
					if sols[2] not in finalSolutions and sols[2] >= highVal:
						finalSolutions[sols[2]] = True
						newSolutions.append(sols[2])
						print "found new solution ", sols[2]
					newMats.append(newMat)
				else:
					print "no new solution found"
				print ""
			#if i==0:
			#	break

		solutions = newSolutions
		mats = newMats
		count +=1
	print "found all solutions"
	print finalSolutions

	print "hyperrectangles"
	print hypers
	print ""

	return hypers


if __name__=="__main__":
	'''solutions = fun1Constraints()
	print "final solutions"
	print solutions'''
	allHypers = findHyper(0.5)

# normalize LP - for min, should be greater than equal to. for max should be less than equal to
#first do simple simplex and then do dual simplex

