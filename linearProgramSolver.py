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

def dualSimplex(normalizedMat):
	numRows = normalizedMat.shape[0]
	numCols = normalizedMat.shape[1]
	numNegsIn1stRow = np.sum(normalizedMat[0,:numCols-1] < 0)
	numPosInLastColumn = np.sum(normalizedMat[1:,numCols-1] > 0)

	if numNegsIn1stRow != 0:
		print "can't apply dual simplex to this matrix"
		return None
	count = 0
	while np.sum(normalizedMat[1:,numCols-1] < 0) > 0:
		print "mat "
		print normalizedMat
		print ""
		enteringVariable, pivot = None, None
		minInCol = np.argsort(normalizedMat[:,numCols-1])
		for i in range(len(minInCol)):
			minInd = minInCol[i]
			foundEnteringVariable = False
			if minInd != 0 and normalizedMat[minInd,numCols-1] < 0:
				pivot = minInd
				print "pivot ", pivot
				ratios = np.divide(-normalizedMat[0,:numCols - 1],normalizedMat[pivot,:numCols-1])
				ratioSortedIndices = np.argsort(ratios)
				print "ratios ", ratios
				print "ratioIndices", ratioSortedIndices
				for ind in ratioSortedIndices:
					if ind != 0 and normalizedMat[pivot,ind] < 0:
						enteringVariable = ind
						foundEnteringVariable = True
						break
				if foundEnteringVariable:
					break
		if foundEnteringVariable == False:
			print "No feasible solution"
			return None

		print "entering variable ", enteringVariable
		normalizedMat = gauss_jordan(normalizedMat,pivot,enteringVariable)
		print "mat after "
		print normalizedMat
		print ""
		count+=1
		'''if count == 2:
			return'''
	sols = np.zeros((numCols - 1))
	for i in range(len(sols)):
		if np.sum(normalizedMat[:,i]==0) == numRows - 1:
			nonzeroIndex = np.where(normalizedMat[:,i] != 0)[0][0]
			if normalizedMat[nonzeroIndex][i] == 1:
				sols[i] = normalizedMat[nonzeroIndex][numCols-1]
	return sols


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
	print "finalSplitConst"
	print finalSplitConst

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
						mat[matIndex][variables[const[j+1]]] = -float(const[j])
					else:
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
						mat[matIndex][variables[const[j+1]]] = float(const[j])
						mat[matIndex+1][variables[const[j+1]]] = -float(const[j])
						#print "mat ", mat[matIndex+1][variables[const[j+1]]]
					elif lessThan == False and greaterThan == False and j == len(const)-1:
						mat[matIndex][mat.shape[1]-1] = float(const[j])
						mat[matIndex+1][mat.shape[1]-1] = -float(const[j])
					elif lessThan == False and j < len(const)-1:
						mat[matIndex][variables[const[j+1]]] = -float(const[j])
					elif lessThan == False and j == len(const)-1:
						mat[matIndex][mat.shape[1]-1] = -float(const[j])
					elif lessThan == True and j < len(const)-1:
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


def convertToStdLP(stringConstraint):
	artificialIndices = []
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
	variablesSign = {} # if negative need to change variables in original LP
	variableConstraints = finalSplitConst[len(finalSplitConst)-1]
	countVariables = 1
	for j in range(len(variableConstraints)-1):
		if variableConstraints[j+1] == "urs":
			variablesSign[variableConstraints[j]] = "urs"
			variables[variableConstraints[j]] = [countVariables, countVariables+1]
			countVariables += 2
		elif variableConstraints[j+1] == ">=":
			variablesSign[variableConstraints[j]] = True #variable must be positive
			variables[variableConstraints[j]] = countVariables
			countVariables += 1
		elif variableConstraints[j+1] == "<=":
			variablesSign[variableConstraints[j]] = False #variable must be negative
			countVariables += 1

	print "variablesSign"
	print variablesSign
	print "variables"
	print variables
	countInequalities = 1 # because of the objective function
	for i in range(1,len(finalSplitConst)-1):
		const = finalSplitConst[i]
		if const[len(const)-2] == ">=":
			countInequalities+=2
		else:
			countInequalities+=1

	mat = np.zeros((len(finalSplitConst)-1,countVariables + countInequalities))
	#print "mat.shape ", mat.shape
	mat[0,0] = 1.0
	slackVarNum = countVariables
	print "finalSplitConst"
	print finalSplitConst
	
	# fill in the matrix
	for i in range(mat.shape[0]):
		const = finalSplitConst[i]
		# deal with min max issue in the objectibe function
		if i== 0:
			maximize = True
			if const[0] == "min":
				maximize = False
			for j in range(1,len(const)-1):
				if is_number(const[j]):
					if maximize:
						#if variable is unrestricted, replace it with two variables
						if variablesSign[const[j+1]] == "urs":
							mat[i][variables[const[j+1]][0]] = -float(const[j])
							mat[i][variables[const[j+1]][1]] = float(const[j])
						elif variablesSign[const[j+1]]:
							mat[i][variables[const[j+1]]] = -float(const[j])
						else:
							mat[i][variables[const[j+1]]] = float(const[j])
					else:
						if variablesSign[const[j+1]] == "urs":
							mat[i][variables[const[j+1]][0]] = float(const[j])
							mat[i][variables[const[j+1]][1]] = -float(const[j])
						if variablesSign[const[j+1]]:
							mat[i][variables[const[j+1]]] = float(const[j])
						else:
							mat[i][variables[const[j+1]]] = -float(const[j])
		else:
			negConstant = False # if negative constant on the right hand side
								# multiply everything by -1 and flip the equality signs
			if (float(const[len(const)-1]) < 0):
				negConstant = True
			lessThan = True
			greaterThan = False
			if const[len(const)-2] == ">=":
				lessThan = False
				greaterThan = True
			elif const[len(const)-2] == "==":
				lessThan = False
				greaterThan = False
			if (lessThan or greaterThan) and negConstant:
				mat[i][mat.shape[1]-1] = -float(const[len(const)-1])
				if lessThan:
					lessThan = False
					greaterThan = True
				else:
					lessThan = True
					greaterThan = False
			else:
				mat[i][mat.shape[1]-1] = float(const[len(const)-1])
			for j in range(len(const)-1):
				if is_number(const[j]):
					if variablesSign[const[j+1]] == "urs":
						if negConstant == False:
							mat[i][variables[const[j+1]][0]] = float(const[j])
							mat[i][variables[const[j+1]][1]] = -float(const[j])
						else:
							mat[i][variables[const[j+1]][0]] = -float(const[j])
							mat[i][variables[const[j+1]][1]] = float(const[j])
					elif variablesSign[const[j+1]]:
						if negConstant == False:
							mat[i][variables[const[j+1]]] = float(const[j])
						else:
							mat[i][variables[const[j+1]]] = -float(const[j])
					else:
						if negConstant == False:
							mat[i][variables[const[j+1]]] = -float(const[j])
						else:
							mat[i][variables[const[j+1]]] = float(const[j])

			if lessThan == False and greaterThan == False:
				mat[i][slackVarNum] = 1.0
				mat[0][slackVarNum] = 1e+15
				artificialIndices.append([i,slackVarNum])
				slackVarNum += 1
			elif lessThan:
				mat[i][slackVarNum] = 1.0
				slackVarNum += 1
			elif greaterThan:
				mat[i][slackVarNum] = -1.0
				slackVarNum += 1
				mat[i][slackVarNum] = 1.0
				mat[0][slackVarNum] = 1e+15
				artificialIndices.append([i,slackVarNum])
				slackVarNum += 1


	return (mat, artificialIndices)

def linearConstraintFun1():
	constraintString = "1 y + 0.133 x <= 0.9205\n\
						1 y + 5.0 x <= 0.0\n\
						1 y + 1.974 x >= 0.0\n\
						1 y <= 1.0\n\
						1 y >= 0.987\n\
						1 y - 0.3 x == 0.1"

def tanhFun(a,val):
	return tanh(a*val)
	#return -(exp(a*val) - exp(-a*val))/(exp(a*val) + exp(-a*val))

'''
takes in non-symbolic python values
calculates the derivative of tanhFun of val
'''
def tanhFunder(a,val):
	den = cosh(a*val)*cosh(a*val)
	#print "den ", den
	return a/(cosh(a*val)*cosh(a*val))
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
			constraint += "1 " + Vout+" >= "+str(dThird) + " " + Vin +" + "+str(cThird)+"\n"
			constraint += "1 " + Vout+" <= "+str(dLow) + " " + Vin +" + "+str(cLow)+"\n"
			constraint += "1 " + Vout+" <= "+str(dHigh) + " " + Vin +" + "+str(cHigh)+"\n"

		elif Vlow <=0 and Vhigh <=0:
			constraint = "max -1 "+Vout+"\n"
			constraint += "-1 " + Vout + " <= "+" -"+str(dThird)+" "+Vin+" + "+str(cThird)+"\n"
			constraint += "-1 " + Vout + " >= "+" -"+str(dLow)+" "+Vin+" + "+str(cLow)+"\n"
			constraint += "-1 " + Vout + " >= "+" -"+str(dHigh)+" "Vin+" + "+str(cHihg)+"\n"

	elif a < 0:
		if Vlow <= 0 and Vhigh <=0:
			constraint = "min 1 "+Vout+"\n"
			constraint += "1 " + Vout+" >= "+str(dThird) + " " + Vin +" + "+str(cThird)+"\n"
			constraint += "1 " + Vout+" <= "+str(dLow) + " " + Vin +" + "+str(cLow)+"\n"
			constraint += "1 " + Vout+" <= "+str(dHigh) + " " + Vin +" + "+str(cHigh)+"\n"

		elif Vlow >=0 and Vhigh >=0:
			constraint = "max -1 "+Vout+"\n"
			constraint += "-1 " + Vout + " <= "+" -"+str(dThird)+" "+Vin+" + "+str(cThird)+"\n"
			constraint += "-1 " + Vout + " >= "+" -"+str(dLow)+" "+Vin+" + "+str(cLow)+"\n"
			constraint += "-1 " + Vout + " >= "+" -"+str(dHigh)+" "Vin+" + "+str(cHihg)+"\n"
	return constraint

def convertTrapezoidBoundsToConstraints():
	#hello

def fun1Constraints():
	#hello



if __name__=="__main__":
	stringConstraint1 = "min 1 y0\n1 y0 + -0.7864 x0 <= 0.0689\n1 y0 + -1 x0 <= 0\n1 y0 + -0.9242 x0 >= 0\n1 y0 + -0.3 x0 == 0.1\n1 x0 <= 0.5\ny0 >= 0 x0 >= 0"
	stringConstraint2 = "min 1 y0\n1 y0 + -0.7864 x0 <= 0.0689\n1 y0 <= 1\n1 y0 + -0.3 x0 == 0.1\n1 x0 >= 0.5\n1 y0 >= 0.4621\ny0 >= 0 x0 >= 0"
	stringConstraint3 = "max -1 y0\n-1 y0 + 0.7864 x0 >= -0.0689\n-1 y0 + 1 x0 >= 0\n-1 y0 + 0.9242 x0 <= 0\n-1 y0 + 0.3 x0 == 0.1\n-1 x0 >= -0.5\ny0 >= 0 x0 >= 0"
	stringConstraint4 = "max -1 y0\n-1 y0 + 0.7864 x0 >= -0.0689\n-1 y0 >= -1\n-1 y0 + 0.3 x0 == 0.1\n-1 x0 <= -0.5\n-1 y0 <= -0.4621\ny0 >= 0 x0 >= 0"
	#stringConstraint = "max -5 x1 + -35 x2 + -20 x3\n1 x1 + -1 x2 + -1 x3 <= -2\n-1 x1 + -3 x2 <= -3\nx1 >= 0 x2 >= 0 x3 >= 0"
	print "stringConstraint1"
	print stringConstraint1
	mat = normalize(stringConstraint1)
	solutions = dualSimplex(mat)
	print "final solutions1"
	print solutions

	'''print "stringConstraint2"
	print stringConstraint2
	mat = normalize(stringConstraint2)
	solutions = dualSimplex(mat)
	print "final solutions2"
	print solutions

	print "stringConstraint3"
	print stringConstraint3
	mat = normalize(stringConstraint3)
	solutions = dualSimplex(mat)
	print "final solutions3"
	print solutions

	print "stringConstraint4"
	print stringConstraint4
	mat = normalize(stringConstraint4)
	solutions = dualSimplex(mat)
	print "final solutions4"
	print solutions'''

# normalize LP - for min, should be greater than equal to. for max should be less than equal to
#first do simple simplex and then do dual simplex

