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
def simplex(theMat):
	standardMat = copy.deepcopy(theMat)
	numRows = standardMat.shape[0]
	numCols = standardMat.shape[1]
	count = 0
	while len(np.where(standardMat[0,:numCols-1] < 0)[0]) > 0:
		#print "mat "
		#print standardMat
		#print ""
		enteringVariable = np.where(standardMat[0,:] < 0)[0][0]
		ratios = np.divide(standardMat[:,numCols - 1],standardMat[:,enteringVariable])
		ratioSortedIndices = np.argsort(ratios)
		for ind in ratioSortedIndices:
			if ind != 0 and standardMat[ind,enteringVariable] > 0:
				pivot = ind
				break
		#print "entering variable ", enteringVariable
		#print "ratios ", ratios
		#print "ratioIndices", ratioSortedIndices
		#print "pivot ", pivot
		standardMat = gauss_jordan(standardMat,pivot,enteringVariable)
		#print "mat after "
		#print standardMat
		#print ""
		count+=1
		'''if count == 2:
			return'''
	sols = np.zeros((numCols - 1))
	for i in range(len(sols)):
		if np.sum(standardMat[:,i]==0) == numRows - 1:
			nonzeroIndex = np.where(standardMat[:,i] != 0)[0][0]
			if standardMat[nonzeroIndex][i] == 1:
				sols[i] = standardMat[nonzeroIndex][numCols-1]
	return standardMat, sols

def dualSimplex(theMat):
	normalizedMat = copy.deepcopy(theMat)
	numRows = normalizedMat.shape[0]
	numCols = normalizedMat.shape[1]
	numNegsIn1stRow = np.sum(normalizedMat[0,:numCols-1] < 0)
	numPosInLastColumn = np.sum(normalizedMat[1:,numCols-1] > 0)

	if numNegsIn1stRow != 0:
		# this is not a correct action maybe
		normalizedMat,_ = simplex(normalizedMat)
		#print "mat after simplex "
		#print normalizedMat
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

	#print "variables"
	#print variables
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
				#print "const ", const
				if is_number(const[j]):
					if lessThan == False and greaterThan == False and j < len(const)-1:
						#print "const[j+1], ", const[j+1]
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

def fun1Num(x,a,params):
	return tanhFun(a,x) - params[0]*x - params[1]

def fun1Der(x,a,params):
	return np.array([tanhFunder(a,x) - params[0]])

'''
params = [b1,b2]
'''
def fun1DerInterval(a,params,bounds):
	lowerBound = bounds[:,0]
	upperBound = bounds[:,1]
	der1 = fun1Der(lowerBound[0],a,params)
	der2 = fun1Der(upperBound[0],a,params)
	der = np.zeros((1,1,2))
	der[:,:,0] = np.minimum(der1,der2)
	der[:,:,1] = np.maximum(der1,der2)
	return der

def fun2Num(x,a,params):
	Iy = tanhFun(a,x[0]) + params[0] - x[1]
	Ix = tanhFun(a,x[1]) - params[0] - x[0]
	return np.array([Ix,Iy])

def fun2Der(x,a,params):
	der = -1*np.ones((len(x),len(x)))
	der[0][1] = tanhFunder(a,x[1])
	der[1][0] = tanhFunder(a,x[0])
	return der

def fun2DerInterval(a,params,bounds):
	lowerBound = bounds[:,0]
	upperBound = bounds[:,1]
	der1 = fun2Der(lowerBound,a,params)
	der2 = fun2Der(upperBound,a,params)
	der = np.zeros((len(lowerBound),len(lowerBound),2))
	der[:,:,0] = np.minimum(der1,der2)
	der[:,:,1] = np.maximum(der1,der2)
	return der

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
			#constraint = "min 1 "+Vout+"\n"
			constraint = "1 " + Vout+" + "+str(-dThird) + " " + Vin +" >= "+str(cThird)+"\n"
			constraint += "1 " + Vout+" + "+str(-dLow) + " " + Vin +" <= "+str(cLow)+"\n"
			constraint += "1 " + Vout+" + "+str(-dHigh) + " " + Vin +" <= "+str(cHigh)+"\n"
			if Vlow != 0:
				constraint += "1 " + Vin + " >= "+str(Vlow)+"\n"
			constraint += "1 " + Vin + " <= "+str(Vhigh)+"\n"

		elif Vlow <=0 and Vhigh <=0:
			#constraint = "max -1 "+Vout+"\n"
			constraint = "-1 " + Vout + " + " +str(dThird)+" "+Vin+" <= "+str(cThird)+"\n"
			constraint += "-1 " + Vout + " + " +str(dLow)+" "+Vin+" >= "+str(cLow)+"\n"
			constraint += "-1 " + Vout + " + " +str(dHigh)+" "+Vin+" >= "+str(cHigh)+"\n"
			constraint += "-1 " + Vin + " >= "+str(Vlow)+"\n"
			if Vhigh != 0:
				constraint += "-1 " + Vin + " <= "+str(Vhigh)+"\n"
	
	elif a < 0:
		if Vlow <= 0 and Vhigh <=0:
			#constraint = "min 1 "+Vout+"\n"
			constraint = "1 " + Vout+" + "+str(dThird) + " " + Vin +" >= "+str(cThird)+"\n"
			constraint += "1 " + Vout+" + "+str(dLow) + " " + Vin +" <= "+str(cLow)+"\n"
			constraint += "1 " + Vout+" + "+str(dHigh) + " " + Vin +" <= "+str(cHigh)+"\n"
			constraint += "-1 " + Vin + " >= "+str(Vlow)+"\n"
			if Vhigh != 0:
				constraint += "-1 " + Vin + " <= "+str(Vhigh)+"\n"
		
		elif Vlow >=0 and Vhigh >=0:
			#constraint = "max -1 "+Vout+"\n"
			constraint = "-1 " + Vout + " + "+str(-dThird)+" "+Vin+" <= "+str(cThird)+"\n"
			constraint += "-1 " + Vout + " + "+str(-dLow)+" "+Vin+" >= "+str(cLow)+"\n"
			constraint += "-1 " + Vout + " + "+str(-dHigh)+" "+Vin+" >= "+str(cHigh)+"\n"
			if Vlow != 0:
				constraint += "1 " + Vin + " >= "+str(Vlow)+"\n"
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
			#constraint = "max -1 "+Vout+"\n"
			constraint = "-1 "+Vout+" + "+str(dLow)+" "+Vin+" >= "+str(cLow)+"\n"
			constraint += "-1 "+Vout+" >= "+"-1\n"
			constraint += "-1 "+Vout+" <= "+str(tanhFunVlow)+"\n"
			constraint += "-1 "+Vin+" <= "+str(Vlow)+"\n"
		elif Vlow >= 0 and Vhigh >=0:
			#constraint = "min 1 "+Vout+"\n"
			constraint = "1 "+Vout+" + "+str(-dHigh)+" "+Vin+" <= "+str(cHigh)+"\n"
			constraint += "1 "+Vout+" <= "+"1\n"
			constraint += "1 "+Vout+" >= "+str(tanhFunVhigh)+"\n"
			constraint += "1 "+Vin+" >= "+str(Vhigh)+"\n"

	elif a < 0:
		if Vlow <= 0 and Vhigh <= 0:
			#constraint = "min 1 "+Vout+"\n"
			constraint = "1 "+Vout+" + "+str(dLow)+" "+Vin+" <= "+str(cLow)+"\n"
			constraint += "1 "+Vout+" <= "+"1\n"
			constraint += "1 "+Vout+" >= "+str(tanhFunVlow)+"\n"
			constraint += "-1 "+Vin+" <= "+str(Vlow)+"\n"
		elif Vlow >= 0 and Vhigh >=0:
			#constraint = "max -1 "+Vout+"\n"
			constraint = "-1 "+Vout+" + "+str(-dHigh)+" "+Vin+" >= "+str(cHigh)+"\n"
			constraint += "-1 "+Vout+" >= "+"-1\n"
			constraint += "-1 "+Vout+" <= "+str(tanhFunVhigh)+"\n"
			constraint += "1 "+Vin+" >= "+str(Vhigh)+"\n"

	return constraint

def fun1Constraints(bounds,a,params,triangle):
	Vin = "x0"
	Vout = "y0"
	solutions = []
	mats = []
	hypers = []

	for i in range(len(bounds)):
		bound = bounds[i]
		Vlow = bound[0][0]
		Vhigh = bound[1][0]

		objMinConstraint = ""
		objMaxConstraint = ""
		overallConstraint = ""
		if a >= 0:
			if Vlow >= 0 and Vhigh >= 0:
				overallConstraint += "1 "+Vout+" + "+str(-params[0])+" "+Vin+" == "+str(params[1])+"\n"
				objMinConstraint += "min 1 "+Vin+"\n"
				objMaxConstraint += "max 1 "+Vin+"\n"
			elif Vlow <= 0 and Vhigh <= 0:
				overallConstraint += "-1 "+Vout+" + "+str(params[0])+" "+Vin+" == "+str(params[1])+"\n"
				objMinConstraint += "min -1 "+Vin+"\n"
				objMaxConstraint += "max -1 "+Vin+"\n"
		else:
			if Vlow <= 0 and Vhigh <= 0:
				overallConstraint += "1 "+Vout+" + "+str(-params[0])+" "+Vin+" == "+str(params[1])+"\n"
				objMinConstraint += "min 1 "+Vin+"\n"
				objMaxConstraint += "max 1 "+Vin+"\n"
			elif Vlow >= 0 and Vhigh >= 0:
				overallConstraint += "-1 "+Vout+" + "+str(params[0])+" "+Vin+" == "+str(params[1])+"\n"
				objMinConstraint += "min -1 "+Vin+"\n"
				objMaxConstraint += "max -1 "+Vin+"\n"
		
		overallConstraint += Vout + " >= 0 " + Vin + " >= 0"
	
		triConstraintMin = objMinConstraint
		triConstraintMin += convertTriangleBoundsToConstraints(a, Vin, Vout, Vlow, Vhigh)
		triConstraintMin += overallConstraint
		print "triConstraintMin: ", "Vlow ", Vlow, " Vhigh ", Vhigh 
		print triConstraintMin
		minMat = normalize(triConstraintMin)
		minMat,minSoln = dualSimplex(minMat)
		print "min solutions ", minSoln
		print ""

		triConstraintMax = objMaxConstraint
		triConstraintMax += convertTriangleBoundsToConstraints(a, Vin, Vout, Vlow, Vhigh)
		triConstraintMax += overallConstraint
		print "triConstraintMax: ", "Vlow ", Vlow, " Vhigh ", Vhigh 
		print triConstraintMax
		maxMat = normalize(triConstraintMax)
		maxMat,maxSoln = dualSimplex(maxMat)

		print "max solutions ", maxSoln
		print ""


		if minSoln is not None and maxSoln is not None:
			if Vlow >= 0 and Vhigh >= 0:
				hypers.append([[minSoln[2]],[maxSoln[2]]])
				print "hyper found ",minSoln[2], maxSoln[2]
			elif Vlow <= 0 and Vhigh <= 0:
				hypers.append([[-minSoln[2]],[-maxSoln[2]]])
				print "hyper found ",-minSoln[2], -maxSoln[2]
			mats.append(minMat)
			mats.append(maxMat) 

		if i == 0 or i == len(bounds)-1 and triangle == False:
			trapConstraintMin = objMinConstraint
			trapConstraintMin += convertTrapezoidBoundsToConstraints(a, Vin, Vout, Vlow, Vhigh)
			trapConstraintMin += overallConstraint
			print "trapConstraintMin: ", "Vlow ", Vlow, " Vhigh ", Vhigh 
			print trapConstraintMin
			minMat = normalize(trapConstraintMin)
			minMat,minSoln = dualSimplex(minMat)
			print "min solutions ", minSoln
			print ""

			trapConstraintMax = objMaxConstraint
			trapConstraintMax += convertTrapezoidBoundsToConstraints(a, Vin, Vout, Vlow, Vhigh)
			trapConstraintMax += overallConstraint
			print "trapConstraintMax: ", "Vlow ", Vlow, " Vhigh ", Vhigh 
			print trapConstraintMax
			maxMat = normalize(trapConstraintMax)
			maxMat,maxSoln = dualSimplex(maxMat)
			print "max solutions ", maxSoln
			print ""

			if minSoln is not None and maxSoln is not None:
				if Vlow >= 0 and Vhigh >= 0:
					hypers.append([[minSoln[2]],[maxSoln[2]]])
					print "hyper found ",minSoln[2], maxSoln[2]
				elif Vlow <= 0 and Vhigh <= 0:
					hypers.append([[-minSoln[2]],[-maxSoln[2]]])
					print "hyper found ",-minSoln[2], -maxSoln[2]
				mats.append(minMat)
				mats.append(maxMat)

		
	print "hypers ", hypers
	return hypers

def fun2Constraints(bounds,a,params,triangle):
	Vins = ["x0", "x1"]
	Vouts = ["y0", "y1"]
	
	allBounds = []
	for i in range(len(Vins)):
		allBounds.append(bounds)

	allConstraints = []
	constraintBounds = []
	for i in range(len(Vins)):
		outConstraints = []
		constraintBound = []
		for j in range(len(bounds)):
			bound = allBounds[i][j]
			Vlow = bound[0][i]
			Vhigh = bound[1][i]
			constraintBound.append([Vlow, Vhigh])
			
			triConstraint = convertTriangleBoundsToConstraints(a, Vins[i], Vouts[i], Vlow, Vhigh)
			outConstraints.append(triConstraint)

			if j == 0 or j == len(bounds)-1:
				trapConstraint = convertTrapezoidBoundsToConstraints(a, Vins[i], Vouts[i], Vlow, Vhigh)
				outConstraints.append(trapConstraint)
				constraintBound.append([Vlow, Vhigh])

		allConstraints.append(outConstraints)
		constraintBounds.append(constraintBound)

	hypers = []
	mats = []
	for i in range(len(allConstraints[0])):
		for j in range(len(allConstraints[1])):
			print "i: ", i, " j: ", j
			constrainti = allConstraints[0][i]
			constraintj = allConstraints[1][j]
			boundi = constraintBounds[0][i]
			boundj = constraintBounds[1][j]
			overallConstraint = constrainti + constraintj
			
			objMinConstraintx1 = ""
			objMinConstraintx2 = ""
			objMaxConstraintx1 = ""
			objMaxConstraintx2 = ""

			if boundj[0] >= 0 and boundj[1] >= 0:
				if a <= 0:
					if boundi[0] <= 0 and boundi[1] <= 0:
						overallConstraint += "1 "+Vouts[0]+" + "+"-1 "+Vins[1]+" == "+str(-params[0])+"\n"
					elif boundi[0] >= 0 and boundi[1] >= 0:
						overallConstraint += "-1 "+Vouts[0]+" + "+"-1 "+Vins[1]+" == "+str(-params[0])+"\n"
				else:
					if boundi[0] >= 0 and boundi[1] >= 0:
						overallConstraint += "1 "+Vouts[0]+" + "+"-1 "+Vins[1]+" == "+str(-params[0])+"\n"
					elif boundi[0] <= 0 and boundi[1] <= 0:
						overallConstraint += "-1 "+Vouts[0]+" + "+"-1 "+Vins[1]+" == "+str(-params[0])+"\n"

				objMinConstraintx2 += "min 1 "+Vins[1]+"\n"
				objMaxConstraintx2 += "max 1 "+Vins[1]+"\n"
			
			elif boundj[0] <= 0 and boundj[1] <= 0:
				if a<= 0:
					if boundi[0] <= 0 and boundi[1] <= 0:
						overallConstraint += "1 "+Vouts[0]+" + "+"1 "+Vins[1]+" == "+str(-params[0])+"\n"
					elif boundi[0] >= 0 and boundi[1] >= 0:
						overallConstraint += "-1 "+Vouts[0]+" + "+"1 "+Vins[1]+" == "+str(-params[0])+"\n"
				else:
					if boundi[0] >= 0 and boundi[1] >= 0:
						overallConstraint += "1 "+Vouts[0]+" + "+"1 "+Vins[1]+" == "+str(-params[0])+"\n"
					elif boundi[0] <= 0 and boundi[1] <= 0:
						overallConstraint += "-1 "+Vouts[0]+" + "+"1 "+Vins[1]+" == "+str(-params[0])+"\n"
				objMinConstraintx2 += "min -1 "+Vins[1]+"\n"
				objMaxConstraintx2 += "max -1 "+Vins[1]+"\n"
			
			if boundi[0] >= 0 and boundi[1] >= 0:
				if a <= 0:
					if boundj[0] <= 0 and boundj[1] <= 0:
						overallConstraint += "1 "+Vouts[1]+" + "+"-1 "+Vins[0]+" == "+str(params[0])+"\n"
					elif boundj[0] >= 0 and boundj[1] >= 0:
						overallConstraint += "-1 "+Vouts[1]+" + "+"-1 "+Vins[0]+" == "+str(params[0])+"\n"
				else:
					if boundj[0] >= 0 and boundj[1] >= 0:
						overallConstraint += "1 "+Vouts[1]+" + "+"-1 "+Vins[0]+" == "+str(params[0])+"\n"
					elif boundj[0] <= 0 and boundj[1] <= 0:
						overallConstraint += "-1 "+Vouts[1]+" + "+"-1 "+Vins[0]+" == "+str(params[0])+"\n"
				objMinConstraintx1 += "min 1 "+Vins[0]+"\n"
				objMaxConstraintx1 += "max 1 "+Vins[0]+"\n"
			
			elif boundi[0] <= 0 and boundi[1] <= 0:
				if a <= 0:
					if boundj[0] <= 0 and boundj[1] <= 0:
						overallConstraint += "1 "+Vouts[1]+" + "+"1 "+Vins[0]+" == "+str(params[0])+"\n"
					elif boundj[0] >= 0 and boundj[1] >= 0:
						overallConstraint += "-1 "+Vouts[1]+" + "+"1 "+Vins[0]+" == "+str(params[0])+"\n"
				else:
					if boundj[0] >= 0 and boundj[1] >= 0:
						overallConstraint += "1 "+Vouts[1]+" + "+"1 "+Vins[0]+" == "+str(params[0])+"\n"
					elif boundj[0] <= 0 and boundj[1] <= 0:
						overallConstraint += "-1 "+Vouts[1]+" + "+"1 "+Vins[0]+" == "+str(params[0])+"\n"
				objMinConstraintx1 += "min -1 "+Vins[0]+"\n"
				objMaxConstraintx1 += "max -1 "+Vins[0]+"\n"
			
			overallConstraint += Vouts[0] + " >= 0 " + Vins[0] + " >= 0 " + Vouts[1] + " >= 0 " + Vins[1] + " >= 0"

			minx1Constraint = objMinConstraintx1 + overallConstraint
			maxx1Constraint = objMaxConstraintx1 + overallConstraint
			minx2Constraint = objMinConstraintx2 + overallConstraint
			maxx2Constraint = objMaxConstraintx2 + overallConstraint

			print "minx1Constraint"
			print minx1Constraint
			minx1Mat = normalize(minx1Constraint)
			minx1Mat, minx1Soln = dualSimplex(minx1Mat)
			print "minx1 soln ", minx1Soln
			print ""

			print "maxx1Constraint"
			print maxx1Constraint
			maxx1Mat = normalize(maxx1Constraint)
			maxx1Mat, maxx1Soln = dualSimplex(maxx1Mat)
			print "maxx1 soln ", maxx1Soln
			print ""

			print "minx2Constraint"
			print minx2Constraint
			minx2Mat = normalize(minx2Constraint)
			minx2Mat, minx2Soln = dualSimplex(minx2Mat)
			print "minx2 soln ", minx2Soln
			print ""

			print "maxx2Constraint"
			print maxx2Constraint
			maxx2Mat = normalize(maxx2Constraint)
			maxx2Mat, maxx2Soln = dualSimplex(maxx2Mat)
			print "maxx2 soln ", maxx2Soln
			print ""

			if minx1Soln is not None and maxx1Soln is not None and \
				minx2Soln is not None and maxx2Soln is not None:
				minBounds = []
				maxBounds = []
				if boundi[0] >= 0 and boundi[1] >= 0:
					print "positive boundi"
					minBounds.append(minx1Soln[2])
					maxBounds.append(maxx1Soln[2])
				elif boundi[0] <= 0 and boundi[1] <= 0:
					print "negative boundi"
					minBounds.append(-minx1Soln[2])
					maxBounds.append(-maxx1Soln[2])
				if boundj[0] >= 0 and boundj[1] >= 0:
					print "positive boundj"
					minBounds.append(minx2Soln[4])
					maxBounds.append(maxx2Soln[4])
				elif boundj[0] <= 0 and boundj[1] <= 0:
					print "negative boundj"
					minBounds.append(-minx2Soln[4])
					maxBounds.append(-maxx2Soln[4])
				hyper = [minBounds,maxBounds]
				print "hyper found ", hyper
				hypers.append(hyper)
				mats.append(minx1Mat)
				mats.append(minx2Mat)
				mats.append(maxx1Mat)
				mats.append(maxx2Mat)

	finalHypers = removeRedundantHypers(hypers)
	print "finalHypers "
	print finalHypers
	return finalHypers

#check if this specific ordering of intervals where the decision variables
# lie from bounds specified by intervalIndices is 
#feasible or not
def ifOrderingFeasibleOscl(bounds,a,g_cc,g_fwd,intervalIndices):
	lenV = len(bounds[0][0])
	V = []
	VoutFwd = []
	VoutCc = []
	decVariableConstraint = ""
	for i in range(lenV):
		variable = "v"+str(i)
		decVariableConstraint += variable + " >= 0 "
		V.append(variable)
		variable = "voutfwd" + str(i)
		decVariableConstraint += variable + " >= 0 "
		VoutFwd.append(variable)
		variable = "voutcc"+str(i)
		decVariableConstraint += variable + " >= 0 "
		VoutCc.append(variable)

	constraint = constructBasicConstraints(V,VoutFwd,VoutCc,bounds,a,g_cc,g_fwd,intervalIndices)
	objConstraint = "min 1 v0\n"
	constraint = objConstraint + constraint + decVariableConstraint
	#print "constraint"
	#print constraint
	mat = normalize(constraint)
	mat, soln = dualSimplex(mat)
	#print "soln ", soln

	if soln is None:
		print "Not feasible"
		return False

	print "Feasible"
	return True

# basic constraints for rambus oscillator without any objective function
def constructBasicConstraints(V,VoutFwd,VoutCc,bounds,a,g_cc,g_fwd,intervalIndices):
	lenV = len(bounds[0][0])
	Vin = [V[i % lenV] for i in range(-1,lenV-1)]
	Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
	constraint = ""
	for i in range(lenV):
		fwdIndex = (i-1)%lenV
		ccIndex = (i+lenV/2)%lenV
		boundfwdInd = intervalIndices[fwdIndex]
		boundccInd = intervalIndices[ccIndex]
		boundiInd = intervalIndices[i]
		VlowFwd, VhighFwd = None, None
		VlowCc, VhighCc = None, None
		Vlowi, Vhighi = None, None
		finalConstraint = None

		if boundiInd == -1:
			Vlowi = bounds[0][0][i]
			Vhighi = bounds[0][1][i]
		elif boundiInd == len(bounds):
			Vlowi = bounds[len(bounds)-1][0][i]
			Vhighi = bounds[len(bounds)-1][1][i]
		elif boundiInd is not None:
			Vlowi = bounds[boundiInd][0][i]
			Vhighi = bounds[boundiInd][1][i]

		if boundfwdInd == -1:
			VlowFwd = bounds[0][0][fwdIndex]
			VhighFwd = bounds[0][1][fwdIndex]
			claimTrapFwd = convertTrapezoidBoundsToConstraints(a, Vin[i], VoutFwd[i], VlowFwd, VhighFwd)
			constraint += claimTrapFwd
		elif boundfwdInd == len(bounds):
			VlowFwd = bounds[len(bounds)-1][0][fwdIndex]
			VhighFwd = bounds[len(bounds)-1][1][fwdIndex]
			claimTrapFwd = convertTrapezoidBoundsToConstraints(a, Vin[i], VoutFwd[i], VlowFwd, VhighFwd)
			constraint += claimTrapFwd
		elif boundfwdInd is not None:
			VlowFwd = bounds[boundfwdInd][0][fwdIndex]
			VhighFwd = bounds[boundfwdInd][1][fwdIndex]
			claimTriFwd = convertTriangleBoundsToConstraints(a, Vin[i], VoutFwd[i], VlowFwd, VhighFwd)
			constraint += claimTriFwd

		if boundccInd == -1:
			VlowCc = bounds[0][0][ccIndex]
			VhighCc = bounds[0][1][ccIndex]
			claimTrapCc = convertTrapezoidBoundsToConstraints(a, Vcc[i], VoutCc[i], VlowCc, VhighCc)
			constraint += claimTrapCc

		elif boundccInd == len(bounds):
			VlowCc = bounds[len(bounds)-1][0][ccIndex]
			VhighCc = bounds[len(bounds)-1][1][ccIndex]
			claimTrapCc = convertTrapezoidBoundsToConstraints(a, Vcc[i], VoutCc[i], VlowCc, VhighCc)
			constraint += claimTrapCc

		elif boundccInd is not None:
			VlowCc = bounds[boundccInd][0][ccIndex]
			VhighCc = bounds[boundccInd][1][ccIndex]
			claimTriCc = convertTriangleBoundsToConstraints(a, Vcc[i], VoutCc[i], VlowCc, VhighCc)
			constraint += claimTriCc

		constFwd = g_fwd
		constCc = g_cc
		constVi1 = -g_fwd
		constVi2 = -g_cc

		if Vlowi <= 0 and Vhighi <=0:
			constVi1 = -constVi1
			constVi2 = -constVi2
			if i == 0:
				objConstraint = "min -1 v0\n"
		if a <= 0 and VlowFwd >= 0 and VhighFwd >= 0:
			constFwd = -constFwd
		elif a >= 0 and VlowFwd <= 0 and VhighFwd <= 0:
			constFwd = -constFwd

		if a <= 0 and VlowCc >= 0 and VhighCc >= 0:
			constCc = -constCc
		elif a >= 0 and VlowCc <= 0 and VhighCc <= 0:
			constCc = -constCc

		finalConstraint = str(constFwd)+" "+VoutFwd[i]+" + "+str(constVi1)+" "+V[i]\
							+ " + "+str(constCc)+" "+VoutCc[i]+" + "+str(constVi2)+" "+V[i]+" == 0\n"
		constraint += finalConstraint

	return constraint
		

#given the appropriate feasible interval index for each decision variable
#find initial hyperrectangles
def createInitialHyperRectangles(bounds,a,g_cc,g_fwd,intervalIndices):
	lenV = len(bounds[0][0])
	V = []
	VoutFwd = []
	VoutCc = []
	decVariableConstraint = ""
	for i in range(lenV):
		variable = "v"+str(i)
		decVariableConstraint += variable + " >= 0 "
		V.append(variable)
		variable = "voutfwd" + str(i)
		decVariableConstraint += variable + " >= 0 "
		VoutFwd.append(variable)
		variable = "voutcc"+str(i)
		decVariableConstraint += variable + " >= 0 "
		VoutCc.append(variable)

	Vin = [V[i % lenV] for i in range(-1,lenV-1)]
	Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
	constraint = ""
	for i in range(lenV):
		fwdIndex = (i-1)%lenV
		ccIndex = (i+lenV/2)%lenV
		boundfwdInd = intervalIndices[fwdIndex]
		boundccInd = intervalIndices[ccIndex]
		boundiInd = intervalIndices[i]
		VlowFwd, VhighFwd = None, None
		VlowCc, VhighCc = None, None
		Vlowi, Vhighi = None, None
		finalConstraint = None

		if boundiInd == -1:
			Vlowi = bounds[0][0][i]
			Vhighi = bounds[0][1][i]
		elif boundiInd == len(bounds):
			Vlowi = bounds[len(bounds)-1][0][i]
			Vhighi = bounds[len(bounds)-1][1][i]
		elif boundiInd is not None:
			Vlowi = bounds[boundiInd][0][i]
			Vhighi = bounds[boundiInd][1][i]

		if boundfwdInd == -1:
			VlowFwd = bounds[0][0][fwdIndex]
			VhighFwd = bounds[0][1][fwdIndex]
			claimTrapFwd = convertTrapezoidBoundsToConstraints(a, Vin[i], VoutFwd[i], VlowFwd, VhighFwd)
			constraint += claimTrapFwd
		elif boundfwdInd == len(bounds):
			VlowFwd = bounds[len(bounds)-1][0][fwdIndex]
			VhighFwd = bounds[len(bounds)-1][1][fwdIndex]
			claimTrapFwd = convertTrapezoidBoundsToConstraints(a, Vin[i], VoutFwd[i], VlowFwd, VhighFwd)
			constraint += claimTrapFwd
		elif boundfwdInd is not None:
			VlowFwd = bounds[boundfwdInd][0][fwdIndex]
			VhighFwd = bounds[boundfwdInd][1][fwdIndex]
			claimTriFwd = convertTriangleBoundsToConstraints(a, Vin[i], VoutFwd[i], VlowFwd, VhighFwd)
			constraint += claimTriFwd

		if boundccInd == -1:
			VlowCc = bounds[0][0][ccIndex]
			VhighCc = bounds[0][1][ccIndex]
			claimTrapCc = convertTrapezoidBoundsToConstraints(a, Vcc[i], VoutCc[i], VlowCc, VhighCc)
			constraint += claimTrapCc

		elif boundccInd == len(bounds):
			VlowCc = bounds[len(bounds)-1][0][ccIndex]
			VhighCc = bounds[len(bounds)-1][1][ccIndex]
			claimTrapCc = convertTrapezoidBoundsToConstraints(a, Vcc[i], VoutCc[i], VlowCc, VhighCc)
			constraint += claimTrapCc

		elif boundccInd is not None:
			VlowCc = bounds[boundccInd][0][ccIndex]
			VhighCc = bounds[boundccInd][1][ccIndex]
			claimTriCc = convertTriangleBoundsToConstraints(a, Vcc[i], VoutCc[i], VlowCc, VhighCc)
			constraint += claimTriCc

		constFwd = g_fwd
		constCc = g_cc
		constVi1 = -g_fwd
		constVi2 = -g_cc

		if Vlowi <= 0 and Vhighi <=0:
			constVi1 = -constVi1
			constVi2 = -constVi2
			if i == 0:
				objConstraint = "min -1 v0\n"
		if a <= 0 and VlowFwd >= 0 and VhighFwd >= 0:
			constFwd = -constFwd
		elif a >= 0 and VlowFwd <= 0 and VhighFwd <= 0:
			constFwd = -constFwd

		if a <= 0 and VlowCc >= 0 and VhighCc >= 0:
			constCc = -constCc
		elif a >= 0 and VlowCc <= 0 and VhighCc <= 0:
			constCc = -constCc

		finalConstraint = str(constFwd)+" "+VoutFwd[i]+" + "+str(constVi1)+" "+V[i]\
							+ " + "+str(constCc)+" "+VoutCc[i]+" + "+str(constVi2)+" "+V[i]+" == 0\n"
		constraint += finalConstraint

	objConstraint = "min 1 v0\n"
	constraint = objConstraint + constraint + decVariableConstraint
	#print "constraint"
	#print constraint
	mat = normalize(constraint)
	mat, soln = dualSimplex(mat)
	#print "soln ", soln

	if soln is None:
		print "Not feasible"
		return False

	print "Feasible"
	return True

			
def removeRedundantHypers(hypers):
	finalHypers = []
	for i in range(len(hypers)):
		hyperi = hypers[i]
		if len(finalHypers) == 0:
			finalHypers.append(hyperi)
			continue
		foundSameHyperrectangle = False
		for j in range(len(finalHypers)):
			hyperj = finalHypers[j]
			sameHyperrectangle = True
			for k in range(len(hyperi[0])):
				if hyperi[k] != hyperj[k]:
					sameHyperrectangle = False
					break
			if sameHyperrectangle:
				foundSameHyperrectangle = True
				break
		if foundSameHyperrectangle == False:
			finalHypers.append(hyperi)

	return finalHypers

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


def multiplyRegularMatWithIntervalMat(regMat,intervalMat):
	mat1 = np.dot(regMat,intervalMat[:,:,0])
	mat2 = np.dot(regMat,intervalMat[:,:,1])
	result = np.zeros((regMat.shape[0],regMat.shape[1],2))
	result[:,:,0] = np.minimum(mat1,mat2)
	result[:,:,1] = np.maximum(mat1,mat2)
	return result

def subtractIntervalMatFromRegularMat(regMat,intervalMat):
	mat1 = regMat - intervalMat[:,:,0]
	mat2 = regMat - intervalMat[:,:,1]
	result = np.zeros((regMat.shape[0],regMat.shape[1],2))
	result[:,:,0] = np.minimum(mat1,mat2)
	result[:,:,1] = np.maximum(mat1,mat2)
	return result

def multiplyIntervalMatWithIntervalVec(mat,vec):
	mat1 = np.dot(mat[:,:,0],vec[:,0])
	mat2 = np.dot(mat[:,:,1],vec[:,0])
	mat3 = np.dot(mat[:,:,0],vec[:,1])
	mat4 = np.dot(mat[:,:,1],vec[:,1])
	result = np.zeros((mat.shape[0],vec.shape[1]))
	result[:,0] = np.minimum(np.minimum(mat1,mat2),np.minimum(mat3,mat4))
	result[:,1] = np.maximum(np.maximum(mat1,mat2),np.maximum(mat3,mat4))
	return result


#Check existence and uniqueness of solution using Krawczyk operator
def checkExistenceOfSolution(a,params,hyperRectangle,funNum,funDer,funDerInterval):
	print "lower bounds ", hyperRectangle[0]
	print "upper bounds ",hyperRectangle[1]
	numVolts = len(hyperRectangle[0])

	startBounds = np.zeros((numVolts,2))
	startBounds[:,0] = hyperRectangle[0]
	startBounds[:,1] = hyperRectangle[1]

	iteration = 0
	while True:
		print "iteration number: ", iteration
		midPoint = (startBounds[:,0] + startBounds[:,1])/2.0
		print "midPoint"
		print midPoint
		IMidPoint = funNum(midPoint,a,params)
		jacMidPoint = funDer(midPoint,a,params)
		C = np.linalg.inv(jacMidPoint)
		I = np.identity(numVolts)

		jacInterval = funDerInterval(a,params,startBounds)
		C_IMidPoint = np.dot(C,IMidPoint)

		C_jacInterval = multiplyRegularMatWithIntervalMat(C,jacInterval)
		I_minus_C_jacInterval = subtractIntervalMatFromRegularMat(I,C_jacInterval)
		xi_minus_midPoint = np.zeros((numVolts,2))
		for i in range(numVolts):
			xi_minus_midPoint[i][0] = startBounds[i][0] - midPoint[i]
			xi_minus_midPoint[i][1] = startBounds[i][1] - midPoint[i]

		lastTerm = multiplyIntervalMatWithIntervalVec(I_minus_C_jacInterval, xi_minus_midPoint)
		
		kInterval1 = midPoint - C_IMidPoint + lastTerm[:,0]
		kInterval2 = midPoint - C_IMidPoint + lastTerm[:,1]
		kInterval = np.zeros((numVolts,2))
		kInterval[:,0] = np.minimum(kInterval1, kInterval2)
		kInterval[:,1] = np.maximum(kInterval1, kInterval2)

		print "kInterval "
		print kInterval

		uniqueSoln = True
		for i in range(numVolts):
			if kInterval[i][0] <= startBounds[i][0] or kInterval[i][0] >= startBounds[i][1]:
				uniqueSoln = False
			if kInterval[i][1] <= startBounds[i][0] or kInterval[i][1] >= startBounds[i][1]:
				uniqueSoln = False

		if uniqueSoln:
			print "Hyperrectangle with unique solution found"
			startBounds = np.transpose(startBounds)
			print startBounds
			return (True,startBounds)

		intersect = np.zeros((numVolts,2))
		for i in range(numVolts):
			minVal = max(kInterval[i][0],startBounds[i][0])
			maxVal = min(kInterval[i][1],startBounds[i][1])
			if minVal <= maxVal and \
				minVal >= kInterval[i][0] and minVal >= startBounds[i][0] and \
				minVal <= kInterval[i][1] and minVal <= startBounds[i][1] and \
				maxVal >= kInterval[i][0] and maxVal >= startBounds[i][0] and \
				maxVal <= kInterval[i][1] and maxVal <= startBounds[i][1]:
				intersect[i] = [minVal,maxVal]
				intervalLength =  intersect[:,1] - intersect[:,0]
			else:
				intersect = None
				break

		print "intersect"
		print intersect

		if intersect is None:
			print "hyperrectangle does not contain any solution"
			return (False,None)
		elif np.linalg.norm(intervalLength) < 1e-8:
			if np.sum(np.absolute(intersect)) == 0:
				print "Hyperrectangle with unique solution found - same bounds"
				intersect = np.transpose(intersect)
				print intersect
				return (True,intersect)

			print "Found smallest hyperrectangle containing solution"
			intersect = np.transpose(intersect)
			print intersect
			return (False,intersect)
		else:
			startBounds = intersect
		iteration += 1

def findUniqueHypers():
	'''a = 1.0
	params = [0.3,0.1]
	bounds = [[[-0.5],[0.0]],[[0.0],[0.5]]]
	funConstraints = fun1Constraints
	hypers = fun1Constraints(bounds,a,params,False)
	funNum = fun1Num
	funDer = fun1Der
	funDerInterval = fun1DerInterval'''
	a = -5.0
	params = [0.0]
	bounds = [[[-0.5,-0.5],[0.0,0.0]],[[0.0,0.0],[0.5,0.5]]]
	funConstraints = fun2Constraints
	funNum = fun2Num
	funDer = fun2Der
	funDerInterval = fun2DerInterval
	finalHypers = []

	hypers = funConstraints(bounds,a,params,False)
	while len(hypers)!=0:
		tempHypers = []
		for i in range(len(hypers)):
			print "Checking existience within hyperrectangle ", i
			uniqueness,interval = checkExistenceOfSolution(a,params,hypers[i],funNum,funDer,funDerInterval)
			if uniqueness:
				print "hyperrectangle contains unique solution"
				finalHypers.append(hypers[i])
			else:
				if interval is not None:
					print "need to refine more"
					hyper = funConstraintsScale(hypers[i],a,params,True)
					tempHypers += hyper
				else:
					print "no solution in hyperrectangle"
			print ""
		hypers = tempHypers

#Data structure to keep track all possible combination of interval indices
class combinationNode:
	def __init__(self,rootArray):
		self.rootArray = copy.deepcopy(rootArray)
		self.children = []

	def addChild(self,childArray):
		childNode = combinationNode(childArray)
		self.children.append(childNode)

def printCombinationNode(rootCombinationNode):
	rootArray = rootCombinationNode.rootArray
	print rootArray
	for i in range(len(rootCombinationNode.children)):
		printCombinationNode(rootCombinationNode.children[i])



def addCombinationsAsChildren(rootCombinationNode,intervalIndexRange):
	theArray = rootCombinationNode.rootArray
	if theArray[len(theArray)-1]!= None:
		return
	possibleIntervalIndices = range(intervalIndexRange[0],intervalIndexRange[1]+1)
	indexOfNone = 0
	for i in range(len(theArray)):
		if theArray[i] == None:
			indexOfNone = i
			break
	for i in range(len(possibleIntervalIndices)):
		childArray = copy.deepcopy(theArray)
		childArray[indexOfNone] = possibleIntervalIndices[i]
		rootCombinationNode.addChild(childArray)

	for i in range(len(rootCombinationNode.children)):
		addCombinationsAsChildren(rootCombinationNode.children[i],intervalIndexRange)

# intervalIndexRange is inclusive for both lower and upper bound
def combinationWithTrees(numIntervalIndices, intervalIndexRange):
	possibleIntervalIndices = range(intervalIndexRange[0],intervalIndexRange[1]+1)
	rootCombinationNodes = []
	for i in range(len(possibleIntervalIndices)):
		arr = [possibleIntervalIndices[i]]
		for j in range(1,numIntervalIndices):
			arr.append(None)
		combNode = combinationNode(arr)
		rootCombinationNodes.append(combNode)
	for i in range(len(rootCombinationNodes)):
		addCombinationsAsChildren(rootCombinationNodes[i],intervalIndexRange)
	
	return rootCombinationNodes
	#printCombinationNode(rootCombinationNodes[0])

def getFeasibleIntervalIndices(rootCombinationNode,a,g_cc,g_fwd,bounds,validIntervalIndices):
	intervalIndices = rootCombinationNode.rootArray
	feasible = ifOrderingFeasibleOscl(bounds,a,g_cc,g_fwd,intervalIndices)
	if feasible == False:
		return
	indexOfNone = None
	for i in range(len(intervalIndices)):
		if intervalIndices[i] is None:
			indexOfNone = i
			break
	if indexOfNone is None:
		validIntervalIndices.append(copy.deepcopy(intervalIndices))
	for i in range(len(rootCombinationNode.children)):
		getFeasibleIntervalIndices(rootCombinationNode.children[i],a,g_cc,g_fwd,bounds,validIntervalIndices)


def osclTest():
	bounds = [[[-0.5,-0.5,-0.5,-0.5],[0.0,0.0,0.0,0.0]],
				[[0.0,0.0,0.0,0.0],[0.5,0.5,0.5,0.5]]]
	a = -5.0
	g_cc = 0.5
	g_fwd = 1.0

	rootCombinationNodes = combinationWithTrees(4,[-1,2])
	feasibleIntervalIndices = []
	for i in range(len(rootCombinationNodes)):
		getFeasibleIntervalIndices(rootCombinationNodes[i],a,g_cc,g_fwd,bounds,feasibleIntervalIndices)
	print "validIntervalIndices ", feasibleIntervalIndices
	


if __name__=="__main__":
	osclTest()

# normalize LP - for min, should be greater than equal to. for max should be less than equal to
#first do simple simplex and then do dual simplex

