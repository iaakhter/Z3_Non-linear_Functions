import numpy as np
import copy

# if not dual feasible, check primal feasibility. if primal feasible apply normal simplex
# otherwise modified simplex. then dual simplex. check if primal simplex is implemented correctly

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
	print "PRIMAL SIMPLEX"
	standardMat = copy.deepcopy(theMat)
	numRows = standardMat.shape[0]
	numCols = standardMat.shape[1]
	count = 0
	while len(np.where(standardMat[0,:numCols-1] < 0)[0]) > 0:
		'''print "mat first row"
		print standardMat[0]
		print ""
		print "mat last column"
		print standardMat[:,numCols-1]
		print ""'''
		enteringVariables = np.where(standardMat[0,:] < 0)[0]
		minInRow = np.argsort(standardMat[0,:])
		enteringVariable, pivot, possiblePivots = None, None, None
		enteringVariable = minInRow[0]
		if enteringVariable == numCols-1:
			enteringVariable = minInRow[1]
		ratios = np.divide(standardMat[:,numCols - 1],standardMat[:,enteringVariable])
		ratioSortedIndices = np.argsort(ratios)
		possiblePivots = []

		for ind in ratioSortedIndices:
			if ind != 0 and standardMat[ind,enteringVariable] > 0:
				possiblePivots.append(ind)

		#print "entering variable ", enteringVariable
		#print "entering variable column"
		#print standardMat[:,enteringVariable]
		#print "pivot ", pivot
		largestMatValue = standardMat[possiblePivots[0]][enteringVariable]
		pivot = possiblePivots[0]
		if len(possiblePivots) > 1:
			ind1 = possiblePivots[0]
			ind2 = possiblePivots[1]
			p = 1
			if ratios[ind1] != ratios[ind2]:
				pivot = ind1
			else:
				while p < len(possiblePivots) and ratios[ind1] == ratios[possiblePivots[p]]:
					ind2 = possiblePivots[p]
					if standardMat[ind2][enteringVariable] > largestMatValue:
						pivot = ind2
						largestMatValue = standardMat[ind2][enteringVariable]
					p+=1

		standardMat = gauss_jordan(standardMat,pivot,enteringVariable)
		'''print "mat after "
		print standardMat
		print ""'''
		count+=1
	sols = np.zeros((numCols - 1))
	for i in range(len(sols)):
		if len(np.where(standardMat[:,i]==0)[0]) == numRows - 1:
			nonzeroIndex = np.where(standardMat[:,i] != 0)[0][0]
			#print "i ", i, "nonzeroIndex ", nonzeroIndex
			if standardMat[nonzeroIndex][i] == 1:
				#print "coming here?"
				sols[i] = standardMat[nonzeroIndex][numCols-1]
	print ""
	return standardMat, sols



def dualSimplex(theMat):
	normalizedMat = copy.deepcopy(theMat)
	numRows = normalizedMat.shape[0]
	numCols = normalizedMat.shape[1]
	count = 0
	if len(np.where(normalizedMat[0]<0)[0]) != 0:
		print "Optimality condition is not met. Cannot apply dual complex"
		return None, None
	
	while len(np.where(normalizedMat[1:,numCols-1] < 0)[0]) > 0:
		print "normalizedMat first row"
		print normalizedMat[0]
		print "normalizedMat last column"
		print normalizedMat[:,numCols-1]
		enteringVariable, pivot = None, None
		minInCol = np.argsort(normalizedMat[:,numCols-1])
		pivot = minInCol[0]
		if pivot== 0:
			pivot = minInCol[1]

		numNegsInRow = len(np.where(normalizedMat[pivot,:numCols-1] < 0)[0])
		if numNegsInRow == 0:
			print "No feasible solution"
			return None, None
		ratios = np.divide(np.absolute(normalizedMat[0,:numCols - 1]),np.absolute(normalizedMat[pivot,:numCols-1]))
		ratioSortedIndices = np.argsort(ratios)
		'''print "ratios ", ratios
		print "ratioIndices", ratioSortedIndices'''
		possibleEnteringVariables = []
		for ind in ratioSortedIndices:
			if normalizedMat[pivot,ind] < 0:
				possibleEnteringVariables.append(ind)

		enteringVariable = possibleEnteringVariables[0]
		
		print "FINALpivot ", pivot, "enteringVariable ", enteringVariable
		print normalizedMat[pivot]
		normalizedMat = gauss_jordan(normalizedMat,pivot,enteringVariable)
		count+=1
		'''if count == 2:
			return'''
	sols = np.zeros((numCols - 1))
	for i in range(len(sols)):
		if len(np.where(normalizedMat[:,i]==0)[0]) == numRows - 1:
			nonzeroIndex = np.where(normalizedMat[:,i] != 0)[0][0]
			if normalizedMat[nonzeroIndex][i] == 1:
				sols[i] = normalizedMat[nonzeroIndex][numCols-1]
	#print "normalizedMat "
	#print normalizedMat[:,normalizedMat.shape[1]-1]
	print ""
	return normalizedMat, sols



def generalizedSimplex(theMat):
	normalizedMat = copy.deepcopy(theMat)
	numRows = normalizedMat.shape[0]
	numCols = normalizedMat.shape[1]
	numNegsIn1stRow = len(np.where(normalizedMat[0,:numCols-1] < 0)[0])
	numNegsInLastCol = len(np.where(normalizedMat[1:,numCols-1] < 0)[0])
	'''print "normalizedMat first row"
	print normalizedMat[0]
	print "normalizedMat last column"
	print normalizedMat[:,numCols-1]'''

	if numNegsIn1stRow != 0 and numNegsInLastCol != 0:
		#print "before simplex"
		#print normalizedMat[0]
		print "MAKING TABLEAU FEASIBLE"
		normalizedMat,soln = dualSimplex(normalizedMat)
		if soln is None:
			print "no feasible solution"
			return None, None
		numNegsIn1stRow = len(np.where(normalizedMat[0,:numCols-1] < 0)[0])
		numNegsInLastCol = len(np.where(normalizedMat[1:,numCols-1] < 0)[0])
		#print "normalizedMat first row"
		#print normalizedMat[0]
		#print "normalizedMat last column"
		#print normalizedMat[:,numCols-1]
		#return normalizedMat,soln
		#print "mat after simplex "
		#print normalizedMat[0:5]
		#print "soln ", soln
		#return
		#normalizedMat = makeBasisDualFeasible(normalizedMat)
		#numRows = normalizedMat.shape[0]
		#numCols = normalizedMat.shape[1]
		#print "after dual feasible process"
		#print "normalizedMat firstRow"
		#print normalizedMat[0]
		#print "Optimality condition is not met"
		#return None, None

		if numNegsInLastCol == 0:
			'''print "normalizedMat first row"
			print normalizedMat[0]
			print "normalizedMat last column"
			print normalizedMat[:,numCols-1]'''
			normalizedMat,soln = simplex(normalizedMat)
			if soln is None:
				print "solution is unbounded in primal simplex"
				return None, None
		return normalizedMat,soln
	
	elif numNegsIn1stRow == 0:
		print "START DUAL SIMPLEX"
		normalizedMat,soln = dualSimplex(normalizedMat)
		if soln is None:
			print "no feasible solution"
			return None, None
		#print "normalizedMat first row"
		#print normalizedMat[0]
		#print "normalizedMat last column"
		#print normalizedMat[:,numCols-1]
		return normalizedMat,soln

	elif numNegsInLastCol == 0:
		normalizedMat,soln = simplex(normalizedMat)
		if soln is None:
			print "solution is unbounded in primal simplex"
			return None, None
		#print "normalizedMat first row"
		#print normalizedMat[0]
		#print "normalizedMat last column"
		#print normalizedMat[:,numCols-1]	
		return normalizedMat,soln

def gauss_jordan(mat, pivot, enteringVariable):
	#print "gauss_jordan before pivoting pivot row ", mat[pivot]
	mat[pivot] = mat[pivot]/mat[pivot][enteringVariable]
	#print "gauss_jordan after pivoting pivot row ", mat[pivot]
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


'''simulate oscillation with python numberical values to check if the
z3 solutions make sense'''
def oscNum(V,a,params):
	g_fwd = params[0]
	g_cc = params[1]
	lenV = len(V)
	Vin = [V[i % lenV] for i in range(-1,lenV-1)]
	Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
	VoutFwd = [tanhFun(a,Vin[i]) for i in range(lenV)]
	VoutCc = [tanhFun(a,Vcc[i]) for i in range(lenV)]
	return [((tanhFun(a,Vin[i])-V[i])*g_fwd
			+(tanhFun(a,Vcc[i])-V[i])*g_cc) for i in range(lenV)]


'''Get jacobian of rambus oscillator at V
'''
def getJacobian(V,a,params):
	g_fwd = params[0]
	g_cc = params[1]
	lenV = len(V)
	Vin = [V[i % lenV] for i in range(-1,lenV-1)]
	Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
	jacobian = np.zeros((lenV, lenV))
	for i in range(lenV):
		jacobian[i,i] = -(g_fwd+g_cc)
		jacobian[i,(i-1)%lenV] = g_fwd*tanhFunder(a,V[(i-1)%lenV])
		jacobian[i,(i + lenV/2) % lenV] = g_cc*tanhFunder(a,V[(i + lenV/2) % lenV])

	return jacobian

def getJacobianInterval(a,params,bounds):
	#print "bounds in getJacobianInterval"
	#print bounds
	g_fwd = params[0]
	g_cc = params[1]
	lowerBound = bounds[:,0]
	upperBound = bounds[:,1]
	lenV = len(lowerBound)
	jacobian = np.zeros((lenV, lenV,2))
	jacobian[:,:,0] = jacobian[:,:,0] 
	jacobian[:,:,1] = jacobian[:,:,1]
	for i in range(lenV):
		jacobian[i,i,0] = -(g_fwd+g_cc)
		jacobian[i,i,1] = -(g_fwd+g_cc)
		gfwdVal1 = g_fwd*tanhFunder(a,lowerBound[(i-1)%lenV])
		gfwdVal2 = g_fwd*tanhFunder(a,upperBound[(i-1)%lenV])
		jacobian[i,(i-1)%lenV,0] = min(gfwdVal1,gfwdVal2)
		jacobian[i,(i-1)%lenV,1] = max(gfwdVal1,gfwdVal2)
		gccVal1 = g_cc*tanhFunder(a,lowerBound[(i + lenV/2) % lenV])
		gccVal2 = g_cc*tanhFunder(a,upperBound[(i + lenV/2) % lenV])
		jacobian[i,(i + lenV/2) % lenV,0] = min(gccVal1,gccVal2)
		jacobian[i,(i + lenV/2) % lenV,1] = max(gccVal1,gccVal2)

	return jacobian

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
			constraint = "1.0 " + Vout+" + "+str(-dThird) + " " + Vin +" >= "+str(cThird)+"\n"
			constraint += "1.0 " + Vout+" + "+str(-dLow) + " " + Vin +" <= "+str(cLow)+"\n"
			constraint += "1.0 " + Vout+" + "+str(-dHigh) + " " + Vin +" <= "+str(cHigh)+"\n"
			'''if Vlow != 0:
				constraint += "1 " + Vin + " >= "+str(Vlow)+"\n"
			constraint += "1 " + Vin + " <= "+str(Vhigh)+"\n"'''

		elif Vlow <=0 and Vhigh <=0:
			#constraint = "max -1 "+Vout+"\n"
			constraint = "-1.0 " + Vout + " + " +str(dThird)+" "+Vin+" <= "+str(cThird)+"\n"
			constraint += "-1.0 " + Vout + " + " +str(dLow)+" "+Vin+" >= "+str(cLow)+"\n"
			constraint += "-1.0 " + Vout + " + " +str(dHigh)+" "+Vin+" >= "+str(cHigh)+"\n"
			'''constraint += "-1 " + Vin + " >= "+str(Vlow)+"\n"
			if Vhigh != 0:
				constraint += "-1 " + Vin + " <= "+str(Vhigh)+"\n"'''
	
	elif a < 0:
		if Vlow <= 0 and Vhigh <=0:
			#constraint = "min 1 "+Vout+"\n"
			constraint = "1.0 " + Vout+" + "+str(dThird) + " " + Vin +" >= "+str(cThird)+"\n"
			constraint += "1.0 " + Vout+" + "+str(dLow) + " " + Vin +" <= "+str(cLow)+"\n"
			constraint += "1.0 " + Vout+" + "+str(dHigh) + " " + Vin +" <= "+str(cHigh)+"\n"
			'''constraint += "-1 " + Vin + " >= "+str(Vlow)+"\n"
			if Vhigh != 0:
				constraint += "-1 " + Vin + " <= "+str(Vhigh)+"\n"'''
		
		elif Vlow >=0 and Vhigh >=0:
			#constraint = "max -1 "+Vout+"\n"
			constraint = "-1.0 " + Vout + " + "+str(-dThird)+" "+Vin+" <= "+str(cThird)+"\n"
			constraint += "-1.0 " + Vout + " + "+str(-dLow)+" "+Vin+" >= "+str(cLow)+"\n"
			constraint += "-1.0 " + Vout + " + "+str(-dHigh)+" "+Vin+" >= "+str(cHigh)+"\n"
			'''if Vlow != 0:
				constraint += "1 " + Vin + " >= "+str(Vlow)+"\n"
			constraint += "1 " + Vin + " <= "+str(Vhigh)+"\n"'''
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
			constraint = "-1.0 "+Vout+" + "+str(dLow)+" "+Vin+" >= "+str(cLow)+"\n"
			constraint += "-1.0 "+Vout+" >= "+"-1\n"
			constraint += "-1.0 "+Vout+" <= "+str(tanhFunVlow)+"\n"
			#constraint += "-1 "+Vin+" <= "+str(Vlow)+"\n"
		elif Vlow >= 0 and Vhigh >=0:
			#constraint = "min 1 "+Vout+"\n"
			constraint = "1.0 "+Vout+" + "+str(-dHigh)+" "+Vin+" <= "+str(cHigh)+"\n"
			constraint += "1.0 "+Vout+" <= "+"1\n"
			constraint += "1.0 "+Vout+" >= "+str(tanhFunVhigh)+"\n"
			#constraint += "1 "+Vin+" >= "+str(Vhigh)+"\n"

	elif a < 0:
		if Vlow <= 0 and Vhigh <= 0:
			#constraint = "min 1 "+Vout+"\n"
			constraint = "1.0 "+Vout+" + "+str(dLow)+" "+Vin+" <= "+str(cLow)+"\n"
			constraint += "1.0 "+Vout+" <= "+"1.0\n"
			constraint += "1.0 "+Vout+" >= "+str(tanhFunVlow)+"\n"
			#constraint += "-1 "+Vin+" <= "+str(Vlow)+"\n"
		elif Vlow >= 0 and Vhigh >=0:
			#constraint = "max -1 "+Vout+"\n"
			constraint = "-1.0 "+Vout+" + "+str(-dHigh)+" "+Vin+" >= "+str(cHigh)+"\n"
			constraint += "-1.0 "+Vout+" >= "+"-1\n"
			constraint += "-1.0 "+Vout+" <= "+str(tanhFunVhigh)+"\n"
			#constraint += "1 "+Vin+" >= "+str(Vhigh)+"\n"

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

	print "hypers "
	print hypers
	finalHypers = removeRedundantHypers(hypers)
	print "finalHypers "
	print finalHypers
	return finalHypers

def checkArrayEqualities(arr1, arr2):
	for i in range(len(arr1)):
		if arr1[i] != arr2[i]:
			return False
	return True

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

	minConstant = 1
	objFun = "min "
	vind = 2
	bound0Ind = intervalIndices[vind]
	if bound0Ind == -1:
		minConstant = -1
		objFun = "max "
	elif bound0Ind is not None and bound0Ind < len(bounds):
		Vlow = bounds[bound0Ind][0][vind]
		Vhigh = bounds[bound0Ind][1][vind]
		if Vlow <=0 and Vhigh <= 0:
			minConstant = -1
			objFun = "max "

	objConstraint = objFun+str(minConstant)+" v2\n"

	constraint = constructBasicConstraints(V,VoutFwd,VoutCc,bounds,a,g_cc,g_fwd,intervalIndices)

	constraint = objConstraint + constraint + decVariableConstraint
	print "constraint"
	print constraint
	mat = normalize(constraint)
	mat, soln = generalizedSimplex(mat)

	
	if soln is None:
		print "Not feasible"
		return False

	print "soln"
	print soln

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
		constVi = -(g_fwd+g_cc)

		if Vlowi <= 0 and Vhighi <=0:
			constVi = -constVi
		if a <= 0 and VlowFwd >= 0 and VhighFwd >= 0:
			constFwd = -constFwd
		elif a >= 0 and VlowFwd <= 0 and VhighFwd <= 0:
			constFwd = -constFwd

		if a <= 0 and VlowCc >= 0 and VhighCc >= 0:
			constCc = -constCc
		elif a >= 0 and VlowCc <= 0 and VhighCc <= 0:
			constCc = -constCc

		if Vlowi is not None and VlowFwd is not None and VlowCc is not None:
			finalConstraint = str(constFwd)+" "+VoutFwd[i]\
							+ " + "+str(constCc)+" "+VoutCc[i]+" + "+str(constVi)+" "+V[i]+" == 0\n"
			constraint += finalConstraint

	return constraint

#given the appropriate feasible interval index for each decision variable
#find initial hyperrectangles
def createInitialHyperRectangles(bounds,a,g_cc,g_fwd,intervalIndices,debug):
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
	
	hypers = np.zeros((2,lenV))
	for i in range(len(V)):
		variable = "v"+str(i)
		objConstant = 1
		boundInd = intervalIndices[i]
		
		if boundInd == -1:
			objConstant = -1
		elif boundInd is not None and boundInd < len(bounds):
			Vlow = bounds[boundInd][0][0]
			Vhigh = bounds[boundInd][1][0]
			if Vlow <=0 and Vhigh <= 0:
				objConstant = -1

		minObjConstraint = "min "+str(objConstant)+" "+variable+"\n"
		minConstraint = minObjConstraint +constraint + decVariableConstraint
		#print "constraint"
		#print constraint
		minMat = normalize(minConstraint)
		minMat, minSoln = generalizedSimplex(minMat)
		if debug:
			print "minConstraint"
			print minConstraint
			#print "minMat"
			#print minMat
			print "minSoln"
			print minSoln
			#solFeasible = sillySyntax(minMat,-0.3024)
			#print "specific solution is feasible: ", solFeasible

		maxObjConstraint = "max "+str(objConstant)+" "+variable+"\n"
		maxConstraint = maxObjConstraint + constraint + decVariableConstraint
		#print "constraint"
		#print constrain
		maxMat = normalize(maxConstraint)
		if debug:
			print "maxConstraint"
			print maxConstraint
			#print "maxMat before simplex"
			#print maxMat
			#print maxMat[:,maxMat.shape[1]-1]
			#print ""
		maxMat, maxSoln = generalizedSimplex(maxMat)
		if debug:
			#print "maxMat"
			#print maxMat
			#print maxMat[:,maxMat.shape[1]-1]
			print "maxSoln"
			print maxSoln

		#minSoln = np.zeros((12))

		if minSoln is not None and maxSoln is not None:
			if objConstant == 1:
				hypers[0][i] = minSoln[i*(lenV-1)+1]
			elif objConstant == -1:
				hypers[0][i] = -minSoln[i*(lenV-1)+1]

			if objConstant == 1:
				hypers[1][i] = maxSoln[i*(lenV-1)+1]
			elif objConstant == -1:
				hypers[1][i] = -maxSoln[i*(lenV-1)+1]
		else:
			return None
	if debug:
		print "hyper found "
		print hypers
	return hypers

			
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
			if np.array_equal(hyperi,hyperj):
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
		#print "C ", C
		#print "IMidPoint ", IMidPoint
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
		elif np.array_equal(intersect,startBounds):
			print "Can't refute existence of solution"
			return (False,intersect)
		else:
			startBounds = intersect
		iteration += 1

def findUniqueHypers():
	a = 1.0
	params = [0.3,0.1]
	bounds = [[[-0.5],[0.0]],[[0.0],[0.5]]]
	funConstraints = fun1Constraints
	hypers = fun1Constraints(bounds,a,params,False)
	funNum = fun1Num
	funDer = fun1Der
	funDerInterval = fun1DerInterval
	'''a = -5.0
	params = [0.0]
	bounds = [[[-0.5,-0.5],[0.0,0.0]],[[0.0,0.0],[0.5,0.5]]]
	funConstraints = fun2Constraints
	funNum = fun2Num
	funDer = fun2Der
	funDerInterval = fun2DerInterval'''
	
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
	params = [g_fwd,g_cc]

	feasible = ifOrderingFeasibleOscl(bounds,a,g_cc,g_fwd,[1,-1,1,0])

	'''rootCombinationNodes = combinationWithTrees(4,[-1,2])
	feasibleIntervalIndices = []
	for i in range(len(rootCombinationNodes)):
		getFeasibleIntervalIndices(rootCombinationNodes[i],a,g_cc,g_fwd,bounds,feasibleIntervalIndices)
	print "validIntervalIndices ", feasibleIntervalIndices'''
	'''hypers = []
	for i in range(len(feasibleIntervalIndices)):
		intervalIndices = feasibleIntervalIndices[i]
		print "considering interval index: ", intervalIndices
		debug = False
		hyper = createInitialHyperRectangles(bounds,a,g_cc,g_fwd,intervalIndices,debug)
		#print "hyper "
		#print hyper
		if hyper is not None:
			hypers.append(hyper)
	hypers = removeRedundantHypers(hypers)
	print "final hyperrectangles"
	for i in range(len(hypers)):
		print "hyper number ", i
		print hypers[i]

	finalHypers = []
	while len(hypers)!=0:
		tempHypers = []
		for i in range(1,2):
			print "Checking existience within hyperrectangle ", i
			#checkExistenceOfSolution(a,params,hypers[i],funNum,funDer,funDerInterval)
			uniqueness,interval = checkExistenceOfSolution(a,params,hypers[i],oscNum,getJacobian,getJacobianInterval)
			if uniqueness:
				print "hyperrectangle contains unique solution"
				finalHypers.append(hypers[i])
			else:
				if interval is not None:
					print "need to refine more"
					midHyper = (hypers[i][0]+hypers[i][1])/2.0
					bounds = [[hypers[i][0],midHyper],[midHyper,hypers[i][1]]]
					print "new bounds "
					print bounds
					rootCombinationNodes = combinationWithTrees(4,[0,1])
					feasibleIntervalIndices = []
					for j in range(len(rootCombinationNodes)):
						getFeasibleIntervalIndices(rootCombinationNodes[j],a,g_cc,g_fwd,bounds,feasibleIntervalIndices)
					print "refine:validIntervalIndices ", feasibleIntervalIndices
					feasible = ifOrderingFeasibleOscl(bounds,a,g_cc,g_fwd,[0,1,0,1])
					for j in range(len(feasibleIntervalIndices)):
						intervalIndices = feasibleIntervalIndices[j]
						print "refine:considering interval index: ", intervalIndices
						debug = False
						hyper = createInitialHyperRectangles(bounds,a,g_cc,g_fwd,intervalIndices,debug)
						#print "hyper "
						#print hyper
						if hyper is not None:
							print "new hyperrectangle ", hyper
							#tempHypers.append(hyper)
					#hyper = funConstraintsScale(hypers[i],a,params,True)
					#tempHypers.append(hyper)
				else:
					print "no solution in hyperrectangle"
			print ""
		hypers = tempHypers'''

def simplexTest():
	mat = np.array([[1.0,5.0,35.0,20.0,0.0,0.0,0.0],[0.0,1.0,-1.0,-1.0,1.0,0.0,-2.0],[0.0,-1.0,-3.0,0.0,0.0,1.0,-3.0]])
	finalMat,soln = dualSimplex(mat)
	print "finalMat "
	print finalMat
	print "soln"
	print soln

if __name__=="__main__":
	osclTest()

# normalize LP - for min, should be greater than equal to. for max should be less than equal to
#first do simple simplex and then do dual simplex

