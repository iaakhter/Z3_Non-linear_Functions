# Itrat Ahmed Akhter
# CPSC 538G
# Final Project
# z3lib.py

'''
A library of functions that model the Rambus mobius ring oscillator in
Z3 so that Z3 can find the DC quilibrium points.
'''

from z3 import *
from numpy import *
import copy
import matplotlib.pyplot as plt
from mpmath import mp, iv

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

'''
A function that returns the absolute value of x.
'''
def myAbs(x):
	return If(x >= 0,x,-x)

'''
takes in non-symbolic python values
calculates the tanhFun of val
'''
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

def fun1Num(x,a,params):
	return tanhFun(a,x) - params[0]*x - params[1]

def fun2Num(x,a,params):
	Iy = tanhFun(a,x[0]) + params[0] - x[1]
	Ix = tanhFun(a,x[1]) - params[0] - x[0]
	return array([Ix,Iy])

def fun1Der(x,a,params):
	return array([tanhFunder(a,x) - params[0]])

def fun2Der(x,a,params):
	der = -1*ones((len(x),len(x)))
	der[0][1] = tanhFunder(a,x[1])
	der[1][0] = tanhFunder(a,x[0])
	return der

'''
params = [b1,b2]
'''
def fun1DerInterval(a,params,bounds):
	lowerBound = bounds[:,0]
	upperBound = bounds[:,1]
	der1 = fun1Der(lowerBound[0],a,params)
	der2 = fun1Der(upperBound[0],a,params)
	der = zeros((1,1,2))
	der[:,:,0] = minimum(der1,der2)
	der[:,:,1] = maximum(der1,der2)
	return der

'''
params = b
'''
def fun2DerInterval(a,params,bounds):
	lowerBound = bounds[:,0]
	upperBound = bounds[:,1]
	der1 = fun2Der(lowerBound,a,params)
	der2 = fun2Der(upperBound,a,params)
	der = zeros((len(lowerBound),len(lowerBound),2))
	der[:,:,0] = minimum(der1,der2)
	der[:,:,1] = maximum(der1,der2)
	return der

def triangleBounds(a, Vin, Vout, Vlow, Vhigh):
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
			return Implies(And(Vin >= Vlow, Vin <= Vhigh),
							And(Vout >= dThird*Vin + cThird,
								Vout <= dLow*Vin + cLow,
								Vout <= dHigh*Vin + cHigh))

		elif Vlow <=0 and Vhigh <=0:
			return Implies(And(Vin >= Vlow, Vin <= Vhigh),
							And(Vout <= dThird*Vin + cThird,
								Vout >= dLow*Vin + cLow,
								Vout >= dHigh*Vin + cHigh))

	elif a < 0:
		if Vlow <= 0 and Vhigh <=0:
			return Implies(And(Vin >= Vlow, Vin <= Vhigh),
							And(Vout >= dThird*Vin + cThird,
								Vout <= dLow*Vin + cLow,
								Vout <= dHigh*Vin + cHigh))

		elif Vlow >=0 and Vhigh >=0:
			return Implies(And(Vin >= Vlow, Vin <= Vhigh),
							And(Vout <= dThird*Vin + cThird,
								Vout >= dLow*Vin + cLow,
								Vout >= dHigh*Vin + cHigh))

def trapezoidBounds(a,Vin,Vout, Vlow, Vhigh):
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
			return Implies(Vin < Vlow, 
							And(Vout > dLow*Vin + cLow,
								Vout >= -1,
								Vout < tanhFunVlow))
		elif Vlow >= 0 and Vhigh >=0:
			return Implies(Vin > Vhigh, 
							And(Vout < dHigh*Vin + cHigh,
								Vout <=  1,
								Vout >  tanhFunVhigh))

	elif a < 0:
		if Vlow <= 0 and Vhigh <= 0:
			return Implies(Vin < Vlow, 
							And(Vout < dLow*Vin + cLow,
								Vout <= 1,
								Vout > tanhFunVlow))
		elif Vlow >= 0 and Vhigh >=0:
			return Implies(Vin > Vhigh, 
							And(Vout > dHigh*Vin + cHigh,
								Vout >= - 1,
								Vout <  tanhFunVhigh))


#params in the form of [p1,p2]
def fun1(s,x,bounds,a,params):
	outVals = RealVector('outVals',len(x))
	allBounds = []
	for i in range(len(x)):
		allBounds.append(bounds)

	for i in range(len(x)):	
		for j in range(len(bounds)):
			bound = allBounds[i][j]
			triangleClaim = triangleBounds(a,x[i],outVals[i],bound[0][i], bound[1][i])
			s.add(triangleClaim)
			if j == 0 or j == len(bounds)-1:
				trapezoidClaim = trapezoidBounds(a,x[i],outVals[i],bound[0][i],bound[1][i])
				s.add(trapezoidClaim)
		s.add(outVals[i] == params[0]*x[i] + params[1])

def fun2(s,x,bounds,a,params):
	outVals = RealVector('outVals',len(x))
	allBounds = []
	for i in range(len(x)):
		allBounds.append(bounds)

	for i in range(len(x)):
		for j in range(len(bounds)):
			bound = allBounds[i][j]
			triangleClaim = triangleBounds(a,x[i],outVals[i],bound[0][i], bound[1][i])
			s.add(triangleClaim)
			if j == 0 or j == len(bounds)-1:
				trapezoidClaim = trapezoidBounds(a,x[i],outVals[i],bound[0][i],bound[1][i])
				s.add(trapezoidClaim)

	s.add(outVals[0] + params[0] - x[1] == 0)
	s.add(outVals[1] - params[0] - x[0] == 0)

def plotFun1(a,params):
	x = arange(-4.0,4.0,0.01)
	ytanh = tanhFun(a,x)
	yLinear = params[0]*x + params[1]
	plt.figure()
	plt.plot(x,ytanh,'r',x,yLinear,'b')
	#plt.plot(x,ytanh-yLinear)
	plt.show()

#params in the form of [b]
def plotFun2(a,params):
	u = arange(-4.0,4.0,0.01)
	plt.figure()
	plt.plot(u,tanhFun(a,u) + params[0],'r',tanhFun(a,u) - params[0],u,'b')
	plt.legend(['I_y == 0','I_x == 0'])
	plt.show()


def findScale(x,bounds,a,params,fun):
	print "Finding Scale"
	opt = Optimize()
	fun(opt,x,bounds,a,params)
	optBounds = zeros((2,len(x)))
	for i in range(len(x)):
		print "i: ", i
		opt.push()
		optMin = opt.minimize(x[0])
		opt.check()
		m = opt.model()
		lowerString = str(opt.lower(optMin))
		indexStar = lowerString
		minVal = float(Fraction(str(opt.lower(optMin))))
		opt.pop()
		opt.push()
		optMax = opt.maximize(x[0])
		opt.check()
		maxVal = float(Fraction(str(opt.upper(optMax))))
		opt.pop()
		optBounds[0][i] = minVal
		optBounds[1][i] = maxVal
	print "lowerBounds ", optBounds[0]
	print "upperbounds ", optBounds[1]
	print ""
	return optBounds
	

# s is solver. I and V are an array of Z3 symbolic  values.
# VlowVhighs contain indicate how triangle bounds are created
# This function finds hyper rectangles containing DC equilibrium points 
# for our oscillator model. Each hyper rectangle has length and width
# equalling distance
def findHyper(x,bounds,a,params,distances,fun,funNum):
	allHyperRectangles = []
	s = Solver()
	print "Finding HyperRectangles"
	count = 0
	while(True):
		#simulate the oscillator and check for solution
		print "count: ", count
		count += 1
		s.push()
		fun(s,x,bounds,a,params)
		#print "solver "
		#print s
		ch = s.check()
		if(ch == sat):
			low = zeros((len(x)))
			high = zeros((len(x)))
			sol = zeros((len(x)))
			m = s.model()
			print "m "
			print m
			for d in m.decls():
				dName = str(d.name())
				index = int(dName[len(dName) - 1])
				val = float(Fraction(str(m[d])))
				if(dName[0]=="x" and dName[1]=="_"):
					sol[index] = val
					low[index] = val-distances[index]
					high[index] = val+distances[index]
					if(low[index] < 0 and high[index]>0):
						if(val >= 0):
							low[index] = 0.0
						elif(val < 0):
							high[index] = 0.0

			print "sol: ", sol
			print "Check solution "
			yNum = funNum(sol,a,params)
			print "yNum should be close to 0"
			print yNum
			
			#create hyperrectangle around the solution formed 
			newHyperRectangle = [low,high]

		else:
			newHyperRectangle = None
			if(ch == unsat):
				print "no more solutions"
				print "allHyperRectangles = " + str(allHyperRectangles)
			elif(ch == unknown):
				print "solver failed"
			else:
				print "INTERNAL ERROR -- unrecognized return code"

		s.pop()
		if newHyperRectangle is None:
			return allHyperRectangles
		else:
			print "newHyperRectangle"
			print newHyperRectangle
			allHyperRectangles.append(newHyperRectangle)
			# Add the constraint so that Z3 can find solutions outside the hyper rectangle just constructed
			s.add(Or(*[Or(x[i] < newHyperRectangle[0][i] - distances[i], 
							x[i] > newHyperRectangle[1][i] + distances[i]) for i in range(len(x))]))

'''def findSolWithNewtons(a,g_fwd,g_cc,hyperRectangle):
	print "lower bounds ", hyperRectangle[0]
	print "upper bounds ",hyperRectangle[1]
	sol = [(hyperRectangle[1][i] + hyperRectangle[0][i])/2.0 for i in range(len(hyperRectangle[0]))]
	print "starting solution ", sol
	Inum = array(oscNum(sol,a,g_cc,g_fwd))
	oldInum = copy.deepcopy(Inum)
	iter = 0
	while(all(abs(Inum) > 1.0e-6)):
		jac = getJacobian(sol,a,g_cc,g_fwd)
		sol = sol - linalg.solve(jac,Inum)
		oldInum = copy.deepcopy(Inum)
		Inum = array(oscNum(sol,a,g_cc,g_fwd))
		print "jac ", jac
		print "sol ", sol
		print "Inum ", Inum
		print ""
		iter+=1
		if iter>7:
			sol = None
			break
	return sol'''

'''def checkExistenceOfSolution(a,g_fwd,g_cc,hyperRectangle):
	print "lower bounds ", hyperRectangle[0]
	print "upper bounds ",hyperRectangle[1]

	#TODO need to take care/check if it matters that the jacobian has
	#zero elements
	jac = getJacobianInterval(hyperRectangle,a,g_cc,g_fwd)
	jacIv = iv.matrix(len(hyperRectangle[0]),len(hyperRectangle[1]))
	for i in range(len(hyperRectangle[0])):
		for j in range(len(hyperRectangle[1])):
			jacIv[i,j] = iv.mpf([jac[i,j,0],jac[i,j,1]])


	#inverseJac = getInverseOfIntervalMatrix(jac,0.1)
	#print "inverseJac"
	#print inverseJac

	print "jacIv"
	print jacIv
	midPoint = array([(hyperRectangle[0][i]+hyperRectangle[1][i])/2.0 for i in range(len(hyperRectangle[0]))])
	_,_,IMidPoint = array(oscNum(midPoint,a,g_cc,g_fwd))
	#nInterval1 = midPoint - dot(inverseJac[:,:,0],IMidPoint)
	#nInterval2 = midPoint - dot(inverseJac[:,:,1],IMidPoint)

	nInterval1 = midPoint - iv.lu_solve(jacIv[:,:,0],IMidPoint)
	nInterval2 = midPoint - iv.lu_solve(jacIv[:,:,1],IMidPoint)
	newtonInterval = zeros((len(nInterval1),2))
	for i in range(len(nInterval1)):
		newtonInterval[i][0] = min(nInterval1[i],nInterval2[i])
		newtonInterval[i][1] = max(nInterval1[i],nInterval2[i])

	print "newtonLowerBounds ", newtonInterval[:,0]
	print "newtonUpperBounds ", newtonInterval[:,1]

	for i in range(len(hyperRectangle[0])):
		if newtonInterval[i][0] < hyperRectangle[0][i] or newtonInterval[i][0] > hyperRectangle[1][i]:
			return False
		if newtonInterval[i][1] < hyperRectangle[0][i] or newtonInterval[i][1] > hyperRectangle[1][i]:
			return False
	return True'''

def multiplyRegularMatWithIntervalMat(regMat,intervalMat):
	mat1 = dot(regMat,intervalMat[:,:,0])
	mat2 = dot(regMat,intervalMat[:,:,1])
	result = zeros((regMat.shape[0],regMat.shape[1],2))
	result[:,:,0] = minimum(mat1,mat2)
	result[:,:,1] = maximum(mat1,mat2)
	return result

def subtractIntervalMatFromRegularMat(regMat,intervalMat):
	mat1 = regMat - intervalMat[:,:,0]
	mat2 = regMat - intervalMat[:,:,1]
	result = zeros((regMat.shape[0],regMat.shape[1],2))
	result[:,:,0] = minimum(mat1,mat2)
	result[:,:,1] = maximum(mat1,mat2)
	return result

def multiplyIntervalMatWithIntervalVec(mat,vec):
	mat1 = dot(mat[:,:,0],vec[:,0])
	mat2 = dot(mat[:,:,1],vec[:,0])
	mat3 = dot(mat[:,:,0],vec[:,1])
	mat4 = dot(mat[:,:,1],vec[:,1])
	result = zeros((mat.shape[0],vec.shape[1]))
	result[:,0] = minimum(minimum(mat1,mat2),minimum(mat3,mat4))
	result[:,1] = maximum(maximum(mat1,mat2),maximum(mat3,mat4))
	return result


def checkExistenceOfSolution(a,params,hyperRectangle,funNum,funDer,funDerInterval):
	print "lower bounds ", hyperRectangle[0]
	print "upper bounds ",hyperRectangle[1]
	numVolts = len(hyperRectangle[0])

	startBounds = zeros((numVolts,2))
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
		C = linalg.inv(jacMidPoint)
		I = identity(numVolts)

		jacInterval = funDerInterval(a,params,startBounds)
		C_IMidPoint = dot(C,IMidPoint)

		C_jacInterval = multiplyRegularMatWithIntervalMat(C,jacInterval)
		I_minus_C_jacInterval = subtractIntervalMatFromRegularMat(I,C_jacInterval)
		xi_minus_midPoint = zeros((numVolts,2))
		for i in range(numVolts):
			xi_minus_midPoint[i][0] = startBounds[i][0] - midPoint[i]
			xi_minus_midPoint[i][1] = startBounds[i][1] - midPoint[i]

		lastTerm = multiplyIntervalMatWithIntervalVec(I_minus_C_jacInterval, xi_minus_midPoint)
		
		kInterval1 = midPoint - C_IMidPoint + lastTerm[:,0]
		kInterval2 = midPoint - C_IMidPoint + lastTerm[:,1]
		kInterval = zeros((numVolts,2))
		kInterval[:,0] = minimum(kInterval1, kInterval2)
		kInterval[:,1] = maximum(kInterval1, kInterval2)

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
			print startBounds
			return startBounds

		intersect = zeros((numVolts,2))
		for i in range(numVolts):
			minVal = max(kInterval[i][0],startBounds[i][0])
			maxVal = min(kInterval[i][1],startBounds[i][1])
			if minVal <= maxVal:
				intersect[i] = [minVal,maxVal]
				intervalLength =  intersect[:,1] - intersect[:,0]
			else:
				intersect = None

		print "intersect"
		print intersect

		if intersect is None:
			print "hyperrectangle does not contain any solution"
			return None
		elif linalg.norm(intervalLength) < 1e-8:
			print "Found smallest hyperrectangle containing solution"
			return intersect
		else:
			startBounds = intersect
		'''elif linalg.norm(intersect-startBounds) < 1e-8:
			print "Found the smallest possible hyperrectangle containing solution"
			return intersect
		if iteration >= 3:
			return'''
		iteration += 1


def testInvRegion():
	'''x = RealVector('x',1)
	a = 1
	params = [0.3,0.1]
	bounds = [[[-4.0],[-3.0]],
				  [[-3.0],[-1.0]],
				  [[-1.0],[0.0]],
				  [[0.0],[1.0]],
				  [[1.0],[3.0]],
				  [[3.0],[4.0]]]
	fun = fun1
	funNum = fun1Num
	funDer = fun1Der
	funDerInterval = fun1DerInterval'''

	x = RealVector('x',2)
	a = -5
	params = [0.0]
	bounds = [[[-4.0,-4.0],[-3.0,-3.0]],
				  [[-3.0,-3.0],[-1.0,-1.0]],
				  [[-1.0,-1.0],[0.0,0.0]],
				  [[0.0,0.0],[1.0,1.0]],
				  [[1.0,1.0],[3.0,3.0]],
				  [[3.0,3.0],[4.0,4.0]]]
	fun = fun2
	funNum = fun2Num
	funDer = fun2Der
	funDerInterval = fun2DerInterval

	overallHyperRectangle = findScale(x,bounds,a,params,fun)
	minOptSol = overallHyperRectangle[0]
	maxOptSol = overallHyperRectangle[1]
	bounds = [[minOptSol,minOptSol/2.0],
				  [minOptSol/2.0,zeros((len(x)))],
				  [zeros((len(x))),maxOptSol/2.0],
				  [maxOptSol/2.0,maxOptSol]]


	distances = (maxOptSol - minOptSol)/8.0
	
	allHyperRectangles = findHyper(x,bounds,a,params,distances,fun,funNum)
	
	print "total number of hyperrectangles: ", len(allHyperRectangles)
	print ""
	
	print "Checking existence of solutions within hyperrectangles"
	for i in range(len(allHyperRectangles)):
		print "Checking existience within hyperrectangle ", i
		checkExistenceOfSolution(a,params,allHyperRectangles[i],funNum,funDer,funDerInterval)
		print ""

	#plotFun1(a,params)
	#plotFun2(a,params)

testInvRegion()
