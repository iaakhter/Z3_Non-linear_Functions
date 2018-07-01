import math
import numpy as np
from scipy.spatial import ConvexHull
from intervalBasics import *

# assume that const > 0 in all cases


#sin(const*x)
def sinFun(x,const):
	funVal = np.sin(const*x)

	if interval_p(x):
		lowVal = math.floor(x[0]/(const*math.pi))
		highVal = math.floor(x[1]/(const*math.pi))
		if highVal - lowVal > 2.0:
			return np.array([-1.0, 1.0])

		if int(lowVal) %2 == 0:
			return np.array([min(funVal[0], funVal[1]), 1.0])
		else:
			return np.array([-1.0, max(funVal[0], funVal[1])])
	return funVal
	

def sinFunder(x,const):
	return interval_mult(const,cosFun(x,const))



#cos(const*x)
def cosFun(x,const):
	funVal = np.cos(const*x)

	if interval_p(x):
		shiftedToSin = interval_sub(x,math.pi/2.0)
		lowVal = math.floor(shiftedToSin[0]/(const*math.pi))
		highVal = math.floor(shiftedToSin[1]/(const*math.pi))
		if highVal - lowVal > 2.0:
			return np.array([-1.0, 1.0])

		if int(lowVal) %2 == 0:
			return np.array([min(funVal[0], funVal[1]), 1.0])
		else:
			return np.array([-1.0, max(funVal[0], funVal[1])])
	return funVal
	

def cosFunder(x,const):
	return interval_mult(const,interval_neg(sinFun(x,const)))


def tanhFun(x, const):
	tanhVal = np.tanh(const*x)
	#tanhVal = (np.exp(const*x) - np.exp(-const*x))/(np.exp(const*x) + np.exp(-const*x))
	if interval_p(x):
		return np.array([min(tanhVal[0], tanhVal[1]), max(tanhVal[0], tanhVal[1])])
	return tanhVal
	#return (exp(a*val) - exp(-a*val))/(exp(a*val) + exp(-a*val))


def tanhFunder(x, const):
	den = np.cosh(const*x)*np.cosh(const*x)
	grad = np.divide(const,den)
	if interval_p(x):
		if x[0]*x[1] >= 0:
			grad = np.array([min(grad[0], grad[1]), max(grad[0], grad[1])])
		else:
			den0 = np.cosh(0.0)*np.cosh(0.0)
			grad0 = np.divide(const,den0)
			grad = np.array([min(grad[0], grad[1], grad0), max(grad[0], grad[1], grad0)])

	return grad

#1/(const*x)
def invFun(x, const):
	if any([xVal == 0 for xVal in x]):
		raise Exception('Invalid argument for invFun ' + str(x))
	funVal = np.divide(1, (const*x))
	if interval_p(x):
		if x[0]*x[1] > 0.0:
			return np.array([min(funVal[0], funVal[1]), max(funVal[0], funVal[1])])
		else:
			return np.array([-float("inf"), float("inf")])
	return funVal
	

def invFunder(x,const):
	if any([xVal == 0 for xVal in x]):
		raise Exception('Invalid argument for invFun ' + str(x))
	der = -np.divide(const,(const*const*x*x))
	if interval_p(x):
		if x[0]*x[1] > 0.0:
			return np.array([min(der[0], der[1]), max(der[0], der[1])])
		else:
			return np.array([-float("inf"), max(der[0], der[1])])
	return der


#arcsin(const*x)
def arcsinFun(x, const):
	if any([xVal < -1 or xVal > 1 for xVal in x]):
		raise Exception('Invalid argument for arcsin ' + str(x))
	fun = np.arcsin(const*x)
	if interval_p(x):
		return np.array([min(fun[0], fun[1]), max(fun[0], fun[1])])
	return fun


def arcsinFunder(x, const):
	if any([xVal < -1 or xVal > 1 for xVal in x]):
		raise Exception('Invalid argument for arcsin ' + str(x))
	grad = np.divide(const,(np.sqrt(1 - const*x*const*x)))
	if interval_p(x):
		if x[0]*x[1] >= 0:
			grad = np.array([min(grad[0], grad[1]), max(grad[0], grad[1])])
		else:
			grad0 = np.divide(const,(1.0))
			grad = np.array([min(grad[0], grad[1], grad0), max(grad[0], grad[1], grad0)])

	return grad


#arctan(const*x)
def arctanFun(x, const):
	Ifun = math.atan(const*x)
	der = const/(1.0 + const*x*const*x)
	return [Ifun, der]


#linear constraints for sin(constant*x)
def sinLinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	#print ("inputLow", inputLow, "inputHigh", inputHigh)
	inputLowPi = math.ceil(inputLow/(math.pi/constant))
	inputHighPi = math.ceil(inputHigh/(math.pi/constant))
	#print ("inputLowPi", inputLowPi, "inputHighPi", inputHighPi)

	if inputLowPi == inputHighPi:
		if inputLowPi%2 == 0:
			#print ("coming here?")
			return triangleBounds(sinFun, sinFunder, constant, inputVar, outputVar, inputLow, inputHigh, "pos")
		else:
			return triangleBounds(sinFun, sinFunder, constant, inputVar, outputVar, inputLow, inputHigh, "neg")

	overallConstraint = "1 " + inputVar + " >= " + str(inputLow) + "\n"
	overallConstraint += "1 " + inputVar + " <= " + str(inputHigh) + "\n"
	allTrianglePoints = []
	inputStart = inputLow
	piInput = inputStart/(math.pi/constant)
	inputEnd = math.ceil(piInput)*(math.pi/constant)
	while(inputStart < inputHigh):
		if inputEnd > inputHigh:
			inputEnd = inputHigh
		#print ("inputStart", inputStart, "inputEnd", inputEnd)
		tPts = trianglePoints(sinFun, sinFunder, inputStart, inputEnd, constant)
		allTrianglePoints += tPts
		inputStart = inputEnd
		inputEnd += math.pi/constant

	allTrianglePoints = np.array(allTrianglePoints)
	try:
		cHullConstraints = convexHullConstraints2D(allTrianglePoints, inputVar, outputVar)
		overallConstraint += cHullConstraints
	except:
		pass
	
	return overallConstraint

#linear constraints for sin(constant*x)
def cosLinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	#print ("inputLow", inputLow, "inputHigh", inputHigh)
	inputLowPi = math.ceil((inputLow - math.pi/2.0)/(math.pi/constant))
	inputHighPi = math.ceil((inputHigh - math.pi/2.0)/(math.pi/constant))
	#print ("inputLowPi", inputLowPi, "inputHighPi", inputHighPi)

	if inputLowPi == inputHighPi:
		if inputLowPi%2 == 0:
			#print ("coming here?")
			return triangleBounds(cosFun, cosFunder, constant, inputVar, outputVar, inputLow, inputHigh, "pos")
		else:
			return triangleBounds(cosFun, cosFunder, constant, inputVar, outputVar, inputLow, inputHigh, "neg")

	overallConstraint = "1 " + inputVar + " >= " + str(inputLow) + "\n"
	overallConstraint += "1 " + inputVar + " <= " + str(inputHigh) + "\n"
	allTrianglePoints = []
	inputStart = inputLow
	piInput = inputStart/(math.pi/constant)
	inputEnd = math.ceil(piInput)*(math.pi/constant)
	while(inputStart < inputHigh):
		if inputEnd > inputHigh:
			inputEnd = inputHigh
		#print ("inputStart", inputStart, "inputEnd", inputEnd)
		tPts = trianglePoints(cosFun, cosFunder, inputStart, inputEnd, constant)
		allTrianglePoints += tPts
		inputStart = inputEnd
		inputEnd += math.pi/constant

	allTrianglePoints = np.array(allTrianglePoints)
	try:
		cHullConstraints = convexHullConstraints2D(allTrianglePoints, inputVar, outputVar)
		overallConstraint += cHullConstraints
	except:
		pass
	
	return overallConstraint



#linear constraints for 1/(constant*x)
def inverseLinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	if inputLow == 0.0 or inputHigh == 0.0:
		raise  Exception("invalid lowBound or highBound for exponential" + str(inputLow) + " " + str(inputHigh))
	if inputLow > 0.0 and inputHigh > 0:
		return triangleBounds(invFun, invFunder, constant, inputVar, outputVar, inputLow, inputHigh, "pos")
	elif inputLow < 0.0 and inputHigh < 0.0:
		return triangleBounds(invFun, invFunder, constant, inputVar, outputVar, inputLow, inputHigh, "neg")
	elif inputLow < 0.0 and inputHigh > 0.0:
		return triangleBounds(invFun, invFunder, constant, inputVar, outputVar, inputLow, inputHigh, None)

#linear constraints for arcsin(constant*x)
def arcsinLinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	if inputLow < -1.0 or inputLow > 1.0 or inputHigh < -1.0 or inputHigh > 1.0:
		raise Exception("invalid lowBound or highBound for arcsin" + str(inputLow) + " " + str(inputHigh))
		return None

	if inputLow >= 0.0 and inputHigh >= 0.0:
		return triangleBounds(arcsinFun, arcsinFunder, constant, inputVar, outputVar, inputLow, inputHigh, "pos")

	elif inputLow <= 0.0 and inputHigh <= 0.0:
		return triangleBounds(arcsinFun, arcsinFunder, constant, inputVar, outputVar, inputLow, inputHigh, "neg")

	overallConstraint = "1 " + inputVar + " >= " + str(inputLow) + "\n"
	overallConstraint += "1 " + inputVar + " <= " + str(inputHigh) + "\n"
	allTrianglePoints = []
	allTrianglePoints += trianglePoints(arcsinFun, arcsinFunder, inputLow, 0.0, constant)
	allTrianglePoints += trianglePoints(arcsinFun, arcsinFunder, 0.0, inputHigh, constant)
	allTrianglePoints = np.array(allTrianglePoints)
	#print ("inputLow", inputLow, "inputHigh", inputHigh)
	#print ("allTrianglePoints")
	#print (allTrianglePoints)
	return overallConstraint + convexHullConstraints2D(allTrianglePoints, inputVar, outputVar)

#linear constraints for tanh
def tanhLinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	if constant < 0:
		if inputLow <= 0.0 and inputHigh <= 0.0:
			return triangleBounds(tanhFun, tanhFunder, constant, inputVar, outputVar, inputLow, inputHigh, "neg")
		if inputLow >= 0.0 and inputHigh >= 0.0:
			return triangleBounds(tanhFun, tanhFunder, constant, inputVar, outputVar, inputLow, inputHigh, "pos")
	elif constant >= 0:
		if inputLow <= 0.0 and inputHigh <= 0.0:
			return triangleBounds(tanhFun, tanhFunder, constant, inputVar, outputVar, inputLow, inputHigh, "pos")
		if inputLow >= 0.0 and inputHigh >= 0.0:
			return triangleBounds(tanhFun, tanhFunder, constant, inputVar, outputVar, inputLow, inputHigh, "neg")

	overallConstraint = "1 " + inputVar + " >= " + str(inputLow) + "\n"
	overallConstraint += "1 " + inputVar + " <= " + str(inputHigh) + "\n"
	overallConstraint += "1 " + outputVar + " <= 1.0\n"
	overallConstraint += "1 " + outputVar + " >= -1.0\n"
	allTrianglePoints = []
	allTrianglePoints += trianglePoints(tanhFun, tanhFunder, inputLow, 0.0, constant)
	allTrianglePoints += trianglePoints(tanhFun, tanhFunder, 0.0, inputHigh, constant)
	allTrianglePoints = np.array(allTrianglePoints)
	try:
		cHullConstraints = convexHullConstraints2D(allTrianglePoints, inputVar, outputVar)
		overallConstraint += cHullConstraints
	except:
		pass
	return overallConstraint


#points of triangle formed by trianglebounds for a function
def trianglePoints(function, functionDer, inputLow, inputHigh, constant):
	if inputLow > inputHigh:
		return []
	funLow = function(inputLow, constant)
	dLow = functionDer(inputLow, constant)
	funHigh = function(inputHigh, constant)
	dHigh = functionDer(inputHigh, constant)
	
	cLow = funLow - dLow*inputLow
	cHigh = funHigh - dHigh*inputHigh

	diff = inputHigh - inputLow
	if(diff == 0):
		diff = 1e-10
	dThird = (funHigh - funLow)/diff
	cThird = funLow - dThird*inputLow

	leftIntersectX, leftIntersectY = None, None
	if abs(dHigh - dLow) < 1e-8:
		leftIntersectX = inputLow
		leftIntersectY = funLow
	else:
		leftIntersectX = (cHigh - cLow)/(dLow - dHigh)
		leftIntersectY = dLow*leftIntersectX + cLow

	#print ("leftIntersectX", leftIntersectX, "leftIntersectY", leftIntersectY)
	tPts = [[inputLow, funLow],[inputHigh, funHigh],[leftIntersectX, leftIntersectY]]
	return tPts


def triangleBounds(function, functionDer, constant, inputVar, outputVar, inputLow, inputHigh, secDer):
	funLow = function(np.array([inputLow]), constant)[0]
	dLow = functionDer(np.array([inputLow]), constant)[0]
	funHigh = function(np.array([inputHigh]), constant)[0]
	dHigh = functionDer(np.array([inputHigh]), constant)[0]
	
	cLow = funLow - dLow*inputLow
	cHigh = funHigh - dHigh*inputHigh

	diff = inputHigh - inputLow
	if(diff == 0):
		diff = 1e-10
	dThird = (funHigh - funLow)/diff
	cThird = funLow - dThird*inputLow

	overallConstraint = ""
	overallConstraint += "1 " + inputVar + " >= " + str(inputLow) + "\n"
	overallConstraint += "1 " + inputVar + " <= " + str(inputHigh) + "\n"
	if secDer == None:
		return overallConstraint

	if secDer == "pos":
		return overallConstraint + "1 "+ outputVar + " + " +str(-dThird) + " " + inputVar + " <= "+str(cThird)+"\n" +\
				"1 "+outputVar + " + " +str(-dLow) + " " + inputVar + " >= "+str(cLow)+"\n" +\
				"1 "+outputVar + " + " +str(-dHigh) + " " + inputVar + " >= "+str(cHigh) + "\n"

	
	if secDer == "neg":
		return overallConstraint + "1 "+ outputVar + " + " +str(-dThird) + " " + inputVar + " >= "+str(cThird)+"\n" +\
				"1 "+outputVar + " + " +str(-dLow) + " " + inputVar + " <= "+str(cLow)+"\n" +\
				"1 "+outputVar + " + " +str(-dHigh) + " " + inputVar + " <= "+str(cHigh) + "\n"

def convexHullConstraints2D(points, inputVar, outputVar):
	#print ("points")
	#print (points)
	hull = ConvexHull(points)
	convexHullMiddle = np.zeros((2))
	numPoints = 0
	for simplex in hull.simplices:
		#print ("simplex", simplex)
		for ind in simplex:
			convexHullMiddle += [points[ind,0],points[ind,1]]
			numPoints += 1
	convexHullMiddle = convexHullMiddle/(numPoints*1.0)
	#print ("convexHullMiddle", convexHullMiddle)
	overallConstraint = ""
	for si in range(len(hull.simplices)):
		simplex = hull.simplices[si]
		#print ("simplex", simplex)

		pt1x = points[simplex[0],0]
		pt1y = points[simplex[0],1]

		pt2x = points[simplex[1],0]
		pt2y = points[simplex[1],1]

		#print ("pt1x ", pt1x, "pt1y", pt1y)
		#print ("pt2x ", pt2x, "pt2y", pt2y)

		grad = (pt2y - pt1y)/(pt2x - pt1x)
		c = pt1y - grad*pt1x
		#print ("grad", grad, "c", c)

		yMiddle = grad*convexHullMiddle[0] + c
		#print ("yMiddle", yMiddle)

		sign = " <= "
		if convexHullMiddle[1] > yMiddle:
			sign = " >= "

		#print ("sign", sign)

		overallConstraint += "1 " + outputVar + " + " + str(-grad) + " " + inputVar + \
			sign + str(c) + "\n"
		
	#print ("overallConstraint")
	#print (overallConstraint)
	return overallConstraint
