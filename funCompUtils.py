import math
import numpy as np
from scipy.spatial import ConvexHull
from intervalBasics import *

# assume that const > 0 in all cases


#sin(const*x)
def sinFun(x,const):
	Ifun = math.sin(const*x)
	der = const*math.cos(const*x)
	return [Ifun, der]

def sinFunInterval(x, const):
	#print ("x", x)
	lowVal = math.floor(x[0]/(const*math.pi))
	highVal = math.floor(x[1]/(const*math.pi))
	if highVal - lowVal > 2.0:
		return np.array([-1.0, 1.0])

	if int(lowVal) %2 == 0:
		return np.array([min(sinFun(x[0], const)[0], sinFun(x[1], const)[0]), 1.0])
	else:
		return np.array([-1.0, max(sinFun(x[0], const)[0], sinFun(x[1], const)[0])])

#cos(const*x)
def cosFun(x,const):
	Ifun = math.cos(const*x)
	der = -const*math.sin(const*x)
	return [Ifun, der]

def cosFunInterval(x, const):
	shiftedToSin = interval_sub(x,math.pi/2.0)
	lowVal = math.floor(shiftedToSin[0]/(const*math.pi))
	highVal = math.floor(shiftedToSin[1]/(const*math.pi))
	if highVal - lowVal > 2.0:
		return np.array([-1.0, 1.0])

	if int(lowVal) %2 == 0:
		return np.array([min(cosFun(x[0], const)[0], cosFun(x[1], const)[0]), 1.0])
	else:
		return np.array([-1.0, max(cosFun(x[0], const)[0], cosFun(x[1], const)[0])])
	return sinInterval

#tanhFun(const*x)
def tanhFun(x, const):
	Ifun = np.tanh(const*x)
	den = np.cosh(const*x)*np.cosh(const*x)
	der = const/den
	return [Ifun, der]

#1/(const*x)
def invFun(x, const):
	Ifun = 1/(const*x)
	der = -const/(const*const*x*x)
	return [Ifun, der]

def invFunInterval(x,const):
	#print ("x in invFunInterval", x)
	if x[0]*x[1] > 0.0:
		return np.array([min(invFun(x[0], const)[0], invFun(x[1], const)[0]), max(invFun(x[0], const)[0], invFun(x[1], const)[0])])
	else:
		return np.array([-float("inf"), float("inf")])


#(const*x)/(2 + const*x)
def invFun1(x,const):
	Ifun = (const*x)/(2 + const*x)
	der = (2*const + const*const*x - const*const*x)/(2 + const*x)**2
	return [Ifun, der]

# log(1 + const*x)
def logFun1(x,const):
	Ifun = math.log(1 + const*x)
	der = const/(1 + const*x)
	return [Ifun, der]

#arcsin(const*x)
def arcsinFun(x, const):
	Ifun = math.asin(const*x)
	der = const/(math.sqrt(1 - const*x*const*x))
	return [Ifun, der]

def arcsinFunInterval(x, const):
	arcsin1 = arcsinFun(x[0], const)
	arcsin2 = arcsinFun(x[1], const)
	return np.array([min(arcsin1, arcsin2)[0], max(arcsin1, arcsin2)[0]])

#arctan(const*x)
def arctanFun(x, const):
	Ifun = math.atan(const*x)
	der = const/(1.0 + const*x*const*x)
	return [Ifun, der]

#(const*x)^2
def xSquareFun(x, const):
	Ifun = const*x*const*x
	der = 2*const*const*x
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
			return triangleBounds(sinFun, constant, inputVar, outputVar, inputLow, inputHigh, "pos")
		else:
			return triangleBounds(sinFun, constant, inputVar, outputVar, inputLow, inputHigh, "neg")

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
		tPts = trianglePoints(sinFun, inputStart, inputEnd, constant)
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
			return triangleBounds(cosFun, constant, inputVar, outputVar, inputLow, inputHigh, "pos")
		else:
			return triangleBounds(cosFun, constant, inputVar, outputVar, inputLow, inputHigh, "neg")

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
		tPts = trianglePoints(cosFun, inputStart, inputEnd, constant)
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


#linear constraints for 1/(2 + constant*x)
def inverse1LinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	if inputLow == -2.0/constant or inputHigh == -2.0/constant:
		print ("invalid lowBound or highBound for exponential", inputLow, inputHigh)
		return None
	if inputLow > -2.0/constant and inputHigh > -2.0/constant:
		return triangleBounds(invFun, constant, inputVar, outputVar, inputLow, inputHigh, "neg")
	elif inputLow < -2.0/constant and inputHigh < -2.0/constant:
		return triangleBounds(invFun, constant, inputVar, outputVar, inputLow, inputHigh, "pos")
	elif inputLow < -2.0/constant and inputHigh > -2.0/constant:
		return triangleBounds(invFun, constant, inputVar, outputVar, inputLow, inputHigh, None)

#linear constraints for# log(1 + const*x)
def log1LinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	if inputLow == -1.0/constant or inputHigh == -1.0/constant:
		print ("invalid lowBound or highBound for exponential", inputLow, inputHigh)
		return None
	if inputLow > -1.0/constant and inputHigh > -1.0/constant:
		return triangleBounds(invFun, constant, inputVar, outputVar, inputLow, inputHigh, "neg")
	elif inputLow < -1.0/constant and inputHigh < -1.0/constant:
		return triangleBounds(invFun, constant, inputVar, outputVar, inputLow, inputHigh, "neg")
	elif inputLow < -1.0/constant and inputHigh > -1.0/constant:
		return triangleBounds(invFun, constant, inputVar, outputVar, inputLow, inputHigh, None)


#linear constraints for 1/(constant*x)
def inverseLinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	if inputLow == 0.0 or inputHigh == 0.0:
		print ("invalid lowBound or highBound for exponential", inputLow, inputHigh)
		return None
	if inputLow > 0.0 and inputHigh > 0:
		return triangleBounds(invFun, constant, inputVar, outputVar, inputLow, inputHigh, "pos")
	elif inputLow < 0.0 and inputHigh < 0.0:
		return triangleBounds(invFun, constant, inputVar, outputVar, inputLow, inputHigh, "neg")
	elif inputLow < 0.0 and inputHigh > 0.0:
		return triangleBounds(invFun, constant, inputVar, outputVar, inputLow, inputHigh, None)

#linear constraints for arcsin(constant*x)
def arcsinLinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	if inputLow < -1.0 or inputLow > 1.0 or inputHigh < -1.0 or inputHigh > 1.0:
		print ("invalid lowBound or highBound for arcsin", inputLow, inputHigh)
		return None

	if inputLow >= 0.0 and inputHigh >= 0.0:
		return triangleBounds(arcsinFun, constant, inputVar, outputVar, inputLow, inputHigh, "pos")

	elif inputLow <= 0.0 and inputHigh <= 0.0:
		return triangleBounds(arcsinFun, constant, inputVar, outputVar, inputLow, inputHigh, "neg")

	overallConstraint = "1 " + inputVar + " >= " + str(inputLow) + "\n"
	overallConstraint += "1 " + inputVar + " <= " + str(inputHigh) + "\n"
	allTrianglePoints = []
	allTrianglePoints += trianglePoints(arcsinFun, inputLow, 0.0, constant)
	allTrianglePoints += trianglePoints(arcsinFun, 0.0, inputHigh, constant)
	allTrianglePoints = np.array(allTrianglePoints)
	#print ("inputLow", inputLow, "inputHigh", inputHigh)
	#print ("allTrianglePoints")
	#print (allTrianglePoints)
	return overallConstraint + convexHullConstraints2D(allTrianglePoints, inputVar, outputVar)

#linear constraints for tanh
def tanhLinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	if constant < 0:
		if inputLow <= 0.0 and inputHigh <= 0.0:
			return triangleBounds(tanhFun, constant, inputVar, outputVar, inputLow, inputHigh, "neg")
		if inputLow >= 0.0 and inputHigh >= 0.0:
			return triangleBounds(tanhFun, constant, inputVar, outputVar, inputLow, inputHigh, "pos")
	elif constant >= 0:
		if inputLow <= 0.0 and inputHigh <= 0.0:
			return triangleBounds(tanhFun, constant, inputVar, outputVar, inputLow, inputHigh, "pos")
		if inputLow >= 0.0 and inputHigh >= 0.0:
			return triangleBounds(tanhFun, constant, inputVar, outputVar, inputLow, inputHigh, "neg")

	overallConstraint = "1 " + inputVar + " >= " + str(inputLow) + "\n"
	overallConstraint += "1 " + inputVar + " <= " + str(inputHigh) + "\n"
	overallConstraint += "1 " + outputVar + " <= 1.0\n"
	overallConstraint += "1 " + outputVar + " >= -1.0\n"
	allTrianglePoints = []
	allTrianglePoints += trianglePoints(tanhFun, inputLow, 0.0, constant)
	allTrianglePoints += trianglePoints(tanhFun, 0.0, inputHigh, constant)
	allTrianglePoints = np.array(allTrianglePoints)
	try:
		cHullConstraints = convexHullConstraints2D(allTrianglePoints, inputVar, outputVar)
		overallConstraint += cHullConstraints
	except:
		pass
	return overallConstraint



#linear constraints for arctan(constant*x)
def arctanLinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	if inputLow >= 0.0 and inputHigh >= 0.0:
		return triangleBounds(arctanFun, constant, inputVar, outputVar, inputLow, inputHigh, "neg")

	elif inputLow <= 0.0 and inputHigh <= 0.0:
		return triangleBounds(arctanFun, constant, inputVar, outputVar, inputLow, inputHigh, "pos")

	overallConstraint = "1 " + inputVar + " >= " + str(inputLow) + "\n"
	overallConstraint += "1 " + inputVar + " <= " + str(inputHigh) + "\n"
	allTrianglePoints = []
	allTrianglePoints += trianglePoints(arctanFun, inputLow, 0.0, constant)
	allTrianglePoints += trianglePoints(arctanFun, 0.0, inputHigh, constant)
	allTrianglePoints = np.array(allTrianglePoints)
	#print ("inputLow", inputLow, "inputHigh", inputHigh)
	#print ("allTrianglePoints")
	#print (allTrianglePoints)
	return overallConstraint + convexHullConstraints2D(allTrianglePoints, inputVar, outputVar)

#linear cnstraints for (const*x)^2
def xSquareLinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	return triangleBounds(xSquare, constant, inputVar, outputVar, inputLow, inputHigh, "pos")

#points of triangle formed by trianglebounds for a function
def trianglePoints(function, inputLow, inputHigh, constant):
	if inputLow > inputHigh:
		return []
	[funLow, dLow] = function(inputLow,constant)
	[funHigh, dHigh] = function(inputHigh,constant)
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


def triangleBounds(function, constant, inputVar, outputVar, inputLow, inputHigh, secDer):
	[funLow, dLow] = function(inputLow, constant)
	[funHigh, dHigh] = function(inputHigh, constant)
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
