# @author Itrat Ahmed Akhter
# Implementation of functions and gradients needed by our examples. 
# They can return either interval or point evaluations depending on 
# whether the arguments are points or intervals
# This file also contains functions that returns linear constraints
# bounding the function given an interval bound. 
import math
import numpy as np
from scipy.spatial import ConvexHull
from intervalBasics import *


#sin(const*x)
# assume that const > 0
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
# assume that const > 0
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


#tanh(a*x + b)
def tanhFun(x, a, b):
	tanhVal = np.tanh(a*x + b)
	if interval_p(x):
		return np.array([min(tanhVal[0], tanhVal[1]), max(tanhVal[0], tanhVal[1])])

	return tanhVal

def tanhFunder(x, a, b):
	den = np.cosh(a*x + b)*np.cosh(a*x + b)
	grad = np.divide(a,den)
	separX = b/(-a*1.0)
	if interval_p(x):
		if (x[0] - separX)*(x[1] - separX) >= 0:
			grad = np.array([min(grad[0], grad[1]), max(grad[0], grad[1])])
		else:
			den0 = np.cosh(separX)*np.cosh(separX)
			grad0 = np.divide(a,den0)
			grad = np.array([min(grad[0], grad[1], grad0), max(grad[0], grad[1], grad0)])

	return grad

#1/(const*x)
# assume that const > 0
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
# assume that const > 0
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



#linear constraints in the form of string for sin(constant*x)
def sinLinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	#print ("inputLow", inputLow, "inputHigh", inputHigh)
	inputLowPi = math.ceil(inputLow/(math.pi/constant))
	inputHighPi = math.ceil(inputHigh/(math.pi/constant))
	#print ("inputLowPi", inputLowPi, "inputHighPi", inputHighPi)

	if inputLowPi == inputHighPi:
		if inputLowPi%2 == 0:
			return triangleBounds(sinFun, sinFunder, inputVar, outputVar, inputLow, inputHigh, "pos", constant)
		else:
			return triangleBounds(sinFun, sinFunder, inputVar, outputVar, inputLow, inputHigh, "neg", constant)

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

#linear constraints in the form of a string for cos(constant*x)
def cosLinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	#print ("inputLow", inputLow, "inputHigh", inputHigh)
	inputLowPi = math.ceil((inputLow - math.pi/2.0)/(math.pi/constant))
	inputHighPi = math.ceil((inputHigh - math.pi/2.0)/(math.pi/constant))
	#print ("inputLowPi", inputLowPi, "inputHighPi", inputHighPi)

	if inputLowPi == inputHighPi:
		if inputLowPi%2 == 0:
			#print ("coming here?")
			return triangleBounds(cosFun, cosFunder, inputVar, outputVar, inputLow, inputHigh, "pos", constant)
		else:
			return triangleBounds(cosFun, cosFunder, inputVar, outputVar, inputLow, inputHigh, "neg", constant)

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



#linear constraints for in the form of a string for 1/(constant*x)
def inverseLinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	if inputLow == 0.0 or inputHigh == 0.0:
		raise  Exception("invalid lowBound or highBound for exponential" + str(inputLow) + " " + str(inputHigh))
	if inputLow > 0.0 and inputHigh > 0:
		return triangleBounds(invFun, invFunder, inputVar, outputVar, inputLow, inputHigh, "pos", constant)
	elif inputLow < 0.0 and inputHigh < 0.0:
		return triangleBounds(invFun, invFunder, inputVar, outputVar, inputLow, inputHigh, "neg", constant)
	elif inputLow < 0.0 and inputHigh > 0.0:
		return triangleBounds(invFun, invFunder, inputVar, outputVar, inputLow, inputHigh, None, constant)

#linear constraints in the form of a string for arcsin(constant*x)
def arcsinLinearConstraints(constant, inputVar, outputVar, inputLow, inputHigh):
	if inputLow < -1.0 or inputLow > 1.0 or inputHigh < -1.0 or inputHigh > 1.0:
		raise Exception("invalid lowBound or highBound for arcsin" + str(inputLow) + " " + str(inputHigh))
		return None

	if inputLow >= 0.0 and inputHigh >= 0.0:
		return triangleBounds(arcsinFun, arcsinFunder, inputVar, outputVar, inputLow, inputHigh, "pos", constant)

	elif inputLow <= 0.0 and inputHigh <= 0.0:
		return triangleBounds(arcsinFun, arcsinFunder, inputVar, outputVar, inputLow, inputHigh, "neg", constant)

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

#linear constraints in the form of a string for tanh
def tanhLinearConstraints(a, b, inputVar, outputVar, inputLow, inputHigh):
	separX = b/(-a*1.0)
	if a < 0:
		if inputLow <= separX and inputHigh <= separX:
			return triangleBounds(tanhFun, tanhFunder, inputVar, outputVar, inputLow, inputHigh, "neg", a, b)
		if inputLow >= separX and inputHigh >= separX:
			return triangleBounds(tanhFun, tanhFunder, inputVar, outputVar, inputLow, inputHigh, "pos", a, b)
	elif a >= 0:
		if inputLow <= separX and inputHigh <= separX:
			return triangleBounds(tanhFun, tanhFunder, inputVar, outputVar, inputLow, inputHigh, "pos", a, b)
		if inputLow >= separX and inputHigh >= separX:
			return triangleBounds(tanhFun, tanhFunder, inputVar, outputVar, inputLow, inputHigh, "neg", a, b)

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


# This function calculates tangents at inputLow and inputHigh and
# and finds the intersection between the tangents. 
# It returns the three points of a triangle:
# (inputLow, function(inputLow)), (inputHigh, function(inputHigh)), (intersectionX, function(intersectionX))
def trianglePoints(function, functionDer, inputLow, inputHigh, a, b=None):
	if inputLow > inputHigh:
		return []
	if b is None:
		funLow = function(inputLow, a)
		dLow = functionDer(inputLow, a)
		funHigh = function(inputHigh, a)
		dHigh = functionDer(inputHigh, a)
	else:
		funLow = function(inputLow, a, b)
		dLow = functionDer(inputLow, a, b)
		funHigh = function(inputHigh, a, b)
		dHigh = functionDer(inputHigh, a, b)
	
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


# This function constructs linear constraints from the given interval bounds
# If there are no inflection points of the function in the interval bounds
# then it returns triangle constraints formed from the tangents at the interval bounds
# and a secant line between the interval bounds depending on the convexity of the
# function. Otherwise it just returns constraints indicating the interval bounds
def triangleBounds(function, functionDer, inputVar, outputVar, inputLow, inputHigh, secDer, a, b=None):
	if b is None:
		funLow = function(np.array([inputLow]), a)[0]
		dLow = functionDer(np.array([inputLow]), a)[0]
		funHigh = function(np.array([inputHigh]), a)[0]
		dHigh = functionDer(np.array([inputHigh]), a)[0]
	else:
		funLow = function(np.array([inputLow]), a, b)[0]
		dLow = functionDer(np.array([inputLow]), a, b)[0]
		funHigh = function(np.array([inputHigh]), a, b)[0]
		dHigh = functionDer(np.array([inputHigh]), a, b)[0]
	
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


# This function finds the convex hull of a list of 2d points and creates
# constraints around the convex hull and returns it in the form of strings
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
