import numpy as np
import lpUtils
from cvxopt import matrix,solvers
from scipy.spatial import ConvexHull
import math

class Example1:
	def __init__(self, lowBound, upperBound):
		self.solver = None
		self.x = "x" # main variable
		self.a = "a" # a = 1/x
		self.b = "b" # b = sin(pi*a)
		self.c = "c" # c = arcsin(cos(0.797)*b)
		self.d = "d" # d = 2*x*c - 0.0331*x - 2*pi + 2.097
		self.constant = -2*math.pi + 2.097
		self.sign = " > "
		self.boundMap = []
		midVal = (lowBound + upperBound)/2.0
		self.boundMap.append({0:[lowBound,midVal],1:[midVal,upperBound]})

	def sinFun(self,x,const):
		Ifun = math.sin(const*x)
		der = const*math.cos(const*x)
		return [Ifun, der]

	def exponentialFun(self, x, const):
		Ifun = 1/(const*x)
		der = -1/(const*const*x*x)
		return [Ifun, der]

	def arcsinFun(self,x, const):
		Ifun = math.asin(const*x)
		der = const/(math.sqrt(1 - const*x*const*x))
		return [Ifun, der]

	def oscNum(self,xVal):
		val = 2*xVal*math.asin(math.cos(0.797)*math.sin(math.pi/xVal)) - 0.0331*xVal + self.constant
		return [None, None, val]

	def jacobian(self,x):
		jac = np.zeros((1,1))
		jac[0,0] = 2*math.asin(math.cos(0.797)*math.sin(math.pi/x)) - (2*math.cos(0.797)*math.pi*math.cos(math.pi/x))/(x*(- math.cos(0.797)*math.cos(0.797)*math.sin(math.pi/x)**2 + 1)**(1/2.0)) - 331/10000.0
		#jac[0,0] = 2*xVal*((-math.cos(0.797)*math.cos(math.pi/xVal)*(1/(xVal*xVal)))/(math.sqrt(1 - (math.cos(0.797)*math.sin(math.pi/xVal)*(math.cos(0.797)*math.sin(math.pi/xVal))))))
		#jac[0,0] += 2*math.asin(math.cos(0.797)*math.sin(math.pi/xVal)) - 0.0331
		return jac

	def jacobianInterval(self, bounds):
		lowBound = bounds[:,0]
		upperBound = bounds[:,1]
		jac = np.zeros((1,1,2))
		jac1 = self.jacobian(lowBound)
		jac2 = self.jacobian(upperBound)
		jac[:,:,0] = np.minimum(jac1, jac2)
		jac[:,:,1] = np.maximum(jac1, jac2)
		return jac

	def sinLinearConstraints(self, inputVar, outputVar, inputLow, inputHigh):
		inputLowPi = math.ceil(inputLow/math.pi)
		inputHighPi = math.ceil(inputHigh/math.pi)
		#print ("inputLowPi", inputLowPi, "inputHighPi", inputHighPi)

		if inputLowPi == inputHighPi:
			if inputLowPi%2 == 0:
				#print ("coming here?")
				return self.triangleBounds(self.sinFun, math.pi, inputVar, outputVar, inputLow, inputHigh, "pos")
			else:
				return self.triangleBounds(self.sinFun, math.pi, inputVar, outputVar, inputLow, inputHigh, "neg")

		overallConstraint = "1 " + inputVar + " >= " + str(inputLow) + "\n"
		overallConstraint += "1 " + inputVar + " <= " + str(inputHigh) + "\n"
		allTrianglePoints = []
		inputStart = inputLow
		piInput = inputStart/math.pi
		inputEnd = math.ceil(piInput)*math.pi
		while(inputStart < inputHigh):
			if inputEnd > inputHigh:
				inputEnd = inputHigh
			tPts = self.trianglePoints(self.sinFun, inputStart, inputEnd, math.pi)
			allTrianglePoints += tPts
			inputStart = inputEnd
			inputEnd += math.pi

		allTrianglePoints = np.array(allTrianglePoints)
		return overallConstraint + self.convexHullConstraints2D(allTrianglePoints, inputVar, outputVar)


	def exponentialLinearConstraints(self, inputVar, outputVar, inputLow, inputHigh):
		if inputLow == 0.0 or inputHigh == 0.0:
			print ("invalid lowBound or highBound for exponential", inputLow, inputHigh)
			return None
		if inputLow > 0.0 and inputHigh > 0:
			return self.triangleBounds(self.exponentialFun, 1, inputVar, outputVar, inputLow, inputHigh, "pos")
		elif inputLow < 0.0 and inputHigh < 0.0:
			return self.triangleBounds(self.exponentialFun, 1, inputVar, outputVar, inputLow, inputHigh, "neg")
		elif inputLow < 0.0 and inputHigh > 0.0:
			return self.triangleBounds(self.exponentialFun, 1, inputVar, outputVar, inputLow, inputHigh, None)
	
	def arcsinLinearConstraints(self, inputVar, outputVar, inputLow, inputHigh):
		if inputLow < -1.0 or inputLow > 1.0 or inputHigh < -1.0 or inputHigh > 1.0:
			print ("invalid lowBound or highBound for arcsin", inputLow, inputHigh)
			return None

		if inputLow >= 0.0 and inputHigh >= 0.0:
			return self.triangleBounds(self.arcsinFun, math.cos(0.797), inputVar, outputVar, inputLow, inputHigh, "pos")

		elif inputLow <= 0.0 and inputHigh <= 0.0:
			return self.triangleBounds(self.arcsinFun, math.cos(0.797), inputVar, outputVar, inputLow, inputHigh, "neg")

		overallConstraint = "1 " + inputVar + " >= " + str(inputLow) + "\n"
		overallConstraint += "1 " + inputVar + " <= " + str(inputHigh) + "\n"
		allTrianglePoints = []
		allTrianglePoints += self.trianglePoints(self.arcsinFun, inputLow, 0.0, math.cos(0.797))
		allTrianglePoints += self.trianglePoints(self.arcsinFun, 0.0, inputHigh, math.cos(0.797))
		allTrianglePoints = np.array(allTrianglePoints)
		#print ("inputLow", inputLow, "inputHigh", inputHigh)
		#print ("allTrianglePoints")
		#print (allTrianglePoints)
		return overallConstraint + self.convexHullConstraints2D(allTrianglePoints, inputVar, outputVar)
	
	def dLinearConstraints(self, zVar, inputVar, outputVar, patch):
		points = np.zeros((patch.shape[0],3))
		for i in range(patch.shape[0]):
			dVal = 2*patch[i,0]*patch[i,1] - 0.0331*patch[i,0] + self.constant
			points[i,:] = [patch[i,0],patch[i,1],dVal]

		#print ("points")
		#print (points)
		# hyperrectangle constraints
		boundaryPlanes = []
		boundaryPts = []
		midPoint = np.sum(points, axis = 0)/(points.shape[0]*1.0)
		for i in range(points.shape[0]):
			point1 = points[i,:]
			point2 = points[(i+1)%points.shape[0],:]
			m = None, None
			norms = np.zeros((3))
			if point2[0] - point1[0] == 0:
				m = float("inf")
				norms[0] = 1
			else:
				m = (point2[1] - point1[1])/(point2[0] - point1[0])
				norms[0] = -m
				norms[1] = 1
			cSign = " <= "
			d = norms[0]*point1[0] + norms[1]*point1[1] + norms[2]*point1[2]
			dMid = norms[0]*midPoint[0] + norms[1]*midPoint[1]
			if dMid > d:
				cSign = " >= "
			boundaryPlanes.append([np.array([[point1[0],point1[1],point1[2]],[norms[0],norms[1],norms[2]]]), cSign])
			boundaryPts.append([point1, point2])

	
		feasiblePoints = self.saddleConvexHull(boundaryPlanes, boundaryPts)
		feasiblePoints = np.array(feasiblePoints)
		if len(feasiblePoints) >= 1:
			feasiblePoints = np.unique(feasiblePoints, axis = 0)

		#print ("feasiblePoints", feasiblePoints)
		overallConstraint = self.convexHullConstraints(feasiblePoints, zVar, inputVar, outputVar)
		return overallConstraint

	def trianglePoints(self, function, inputLow, inputHigh, constant):
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

		tPts = [[inputLow, funLow],[inputHigh, funHigh],[leftIntersectX, leftIntersectY]]
		return tPts


	def triangleBounds(self, function, constant, inputVar, outputVar, inputLow, inputHigh, secDer):
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

	def convexHullConstraints2D(self, points, inputVar, outputVar):
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
			
		return overallConstraint


	def convexHullConstraints(self, feasiblePoints, zVar, inVar, outVar):
		hull = ConvexHull(feasiblePoints)
		convexHullMiddle = np.zeros((3))
		numPoints = 0
		for simplex in hull.simplices:
			for index in simplex:
				convexHullMiddle += feasiblePoints[index,:]
				numPoints += 1
		convexHullMiddle = convexHullMiddle/(numPoints*1.0)

		overallConstraint = ""
		for si in range(len(hull.simplices)):
			simplex = hull.simplices[si]
			#print ("simplex", simplex)
			pointsFromSimplex = np.zeros((3,3))
			for ii in range(3):
				pointsFromSimplex[ii] = feasiblePoints[simplex[ii]]
			
			#print ("pointsFromSimplex", pointsFromSimplex)
			normal = np.cross(pointsFromSimplex[1] - pointsFromSimplex[0], pointsFromSimplex[2] - pointsFromSimplex[0])
			'''if normal[2] < 0:
				normal = -normal'''
			pointInPlane = pointsFromSimplex[0]
			#print ("pointsFromSimplex", pointsFromSimplex)
			d = normal[0]*pointInPlane[0] + normal[1]*pointInPlane[1] + normal[2]*pointInPlane[2]
			middleD = normal[0]*convexHullMiddle[0] + normal[1]*convexHullMiddle[1] + normal[2]*convexHullMiddle[2]
			# Determine if the middle of the convex hull is above or below
			# the plane and add the constraint related to the plane accordingly
			sign = " <= "

			#print ("middleD", middleD)
			#print ("d", d)
			if middleD > d:
				sign = " >= "

			#print ("normal", normal)
			#print ("pointInPlane", pointInPlane)
			'''print ("sign", sign)
			print ("")'''
			
			#if np.greater_equal(np.absolute(normal),np.ones(normal.shape)*1e-5).any():
			overallConstraint += str(normal[2])+" " + zVar + " + " + str(normal[0]) + " " + inVar +\
				" + " + str(normal[1]) + " " + outVar + sign + str(d) + "\n"
		return overallConstraint

	def intersectSurfPlaneFunDer(self, inVal, outVal, plane):
		planePt = plane[0,:]
		planeNorm = plane[1,:]
		m, d = None, None
		if planeNorm[1] != 0:
			m = -planeNorm[0]
			d = planeNorm[0]*planePt[0] + planeNorm[1]*planePt[1] + planeNorm[2]*planePt[2]
		else:
			d = planePt[0]

		z = 0.0
		firDers = np.zeros((2))
		derTypes = [False, False]
		if m is None:
			z += 2*d*outVal - 0.0331*d + self.constant
			firDers[1] += 2*d
			derTypes[1] = True
		else:
			z += 2*m*inVal*inVal + 2*d*inVal - 0.0331*inVal + self.constant
			firDers[0] += 4*m*inVal + 2*d - 0.0331
			derTypes[0] = True

		return [z, firDers, derTypes]

	def saddleConvexHull(self, boundaryPlanes, boundaryPts):
		#print ("saddleConvexHull", transistorNumber)
		feasiblePoints = []
		for pi in range(len(boundaryPlanes)):
			plane = boundaryPlanes[pi][0]
			point1 = boundaryPts[pi][0]
			point2 = boundaryPts[pi][1]
			funValue1, firDers1, derTypes1 = self.intersectSurfPlaneFunDer(point1[0], point1[1], plane)
			funValue2, firDers2, derTypes2 = self.intersectSurfPlaneFunDer(point2[0], point2[1], plane)
			planeNormal = plane[1,:]
			planePt = plane[0,:]
			d = planePt[0]*planeNormal[0] + planePt[1]*planeNormal[1] + planePt[2]*planeNormal[2]
			feasiblePoints.append(point1)
			feasiblePoints.append(point2)
			#print ("point1", point1, "point2", point2)
			#print ("planeNormal", planeNormal)
			if not(derTypes1[0]) and not(derTypes2[0]):
				m1, m2 = firDers1[1], firDers2[1]
				c1 = point1[2] - m1*point1[1]
				c2 = point2[2] - m2*point2[1]
				intersectingPt = None
				if abs(m1 - m2) < 1e-14:
					xPt = (point1[1] + point2[1])/2.0
					yPt = (point1[2] + point2[2])/2.0
					intersectingPt = np.array([xPt, yPt])
				else:
					xPt = (c2 - c1)/(m1 - m2)
					yPt = m1*xPt + c1
					intersectingPt = np.array([xPt, yPt])
				# TODO: check if this makes sense
				missingCoord = None
				if planeNormal[1] == 0:
					missingCoord = point1[0]
				elif planeNormal[0] != 0:
					missingCoord = (intersectingPt[0] - d)/planeNormal[0]
				if missingCoord is not None:
					feasiblePoints.append([missingCoord,intersectingPt[0],intersectingPt[1]])
				#print ("feasiblePt added if", feasiblePoints[-1])

			elif not(derTypes1[1]) and not(derTypes2[1]):
				m1, m2 = firDers1[0], firDers2[0]
				c1 = point1[2] - m1*point1[0]
				c2 = point2[2] - m2*point2[0]
				#print ("m1", m1, "m2", m2)
				intersectingPt = None
				if abs(m1 - m2) < 1e-14:
					xPt = (point1[0] + point2[0])/2.0
					yPt = (point1[2] + point2[2])/2.0
					intersectingPt = np.array([xPt, yPt])
				else:
					xPt = (c2 - c1)/(m1 - m2)
					yPt = m1*xPt + c1
					intersectingPt = np.array([xPt, yPt])
				# TODO: check if this makes sense
				missingCoord = planeNormal[0]*intersectingPt[0] + d
				feasiblePoints.append([intersectingPt[0],missingCoord,intersectingPt[1]])
				#print ("feasiblePt added else", feasiblePoints[-1])
		return feasiblePoints


	def linearConstraints(self, hyperRectangle):
		solvers.options["show_progress"] = False
		allConstraints = ""
		#self.a = "a" # a = 1/x
		#self.b = "b" # b = sin(pi*a)
		#self.c = "c" # c = arcsin(cos(0.797)*b)
		#self.d = "d" # d = 2*x*c - 0.0331*x - 2*pi + 2.097

		xLowBound = hyperRectangle[0,0]
		xUpperBound = hyperRectangle[0,1]
		aLowBound = 1.0/xUpperBound
		aUpperBound = 1.0/xLowBound
		bLowBound = min(self.sinFun(aLowBound,math.pi)[0], self.sinFun(aUpperBound,math.pi)[0])
		bUpperBound = max(self.sinFun(aLowBound,math.pi)[0], self.sinFun(aUpperBound,math.pi)[0])
		#print ("(aLowBound - math.pi/2.0)/(math.pi)",aLowBound/(math.pi/2.0))
		#print ("(aUpperBound - math.pi/2.0)/(math.pi)", aUpperBound/(math.pi/2.0))
		if abs(aLowBound/aUpperBound) > 1.0:
			bLowBound= -1.0
			bUpperBound = 1.0
		#print ("bLowBound", bLowBound, "bUpperBound", bUpperBound)
		cLowBound = min(self.arcsinFun(bLowBound,math.cos(0.797))[0], self.arcsinFun(bUpperBound,math.cos(0.797))[0])
		cUpperBound = max(self.arcsinFun(bLowBound,math.cos(0.797))[0], self.arcsinFun(bUpperBound,math.cos(0.797))[0])

		allConstraints += self.exponentialLinearConstraints(self.x, self.a, xLowBound, xUpperBound)
		allConstraints += self.sinLinearConstraints(self.a, self.b, aLowBound, aUpperBound)
		allConstraints += self.arcsinLinearConstraints(self.b, self.c, bLowBound, bUpperBound)
		

		patch = np.zeros((4,2))
		patch[0,:] = [xLowBound, cLowBound]
		patch[1,:] = [xUpperBound, cLowBound]
		patch[2,:] = [xUpperBound, cUpperBound]
		patch[3,:] = [xLowBound, cUpperBound]
		allConstraints += self.dLinearConstraints(self.d, self.x, self.c, patch)
		#allConstraints += "1 " + self.x + " >= -3.60\n"
		#allConstraints += "1 " + self.x + " <= -3.58\n"
		#allConstraints += "1 " + self.a + " >= -0.278\n"
		#allConstraints += "1 " + self.a + " <= -0.276\n"
		#allConstraints += "1 " + self.b + " >= -0.767\n"
		#allConstraints += "1 " + self.b + " <= -0.765\n"
		#allConstraints += "1 " + self.c + " >= -0.566\n"
		#allConstraints += "1 " + self.c + " <= -0.564\n"
		#print ("xLowBound", xLowBound, "xUpperBound", xUpperBound)
		#print ("aLowBound", aLowBound, "aUpperBound", aUpperBound)
		#print ("cLowBound", cLowBound, "cUpperBound", cUpperBound)
		allConstraints += "1 " + self.d + " >= 0\n"
		allConstraints += "1 " + self.d + " <= 0\n"

		
		#print ("allConstraints")
		#print (allConstraints)

		'''allConstraintList = allConstraints.splitlines()
		allConstraints = ""
		for i in range(len(allConstraintList)):
			if (i >= 2 and i <= 3) or (i == 9) or (i == 17) or (i >= 21 and i <= 23):
				allConstraints += allConstraintList[i] + "\n"
		print ("numConstraints ", len(allConstraintList))'''

		if self.solver is None:
			variableDict, A, B = lpUtils.constructCoeffMatrices(allConstraints)
			newHyperRectangle = np.copy(hyperRectangle)
			feasible = True

			minObjConstraint = "min 1 " + self.x
			maxObjConstraint = "max 1 " + self.x
			Cmin = lpUtils.constructObjMatrix(minObjConstraint,variableDict)
			Cmax = lpUtils.constructObjMatrix(maxObjConstraint,variableDict)
			minSol = solvers.lp(Cmin,A,B)
			maxSol = solvers.lp(Cmax,A,B)
			if minSol["status"] == "primal infeasible" and maxSol["status"] == "primal infeasible":
				feasible = False
			else:
				if minSol["status"] == "optimal":
					newHyperRectangle[0,0] = minSol['x'][variableDict[self.x]] - 1e-6
				if maxSol["status"] == "optimal":
					newHyperRectangle[0,1] = maxSol['x'][variableDict[self.x]] + 1e-6

			return [feasible, newHyperRectangle]



if __name__ == "__main__":
	example1 = Example1()
	print (example1.oscNum(20))
	print (example1.jacobian(20))

