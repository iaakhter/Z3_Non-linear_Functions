import numpy as np
import lpUtils
from cvxopt import matrix,solvers
from scipy.spatial import ConvexHull
from scipy import optimize
import math
import funCompUtils as fcUtils
from intervalBasics import *

class Meti25:
	def __init__(self, lowBound, upperBound, sign):
		self.x = "x" # main variable
		self.a = "a" # a = sin(x/3)
		self.b = "b" # b = sin(3*x)
		self.c = "c" # c = a + b/6.0
		self.sign = sign
		self.bounds = [[lowBound, upperBound]]

	def f(self, V):
		if hasattr(V,'ndim') and len(V.shape) == 2:
			V = V[0]
		return np.array([interval_add(
			fcUtils.sinFun(V, 1.0/3.0),
			interval_div(fcUtils.sinFun(V, 3.0),6.0))])


	def jacobian(self,x):
		if hasattr(x,'ndim') and len(x.shape) == 2:
			x = x[0]
		intervalVal = interval_p(x)

		if intervalVal:
			return self.jacobianInterval(x)
		else:	
			jac = np.zeros((1,1))
			jac[0,0] = math.cos(x/3.0)/3.0 + math.cos(x*3.0)/2.0
			return jac

	def secondDer(self,x):
		return (-1.0/9.0)*math.sin(x/3.0) - 1.5*math.sin(3*x)


	def jacobianInterval(self, bounds):
		lowBound = bounds[0]
		upperBound = bounds[1]

		jac = np.zeros((1,1,2))
		jac1 = self.jacobian(lowBound)
		jac2 = self.jacobian(upperBound)
		
		jac[:,:,0] = np.minimum(jac1, jac2)
		jac[:,:,1] = np.maximum(jac1, jac2)

		# try to find x between bounds where secondDer is zer0
		startBound = lowBound
		while startBound < upperBound:
			endBound = startBound + 0.5
			if endBound > upperBound:
				endBound = upperBound
			try:
				xRoot = optimize.bisect(self.secondDer, startBound, endBound)
				derXRoot = self.jacobian(xRoot)
				#print ("xRoot", xRoot, "derXRoot", derXRoot)
				jac[:,:,0] = min(jac[:,:,0], derXRoot)
				jac[:,:,1] = max(jac[:,:,1], derXRoot)
			except ValueError:
				pass
			startBound = endBound

		#print ("jac", jac)
		
		return jac
	

	def linearConstraints(self, hyperRectangle):
		solvers.options["show_progress"] = False
		allConstraints = ""
		#self.x = "x" # main variable
		#self.a = "a" # a = sin(x/3)
		#self.b = "b" # b = sin(3*x)
		#self.c = "c" # c = a + b/6.0

		xLowBound = hyperRectangle[0,0]
		xUpperBound = hyperRectangle[0,1]

		allConstraints += fcUtils.sinLinearConstraints(1/3.0, self.x, self.a, xLowBound, xUpperBound)
		allConstraints += fcUtils.sinLinearConstraints(3.0, self.x, self.b, xLowBound, xUpperBound)

		allConstraints += "1 " + self.c + " + " + "-1 " + self.a + " + " + str(-1/6.0) + " " + self.b + " >= 0\n"
		allConstraints += "1 " + self.c + " + " + "-1 " + self.a + " + " + str(-1/6.0) + " " + self.b + " <= 0\n"
		
		allConstraints += "1 " + self.c + " >= 0\n"
		allConstraints += "1 " + self.c + " <= 0\n"


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


class Meti18:
	def __init__(self, lowBound, upperBound, sign):
		self.x = "x" # main variable
		self.a = "a" # a = cos(pi*x)
		self.b = "b" # b = a - 1 + 2x
		self.sign = sign
		self.bounds = [[lowBound, upperBound]]

	def f(self, V):
		if hasattr(V,'ndim') and len(V.shape) == 2:
			V = V[0]
		return np.array([interval_add(
						interval_sub(fcUtils.cosFun(V, math.pi), 1),
						interval_mult(2,V))])
	

	def jacobian(self,x):
		if hasattr(x,'ndim') and len(x.shape) == 2:
			x = x[0]
		intervalVal = interval_p(x)

		if intervalVal:
			return self.jacobianInterval(x)
		
		jac = np.zeros((1,1))
		jac[0,0] = -math.pi*math.sin(math.pi*x) + 2.0
		return jac

	def secondDer(self,x):
		return -math.pi*math.pi*math.cos(math.pi*x)


	def jacobianInterval(self, bounds):
		lowBound = bounds[0]
		upperBound = bounds[1]

		jac = np.zeros((1,1,2))
		jac1 = self.jacobian(lowBound)
		jac2 = self.jacobian(upperBound)
		
		jac[:,:,0] = np.minimum(jac1, jac2)
		jac[:,:,1] = np.maximum(jac1, jac2)

		# try to find x between bounds where secondDer is zer0
		startBound = lowBound
		while startBound < upperBound:
			endBound = startBound + 0.5
			if endBound > upperBound:
				endBound = upperBound
			try:
				xRoot = optimize.bisect(self.secondDer, startBound, endBound)
				derXRoot = self.jacobian(xRoot)
				#print ("xRoot", xRoot, "derXRoot", derXRoot)
				jac[:,:,0] = min(jac[:,:,0], derXRoot)
				jac[:,:,1] = max(jac[:,:,1], derXRoot)
			except ValueError:
				pass
			startBound = endBound

		#print ("jac", jac)
		
		return jac
	

	def linearConstraints(self, hyperRectangle):
		solvers.options["show_progress"] = False
		allConstraints = ""
		#self.x = "x" # main variable
		#self.a = "a" # a = cos(pi*x)
		#self.c = "b" # b = a - 1 + 2x

		xLowBound = hyperRectangle[0,0]
		xUpperBound = hyperRectangle[0,1]

		allConstraints += fcUtils.cosLinearConstraints(math.pi, self.x, self.a, xLowBound, xUpperBound)

		allConstraints += "1 " + self.b + " + " + "-1 " + self.a + " + " + str(-2.0) + " " + self.x + " >= -1\n"
		allConstraints += "1 " + self.b + " + " + "-1 " + self.a + " + " + str(-2.0) + " " + self.x + " <= -1\n"
		
		allConstraints += "1 " + self.b + " >= 0\n"
		allConstraints += "1 " + self.b + " <= 0\n"


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

class Meti10:
	def __init__(self, lowBound, upperBound, sign):
		self.x = "x" # main variable
		self.a = "a" # a = x/(2 + x)
		self.b = "b" # b = ln(1 + x)
		self.c = "c" # c = 2*a - b
		self.sign = sign
		self.bounds = [[lowBound, upperBound]]


	def oscNum(self,xVal):
		#print ("xVal", xVal)
		val = (2.0*xVal)/(2 + xVal) - math.log(1.0 + xVal)
		return [None, None, val]

	def jacobian(self,x):
		jac = np.zeros((1,1))
		jac[0,0] = 2.0/(x + 2.0) - 1.0/(x + 1.0) - (2.0*x)/(x + 2.0)**2
		return jac


	def jacobianInterval(self, bounds):
		lowBound = bounds[:,0]
		upperBound = bounds[:,1]

		jac = np.zeros((1,1,2))
		jac1 = self.jacobian(lowBound)
		jac2 = self.jacobian(upperBound)
		
		jac[:,:,0] = np.minimum(jac1, jac2)
		jac[:,:,1] = np.maximum(jac1, jac2)
		print "jac", jac
		return jac
	

	def linearConstraints(self, hyperRectangle):
		solvers.options["show_progress"] = False
		allConstraints = ""
		#self.x = "x" # main variable
		#self.a = "a" # a = x/(2 + x)
		#self.b = "b" # b = ln(1 + x)
		#self.c = "c" # c = 2*a - b


		xLowBound = hyperRectangle[0,0]
		xUpperBound = hyperRectangle[0,1]

		allConstraints += fcUtils.inverse1LinearConstraints(1.0, self.x, self.a, xLowBound, xUpperBound)
		allConstraints += fcUtils.log1LinearConstraints(1.0, self.x, self.b, xLowBound, xUpperBound)

		allConstraints += "1 " + self.c + " + " + "-2 " + self.a + " + "  + "1 " + self.b + " >= 0\n"
		allConstraints += "1 " + self.c + " + " + "-2 " + self.a + " + "  + "1 " + self.b + " <= 0\n"
		
		allConstraints += "1 " + self.c + " >= 0\n"
		allConstraints += "1 " + self.c + " <= 0\n"


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
	model = Meti25(0.0, 100.0, ">")
	#print (example1.oscNum(20))
	#print (example1.jacobian(20))
	x = np.linspace(0.0, 100.0, 1000)
	for i in range(len(x)-1):
		xBound = np.array([x[i], x[i+1]])
		fVal = model.f(xBound)

		sampleDelta = 0.0001
		xSamples = np.linspace(xBound[0] + sampleDelta, xBound[1] - sampleDelta, 100)
		for xs in xSamples:
			f = model.oscNum(xs)[2]
			if f < fVal[0] or f > fVal[1]:
				print "oops x ", xs, "for interval", xBound
				print "actual f", f, "should be in", fVal

	model = Meti18(0.0, 100.0, ">")
	#print (example1.oscNum(20))
	#print (example1.jacobian(20))
	x = np.linspace(0.0, 100.0, 1000)
	for i in range(len(x)-1):
		xBound = np.array([x[i], x[i+1]])
		fVal = model.f(xBound)

		sampleDelta = 0.0001
		xSamples = np.linspace(xBound[0] + sampleDelta, xBound[1] - sampleDelta, 100)
		for xs in xSamples:
			f = model.oscNum(xs)[2]
			if f < fVal[0] or f > fVal[1]:
				print "oops x ", xs, "for interval", xBound
				print "actual f", f, "should be in", fVal

