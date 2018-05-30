import numpy as np
import lpUtils
from cvxopt import matrix,solvers
import funCompUtils as fcUtils
#from z3 import *
from intervalBasics import *

class TanhModel:
	def __init__(self, modelParam, g_cc, g_fwd, numStages, solver = None):
		# gradient of tanh -- y = tanh(modelParam*x)
		self.solver = solver
		self.modelParam = modelParam
		self.g_cc = g_cc
		self.g_fwd = g_fwd
		self.numStages = numStages
		self.xs = []
		self.ys = []
		self.zs = []
		if solver is None:
			for i in range(numStages*2):
				self.xs.append("x" + str(i))
				self.ys.append("y" + str(i))
				self.zs.append("z" + str(i))
		else:
			self.xs = RealVector("x", numStages*2)
			self.ys = RealVector("y", numStages*2)
			self.zs = RealVector("z", numStages*2)

		self.boundMap = []
		for i in range(numStages*2):
			self.boundMap.append({0:[-1.0,0.0],1:[0.0,1.0]})

	'''
	takes in non-symbolic python values
	calculates the tanhFun of val
	'''
	def tanhFun(self,val):
		tanhVal = np.tanh(self.modelParam*val)
		if interval_p(val):
			return np.array([min(tanhVal[0], tanhVal[1]), max(tanhVal[0], tanhVal[1])])
		return tanhVal
		#return -(exp(a*val) - exp(-a*val))/(exp(a*val) + exp(-a*val))


	'''
	takes in non-symbolic python values
	calculates the derivative of tanhFun of val
	'''
	def tanhFunder(self,val):
		den = np.cosh(self.modelParam*val)*np.cosh(self.modelParam*val)
		#print "den ", den
		return self.modelParam/den
				
	def oscNum(self,V):
		lenV = len(V)
		Vin = [V[i % lenV] for i in range(-1,lenV-1)]
		Vcc = [V[(i + lenV//2) % lenV] for i in range(lenV)]
		VoutFwd = [self.tanhFun(Vin[i]) for i in range(lenV)]
		VoutCc = [self.tanhFun(Vcc[i]) for i in range(lenV)]
		return (VoutFwd, VoutCc, [((self.tanhFun(Vin[i])-V[i])*self.g_fwd
				+(self.tanhFun(Vcc[i])-V[i])*self.g_cc) for i in range(lenV)])

	def f(self, bounds):
		lenV = bounds.shape[0]
		IBounds = np.zeros((lenV,2))
		for i in range(lenV):
			fwdInd = (i-1)%lenV
			ccInd = (i + lenV//2) % lenV
			tanhFwd = self.tanhFun(bounds[fwdInd,:])
			tanhCc = self.tanhFun(bounds[ccInd,:])
			#print ("bounds[fwdInd]", bounds[fwdInd], "tanhFwd", tanhFwd)
			#print ("bounds[ccInd]", bounds[ccInd], "tanhCc", tanhCc)
			fwdTerm = interval_sub(tanhFwd, bounds[i,:])
			ccTerm = interval_sub(tanhCc, bounds[i,:])
			#print ("bounds[i,:]", bounds[i,:])
			#print ("fwdTerm", fwdTerm, "ccTerm", ccTerm)
			IBounds[i,:] = interval_add(interval_mult(self.g_fwd, fwdTerm),interval_mult(self.g_cc, ccTerm))

		return IBounds


	'''Get jacobian of rambus oscillator at V
	'''
	def jacobian(self,V):
		lenV = len(V)
		Vin = [V[i % lenV] for i in range(-1,lenV-1)]
		Vcc = [V[(i + lenV//2) % lenV] for i in range(lenV)]
		jac = np.zeros((lenV, lenV))
		for i in range(lenV):
			jac[i,i] = -(self.g_fwd + self.g_cc)
			jac[i,(i-1)%lenV] = self.g_fwd * self.tanhFunder(V[(i-1)%lenV])
			jac[i,(i + lenV//2) % lenV] = self.g_cc * self.tanhFunder(V[(i + lenV//2) % lenV])

		return jac

	def jacobianInterval(self,bounds):
		lowerBound = bounds[:,0]
		upperBound = bounds[:,1]
		lenV = len(lowerBound)
		jac = np.zeros((lenV, lenV,2))
		zerofwd =  self.g_fwd * self.tanhFunder(0)
		zerocc = self.g_cc * self.tanhFunder(0)
		for i in range(lenV):
			jac[i,i,0] = -(self.g_fwd + self.g_cc)
			jac[i,i,1] = -(self.g_fwd + self.g_cc)
			gfwdVal1 = self.g_fwd * self.tanhFunder(lowerBound[(i-1)%lenV])
			gfwdVal2 = self.g_fwd * self.tanhFunder(upperBound[(i-1)%lenV])
			if lowerBound[(i-1)%lenV] < 0 and upperBound[(i-1)%lenV] > 0:
				jac[i,(i-1)%lenV,0] = min(gfwdVal1,gfwdVal2,zerofwd)
				jac[i,(i-1)%lenV,1] = max(gfwdVal1,gfwdVal2,zerofwd)
			else:
				jac[i,(i-1)%lenV,0] = min(gfwdVal1,gfwdVal2)
				jac[i,(i-1)%lenV,1] = max(gfwdVal1,gfwdVal2)
			gccVal1 = self.g_cc * self.tanhFunder(lowerBound[(i + lenV//2) % lenV])
			gccVal2 = self.g_cc * self.tanhFunder(upperBound[(i + lenV//2) % lenV])
			if lowerBound[(i + lenV//2) % lenV] < 0 and upperBound[(i + lenV//2) % lenV] > 0:
				jac[i,(i + lenV//2) % lenV,0] = min(gccVal1,gccVal2,zerocc)
				jac[i,(i + lenV//2) % lenV,1] = max(gccVal1,gccVal2,zerocc)
			else:
				jac[i,(i + lenV//2) % lenV,0] = min(gccVal1,gccVal2)
				jac[i,(i + lenV//2) % lenV,1] = max(gccVal1,gccVal2)
		return jac

	def ignoreHyperInZ3(self, hyperRectangle):
		lenV = hyperRectangle.shape[0]
		constraintList = []
		for i in range(lenV):
			constraintList.append(self.xs[i] < hyperRectangle[i][0])
			constraintList.append(self.xs[i] > hyperRectangle[i][1])
		self.solver.add(Or(*constraintList))

	def ignoreSolInZ3(self, sol):
		lenV = len(sol)
		self.solver.add(Or(*[self.xs[i] != sol[i] for i in range(lenV)]))

	def addDomainConstraint(self):
		lenV = self.numStages*2
		for i in range(lenV):
			self.solver.add(self.xs[i] >= -1.0)
			self.solver.add(self.xs[i] <= 1.0)
		'''for numStage in range(self.numStages):
			lastStage = self.numStages - numStage-1
			if lastStage == 0:
				break
			secondLastStage = lastStage - 1
			lastStageIndex1 = lastStage
			lastStageIndex2 = (lastStage + lenV//2) % lenV
			secondLastStageIndex1 = secondLastStage
			secondLastStageIndex2 = (secondLastStage + lenV//2) % lenV
			print ("lastStageIndex1", lastStageIndex1, "lastStageIndex2", lastStageIndex2)
			print ("secondLastStageIndex1", secondLastStageIndex1, "secondLastStageIndex2", secondLastStageIndex2)
			self.solver.add(Implies(And(self.xs[lastStageIndex1] >= 0,  self.xs[lastStageIndex2] >= 0),
							And(self.xs[secondLastStageIndex1] <= 0, self.xs[secondLastStageIndex2] <= 0)))
			self.solver.add(Implies(And(self.xs[lastStageIndex1] <= 0,  self.xs[lastStageIndex2] <= 0),
							And(self.xs[secondLastStageIndex1] >= 0, self.xs[secondLastStageIndex2] >= 0)))'''

		for i in range(lenV):
			self.solver.add(self.g_fwd*self.ys[i] + (-self.g_fwd - self.g_cc)*self.xs[i] + self.g_cc*self.zs[i] == 0 )

	def pickVariableRangeConstraint(self, index, interval):
		self.solver.add(self.xs[index] >= interval[0])
		self.solver.add(self.xs[index] <= interval[1])

	def thisOrThatHyperConstraint(self, hyper1, hyper2):
		lenV = self.numStages*2
		for i in range(lenV):
			if i == self.numStages - 1 or i == lenV - 1:
				self.solver.add(self.xs[i] >= hyper1[i][0])
				self.solver.add(self.xs[i] <= hyper2[i][1])

	def linearConstraints(self, hyperRectangle):
		solvers.options["show_progress"] = False
		allConstraints = ""
		lenV = self.numStages*2

		allConstraints = ""
		for i in range(lenV):
			fwdInd = (i-1)%lenV
			ccInd = (i+lenV//2)%lenV
			#print "fwdInd ", fwdInd, " ccInd ", ccInd
			#print "hyperRectangle[fwdInd][0]", hyperRectangle[fwdInd][0], "hyperRectangle[fwdInd][1]", hyperRectangle[fwdInd][1]
			
			if self.solver is None:
				triangleClaimFwd = fcUtils.tanhLinearConstraints(self.modelParam, self.xs[fwdInd], self.ys[i], hyperRectangle[fwdInd,0],hyperRectangle[fwdInd,1])
				allConstraints += triangleClaimFwd

				triangleClaimCc = fcUtils.tanhLinearConstraints(self.modelParam, self.xs[ccInd], self.zs[i], hyperRectangle[ccInd,0],hyperRectangle[ccInd,1])
				allConstraints += triangleClaimCc
					
				allConstraints += str(self.g_fwd) + " " + self.ys[i] + " + " + str(-self.g_fwd-self.g_cc) + \
				" " + self.xs[i] + " + " + str(self.g_cc) + " "  + self.zs[i] + " >= 0.0\n"
				allConstraints += str(self.g_fwd) + " " + self.ys[i] + " + " + str(-self.g_fwd-self.g_cc) + \
				" " + self.xs[i] + " + " + str(self.g_cc) + " "  + self.zs[i] + " <= 0.0\n"
			else:
				
				if hyperRectangle[fwdInd,0] < 0 and hyperRectangle[fwdInd,1] > 0:
					self.triangleConvexHullBounds(self.xs[fwdInd],self.ys[i],hyperRectangle[fwdInd,0],hyperRectangle[fwdInd,1])
				else:
					self.triangleBounds(self.xs[fwdInd],self.ys[i],hyperRectangle[fwdInd,0],hyperRectangle[fwdInd,1])

				if hyperRectangle[ccInd,0] < 0 and hyperRectangle[ccInd,1] > 0:
					self.triangleConvexHullBounds(self.xs[ccInd],self.zs[i],hyperRectangle[ccInd,0],hyperRectangle[ccInd,1])
				else:
					self.triangleBounds(self.xs[ccInd],self.zs[i],hyperRectangle[ccInd,0],hyperRectangle[ccInd,1])

		if self.solver is None:
			'''allConstraintList = allConstraints.splitlines()
			allConstraints = ""
			for i in range(len(allConstraintList)):
				allConstraints += allConstraintList[i] + "\n"
			print "numConstraints ", len(allConstraintList)'''
			#print "allConstraints"
			#print allConstraints
			variableDict, A, B = lpUtils.constructCoeffMatrices(allConstraints)
			newHyperRectangle = np.copy(hyperRectangle)
			feasible = True
			for i in range(lenV):
				#print "min max ", i
				minObjConstraint = "min 1 " + self.xs[i]
				maxObjConstraint = "max 1 " + self.xs[i]
				Cmin = lpUtils.constructObjMatrix(minObjConstraint,variableDict)
				Cmax = lpUtils.constructObjMatrix(maxObjConstraint,variableDict)
				minSol = solvers.lp(Cmin,A,B)
				maxSol = solvers.lp(Cmax,A,B)
				if minSol["status"] == "primal infeasible" and maxSol["status"] == "primal infeasible":
					feasible = False
					break
				else:
					if minSol["status"] == "optimal":
						newHyperRectangle[i,0] = minSol['x'][variableDict[self.xs[i]]] - 1e-6
					if maxSol["status"] == "optimal":
						newHyperRectangle[i,1] = maxSol['x'][variableDict[self.xs[i]]] + 1e-6

			return [feasible, newHyperRectangle]


if __name__ == "__main__":
	# do some testing to check if the interval function makes sense
	model = TanhModel(modelParam = -5, g_cc = 0.5, g_fwd = 1.0, numStages = 2)
	V0 = np.linspace(-1.0, 1.0, 10)
	V1 = np.linspace(-1.0, 1.0, 10)
	V2 = np.linspace(-1.0, 1.0, 10)
	V3 = np.linspace(-1.0, 1.0, 10)
	for i in range(len(V0)-1):
		for j in range(len(V1)-1):
			for k in range(len(V2)-1):
				for l in range(len(V3)-1):
					vInterval = np.array([[V0[i], V0[i+1]],
											[V1[j], V1[j+1]],
											[V2[k], V2[k+1]],
											[V3[l], V3[l+1]]])
					INum = model.f(vInterval)
				
					sampleDelta = 0.0001
					v0Samples = np.linspace(V0[i]+ sampleDelta, V0[i+1]- sampleDelta,4)
					v1Samples = np.linspace(V1[j]+ sampleDelta, V1[j+1]- sampleDelta,4)
					v2Samples = np.linspace(V2[k]+ sampleDelta, V2[k+1]- sampleDelta,4)
					v3Samples = np.linspace(V3[l]+ sampleDelta, V3[l+1]- sampleDelta,4)

					for v0 in v0Samples:
						for v1 in v1Samples:
							for v2 in v2Samples:
								for v3 in v3Samples:
									v = np.array([v0, v1, v2, v3])
									inum = model.oscNum(v)[2]
									if np.less(inum, INum[:,0]).any() or np.greater(inum, INum[:,1]).any():
										print ("oops interval weird for ", v, " in interval ", vInterval)
										print ("inum ", inum, " INum ", INum)
										exit()
									



