import numpy as np
import lpUtils
from cvxopt import matrix,solvers
import funCompUtils as fcUtils
from intervalBasics import *

class RambusTanh:
	def __init__(self, modelParam, g_cc, g_fwd, numStages):
		# gradient of tanh -- y = tanh(modelParam*x)
		self.modelParam = modelParam
		self.g_cc = g_cc
		self.g_fwd = g_fwd
		self.numStages = numStages
		self.xs = []
		self.ys = []
		self.zs = []
		for i in range(numStages*2):
			self.xs.append("x" + str(i))
			self.ys.append("y" + str(i))
			self.zs.append("z" + str(i))

		self.bounds = []
		for i in range(numStages*2):
			self.bounds.append([-1.0,1.0])

	def f(self, V):
		intervalVal = any([interval_p(x) for x in V])
		lenV = len(V)
		
		if intervalVal:
			fVal = np.zeros((lenV,2))
		else:
			fVal = np.zeros((lenV))

		for i in range(lenV):
			fwdInd = (i-1)%lenV
			ccInd = (i + lenV//2) % lenV
			tanhFwd = fcUtils.tanhFun(V[fwdInd], self.modelParam)
			tanhCc = fcUtils.tanhFun(V[ccInd], self.modelParam)
			fwdTerm = interval_sub(tanhFwd, V[i])
			ccTerm = interval_sub(tanhCc, V[i])
			fVal[i] = interval_add(interval_mult(self.g_fwd, fwdTerm),interval_mult(self.g_cc, ccTerm))

		return fVal


	'''Get jacobian of rambus oscillator at V
	'''
	def jacobian(self,V):
		lenV = len(V)
		intervalVal = any([interval_p(x) for x in V])

		if intervalVal:
			jac = np.zeros((lenV, lenV, 2))
		else:
			jac = np.zeros((lenV, lenV))
		for i in range(lenV):
			if intervalVal:
				jac[i,i] = interval_fix(-(self.g_fwd + self.g_cc))
			else:
				jac[i,i] = -(self.g_fwd + self.g_cc)
			jac[i,(i-1)%lenV] = interval_mult(self.g_fwd, fcUtils.tanhFunder(V[(i-1)%lenV], self.modelParam))
			jac[i,(i + lenV//2) % lenV] = interval_mult(self.g_cc, fcUtils.tanhFunder(V[(i + lenV//2) % lenV], self.modelParam))

		return jac



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
			
			triangleClaimFwd = fcUtils.tanhLinearConstraints(self.modelParam, self.xs[fwdInd], self.ys[i], hyperRectangle[fwdInd,0],hyperRectangle[fwdInd,1])
			allConstraints += triangleClaimFwd

			triangleClaimCc = fcUtils.tanhLinearConstraints(self.modelParam, self.xs[ccInd], self.zs[i], hyperRectangle[ccInd,0],hyperRectangle[ccInd,1])
			allConstraints += triangleClaimCc
				
			allConstraints += str(self.g_fwd) + " " + self.ys[i] + " + " + str(-self.g_fwd-self.g_cc) + \
			" " + self.xs[i] + " + " + str(self.g_cc) + " "  + self.zs[i] + " >= 0.0\n"
			allConstraints += str(self.g_fwd) + " " + self.ys[i] + " + " + str(-self.g_fwd-self.g_cc) + \
			" " + self.xs[i] + " + " + str(self.g_cc) + " "  + self.zs[i] + " <= 0.0\n"
		

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
	model = RambusTanh(modelParam = -5, g_cc = 0.5, g_fwd = 1.0, numStages = 2)
	print (model.jacobian(np.array([[-0.3, 0.0],[0.0, 0.3],[-0.3, 0.0],[0.0, 0.3]])))
	'''V0 = np.linspace(-1.0, 1.0, 10)
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
										exit()'''
									



