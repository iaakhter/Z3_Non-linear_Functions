import numpy as np
import lpUtils
from cvxopt import matrix,solvers
from scipy.spatial import ConvexHull
import circuit
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
			#print ("fwdInd ", fwdInd, " ccInd ", ccInd)
			#print "hyperRectangle[fwdInd][0]", hyperRectangle[fwdInd][0], "hyperRectangle[fwdInd][1]", hyperRectangle[fwdInd][1]
			
			triangleClaimFwd = fcUtils.tanhLinearConstraints(self.modelParam, self.xs[fwdInd], self.ys[i], hyperRectangle[fwdInd,0],hyperRectangle[fwdInd,1])
			#print ("triangleClaimFwd", triangleClaimFwd)
			allConstraints += triangleClaimFwd

			triangleClaimCc = fcUtils.tanhLinearConstraints(self.modelParam, self.xs[ccInd], self.zs[i], hyperRectangle[ccInd,0],hyperRectangle[ccInd,1])
			#print ("triangleClaimCc", triangleClaimCc)
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
		numSuccessLp, numUnsuccessLp, numTotalLp = 0, 0, 0
		for i in range(lenV):
			numTotalLp += 2
			#print "min max ", i
			minObjConstraint = "min 1 " + self.xs[i]
			maxObjConstraint = "max 1 " + self.xs[i]
			Cmin = lpUtils.constructObjMatrix(minObjConstraint,variableDict)
			Cmax = lpUtils.constructObjMatrix(maxObjConstraint,variableDict)
			minSol = solvers.lp(Cmin,A,B)
			maxSol = solvers.lp(Cmax,A,B)
			if minSol["status"] == "primal infeasible" and maxSol["status"] == "primal infeasible":
				feasible = False
				numSuccessLp += 2
				break
			else:
				if minSol["status"] == "optimal":
					newHyperRectangle[i,0] = minSol['x'][variableDict[self.xs[i]]] - 1e-6
					numSuccessLp += 1
				else:
					numUnsuccessLp += 1
					#print ("min lp not optimal", minSol["status"])
					#print (allConstraints)
				if maxSol["status"] == "optimal":
					newHyperRectangle[i,1] = maxSol['x'][variableDict[self.xs[i]]] + 1e-6
					numSuccessLp += 1
				else:
					numUnsuccessLp += 1

		return [feasible, newHyperRectangle, numTotalLp, numSuccessLp, numUnsuccessLp]

class RambusMosfetMark:
	def __init__(self, modelParam, g_cc, g_fwd, numStages):
		# gradient of tanh -- y = tanh(modelParam*x)
		self.Vtp = modelParam[0]
		self.Vtn = modelParam[1]
		self.Vdd = modelParam[2]
		self.Kn = modelParam[3]
		self.Kp = modelParam[4]
		#self.Kp = -self.Kn/2.0
		#self.Kp = -self.Kn/3.0
		s0 = 3.0
		self.g_cc = g_cc
		self.g_fwd = g_fwd
		self.numStages = numStages
		lenV = numStages*2

		nfet = circuit.MosfetModel('nfet', self.Vtn, self.Kn)
		pfet = circuit.MosfetModel('pfet', self.Vtp, self.Kp)

		# for 4 stage oscillator for example
		# V is like this for Mark's model V = [V0, V1, V2, V3, V4, V5, V6, V7, src, Vdd]

		transistorList = []
		for i in range(lenV):
			fwdInd = (i-1)%(lenV)
			ccInd = (i+lenV//2)%(lenV)
			#if i == 3 or fwdInd == 3 or ccInd == 3:
			#	print ("faulty node involved in transistors", len(transistorList),
			#		len(transistorList) + 1, len(transistorList) + 2, len(transistorList)+3)
			transistorList.append(circuit.Mosfet(lenV, fwdInd, i, nfet, s0*g_fwd))
			transistorList.append(circuit.Mosfet(lenV+1, fwdInd, i, pfet, 2.0*s0*g_fwd))
			transistorList.append(circuit.Mosfet(lenV, ccInd, i, nfet, s0*g_cc))
			transistorList.append(circuit.Mosfet(lenV+1, ccInd, i, pfet, 2.0*s0*g_cc))

		self.c = circuit.Circuit(transistorList)
		#self.c = circuit.TransistorCircuit(transistorList)


		self.bounds = []
		for i in range(numStages*2):
			self.bounds.append([0.0, self.Vdd])


	def f(self,V):
		myV = [x for x in V] + [0.0, self.Vdd]
		# print 'rambusMosfetMark.f: myV = ' + str(myV)
		funcVal = self.c.f(myV)
		# print 'rambusMosfetMark.f: funcVal = ' + str(funcVal)
		return funcVal[:-2]

	def jacobian(self,V):
		myV = [x for x in V] + [0.0, self.Vdd]
		myJac = self.c.jacobian(myV)
		return np.array(myJac[:-2,:-2])	

	def linearConstraints(self, hyperRectangle):
		lenV = len(hyperRectangle)
		cHyper = [x for x in hyperRectangle] + [0.0, self.Vdd]
		[feasible, newHyper, numTotalLp, numSuccessLp, numUnsuccessLp] = self.c.linearConstraints(cHyper, [lenV, lenV + 1])
		newHyper = newHyper[:-2]
		newHyperMat = np.zeros((lenV,2))
		for i in range(lenV):
			newHyperMat[i,:] = [newHyper[i][0], newHyper[i][1]]
		return [feasible, newHyperMat, numTotalLp, numSuccessLp, numUnsuccessLp]


class RambusStMosfet:
	def __init__(self, modelParam, g_cc, g_fwd, numStages):
		# gradient of tanh -- y = tanh(modelParam*x)
		
		self.Vdd = modelParam[0]
		self.g_cc = g_cc
		self.g_fwd = g_fwd
		self.numStages = numStages
		lenV = numStages*2

		nfet = circuit.MosfetModel('nfet')
		pfet = circuit.MosfetModel('pfet')

		# for 4 stage oscillator for example
		# V is like this for Mark's model V = [V0, V1, V2, V3, V4, V5, V6, V7, src, Vdd]

		transistorList = []
		for i in range(lenV):
			fwdInd = (i-1)%(lenV)
			ccInd = (i+lenV//2)%(lenV)
			#if i == 3 or fwdInd == 3 or ccInd == 3:
			#	print ("faulty node involved in transistors", len(transistorList),
			#		len(transistorList) + 1, len(transistorList) + 2, len(transistorList)+3)
			transistorList.append(circuit.StMosfet(lenV, fwdInd, i, nfet, g_fwd))
			transistorList.append(circuit.StMosfet(lenV+1, fwdInd, i, pfet, g_fwd))
			transistorList.append(circuit.StMosfet(lenV, ccInd, i, nfet, g_cc))
			transistorList.append(circuit.StMosfet(lenV+1, ccInd, i, pfet, g_cc))

		self.c = circuit.Circuit(transistorList)
		#self.c = circuit.TransistorCircuit(transistorList)


		self.bounds = []
		for i in range(numStages*2):
			self.bounds.append([0.0, self.Vdd])


	def f(self,V):
		myV = [x for x in V] + [0.0, self.Vdd]
		# print 'rambusMosfetMark.f: myV = ' + str(myV)
		funcVal = self.c.f(myV)
		# print 'rambusMosfetMark.f: funcVal = ' + str(funcVal)
		return funcVal[:-2]

	def jacobian(self,V):
		myV = [x for x in V] + [0.0, self.Vdd]
		myJac = self.c.jacobian(myV)
		return np.array(myJac[:-2,:-2])	


class SchmittMosfetMark:
	def __init__(self, modelParam, inputVoltage):
		self.Vtp = modelParam[0]
		self.Vtn = modelParam[1]
		self.Vdd = modelParam[2]
		self.Kn = modelParam[3]
		self.Kp = modelParam[4]
		s0 = 3.0
		#self.Sn = modelParam[5]
		#self.Sp = self.Sn*2.0
		self.inputVoltage = inputVoltage

		nfet = circuit.MosfetModel('nfet', self.Vtn, self.Kn, "default")
		pfet = circuit.MosfetModel('pfet', self.Vtp, self.Kp, "default")

		#nfet = circuit.MosfetModel('nfet', self.Vtn, self.Kn)
		#pfet = circuit.MosfetModel('pfet', self.Vtp, self.Kp)

		# with the voltage array containing [grnd, Vdd, input, X[0], X[1], X[2]]
		# where X[0] is the output voltage and X[1] is the voltage at node with 
		# nfets and X[2] is the voltage at node with pfets

		# src, gate, drain = grnd, input, X[1]
		m0 = circuit.Mosfet(0, 2, 4, nfet, s0)

		# src, gate, drain = X[1], input, X[0]
		m1 = circuit.Mosfet(4, 2, 3, nfet, s0)

		# src, gate, drain = X[1], X[0], Vdd
		m2 = circuit.Mosfet(4, 3, 1, nfet, s0)

		# src, gate, drain = Vdd, input, X[2]
		m3 = circuit.Mosfet(1, 2, 5, pfet, s0*2.0)

		# src, gate, drain = X[2], in, X[0]
		m4 = circuit.Mosfet(5, 2, 3, pfet, s0*2.0)

		# src, gate, drain = X[2], X[0], grnd
		m5 = circuit.Mosfet(5, 3, 0, pfet, s0*2.0)

		self.c = circuit.Circuit([m0, m1, m2, m3, m4, m5])

		self.bounds = []
		for i in range(3):
			self.bounds.append([0.0,self.Vdd])


	def f(self,V):
		myV = [0.0, self.Vdd, self.inputVoltage] + [x for x in V]
		funcVal = self.c.f(myV)
		return funcVal[3:]


	def jacobian(self,V):
		#print ("calculating jacobian")
		myV = [0.0, self.Vdd, self.inputVoltage] + [x for x in V]
		#print ("V", V)
		jac = self.c.jacobian(myV)
		#print ("jac before")
		#print (jac)
		return jac[3:,3:]


	def linearConstraints(self, hyperRectangle):
		lenV = len(hyperRectangle)
		cHyper = [0.0, self.Vdd, self.inputVoltage] + [x for x in hyperRectangle]
		[feasible, newHyper, numTotalLp, numSuccessLp, numUnsuccessLp]  = self.c.linearConstraints(cHyper, [0, 1])
		newHyper = newHyper[3:]
		newHyperMat = np.zeros((lenV,2))
		for i in range(lenV):
			newHyperMat[i,:] = [newHyper[i][0], newHyper[i][1]]
		return [feasible, newHyperMat, numTotalLp, numSuccessLp, numUnsuccessLp]

		






