import numpy as np
import lpUtils
from cvxopt import matrix,solvers
from scipy.spatial import ConvexHull
import circuit

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

		#nfet = circuit.MosfetModel('nfet', self.Vtn, self.Kn, "default")
		#pfet = circuit.MosfetModel('pfet', self.Vtp, self.Kp, "default")

		nfet = circuit.MosfetModel('nfet', self.Vtn, self.Kn)
		pfet = circuit.MosfetModel('pfet', self.Vtp, self.Kp)

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

		