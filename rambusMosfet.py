import numpy as np
import lpUtils
from cvxopt import matrix,solvers
from scipy.spatial import ConvexHull
import circuit

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
			transistorList.append(circuit.Mosfet(lenV, fwdInd, i, nfet, s0*g_fwd))
			transistorList.append(circuit.Mosfet(lenV+1, fwdInd, i, pfet, 2.0*s0*g_fwd))
			transistorList.append(circuit.Mosfet(lenV, ccInd, i, nfet, s0*g_cc))
			transistorList.append(circuit.Mosfet(lenV+1, ccInd, i, pfet, 2.0*s0*g_cc))

		self.c = circuit.Circuit(transistorList)


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


