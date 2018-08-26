import numpy as np
import lpUtils
from cvxopt import matrix,solvers
from scipy.spatial import ConvexHull
import circuit

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


