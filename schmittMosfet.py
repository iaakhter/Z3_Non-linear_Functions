import numpy as np
import lpUtils
from cvxopt import matrix,solvers
from scipy.spatial import ConvexHull
import transistorMosfet as tMosfet
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

		nfet = circuit.MosfetModel('nfet', self.Vtn, self.Kn, "default")
		pfet = circuit.MosfetModel('pfet', self.Vtp, self.Kp, "default")

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

		self.boundMap = []
		for i in range(3):
			self.boundMap.append({0:[0.0,self.Vdd/2.0],1:[self.Vdd/2.0,self.Vdd]})

		self.solver = None

	def f(self,V):
		myV = [0.0, self.Vdd, self.inputVoltage] + [x for x in V]
		'''intervalVal = False
		for i in range(V.shape[0]):
			if circuit.interval_p(V[i]):
				intervalVal = True
				break
		VForSchmitt = None
		if intervalVal:
			VForSchmitt = np.array([[0.0, 0.0], 
									[self.Vdd, self.Vdd], 
									[self.inputVoltage, self.inputVoltage], 
									V[0,:], 
									V[1,:], 
									V[2,:]])
		else:
			VForSchmitt = np.array([0.0, self.Vdd, self.inputVoltage, V[0], V[1], V[2]])'''
		funcVal = self.c.f(myV)
		return funcVal[3:]

	def oscNum(self,V):
		return [None, None, self.f(V)]

	def jacobian(self,V):
		#print ("calculating jacobian")
		myV = [0.0, self.Vdd, self.inputVoltage] + [x for x in V]
		#print ("V", V)
		jac = self.c.jacobian(myV)
		#print ("jac before")
		#print (jac)
		return jac[3:,3:]


	def jacobianInterval(self,bounds):
		#print ("calculating jacobian interval")
		return self.jacobian(bounds)

class SchmittMosfet:
	def __init__(self, modelParam, inputVoltage):
		# gradient of tanh -- y = tanh(modelParam*x)
		self.Vtp = modelParam[0]
		self.Vtn = modelParam[1]
		self.Vdd = modelParam[2]
		self.Kn = modelParam[3]
		self.Kp = modelParam[4]
		self.Sn = modelParam[5]
		#self.Kp = -self.Kn/2.0
		#self.Kp = -self.Kn/3.0
		self.Sp = self.Sn*2.0
		self.inputVoltage = inputVoltage
		self.xs = []
		self.tIs = []
		self.nIs = []
		
		#xs[0] is the output voltage, xs[1] is the voltage at lower node
		#xs[1] is the voltage at the upper node 
		for i in range(3):
			self.xs.append("x" + str(i))
			self.nIs.append("ni" + str(i))
		for i in range(6):
			self.tIs.append("ti" + str(i))
		self.boundMap = []
		for i in range(3):
			self.boundMap.append({0:[0.0,self.Vdd/2.0],1:[self.Vdd/2.0,self.Vdd]})

		self.tModels = []
		self.tModels.append(tMosfet.TransistorMosfet(
										modelParam = [self.Vtn, self.Vdd, self.Kn, self.Sn],
										channelType = "nFet",
										srcVar = None, gateVar = None, drainVar = self.xs[1], IVar = self.tIs[0],
										useLeakage = True))

		self.tModels.append(tMosfet.TransistorMosfet(
										modelParam = [self.Vtn, self.Vdd, self.Kn, self.Sn],
										channelType = "nFet",
										srcVar = self.xs[1], gateVar = None, drainVar = self.xs[0], IVar = self.tIs[1],
										useLeakage = True))

		self.tModels.append(tMosfet.TransistorMosfet(
										modelParam = [self.Vtn, self.Vdd, self.Kn, self.Sn],
										channelType = "nFet",
										srcVar = self.xs[1], gateVar = self.xs[0], drainVar = None, IVar = self.tIs[2],
										useLeakage = True))

		self.tModels.append(tMosfet.TransistorMosfet(
										modelParam = [self.Vtp, self.Vdd, self.Kp, self.Sp],
										channelType = "pFet",
										srcVar = None, gateVar = None, drainVar = self.xs[2], IVar = self.tIs[3],
										useLeakage = True))

		self.tModels.append(tMosfet.TransistorMosfet(
										modelParam = [self.Vtp, self.Vdd, self.Kp, self.Sp],
										channelType = "pFet",
										srcVar = self.xs[2], gateVar = None, drainVar = self.xs[0], IVar = self.tIs[4],
										useLeakage = True))

		self.tModels.append(tMosfet.TransistorMosfet(
										modelParam = [self.Vtp, self.Vdd, self.Kp, self.Sp],
										channelType = "pFet",
										srcVar = self.xs[2], gateVar = self.xs[0], drainVar = None, IVar = self.tIs[5],
										useLeakage = True))

		self.gn = self.tModels[0].g
		self.gp = self.tModels[5].g
		self.solver = None


	#V[0] is the output voltage, V[1] is the voltage at lower node
	#V[1] is the voltage at the upper node 
	def oscNum(self,V):
		transCurs = np.zeros((6))
		transCurs[0] = self.tModels[0].ids(0.0, self.inputVoltage, V[1])[0]
		transCurs[1] = self.tModels[1].ids(V[1], self.inputVoltage, V[0])[0]
		transCurs[2] = self.tModels[2].ids(V[1], V[0], self.Vdd)[0]
		transCurs[3] = self.tModels[3].ids(self.Vdd, self.inputVoltage, V[2])[0]
		transCurs[4] = self.tModels[4].ids(V[2], self.inputVoltage, V[0])[0]
		transCurs[5] = self.tModels[5].ids(V[2], V[0], 0.0)[0]
		#print ("transCurs", transCurs)
		nodeCurs = np.zeros((3))
		nodeCurs[0] = -transCurs[4] - transCurs[1]
		nodeCurs[1] = -transCurs[0] + transCurs[1] + transCurs[2]
		nodeCurs[2] = -transCurs[3] + transCurs[5] + transCurs[4]
		return [None, None, nodeCurs]

	def jacobian(self,V):
		#print ("calculating jacobian")
		#print ("V", V)
		jac = np.zeros((3,3))
		transFirDerSrc = np.zeros((6))
		transFirDerGate = np.zeros((6))
		transFirDerDrain = np.zeros((6))

		transFirDerSrc[0], transFirDerGate[0], transFirDerDrain[0] = self.tModels[0].jacobian(0.0, self.inputVoltage, V[1])
		transFirDerSrc[1], transFirDerGate[1], transFirDerDrain[1]  = self.tModels[1].jacobian(V[1], self.inputVoltage, V[0])
		transFirDerSrc[2], transFirDerGate[2], transFirDerDrain[2]  = self.tModels[2].jacobian(V[1], V[0], self.Vdd)
		transFirDerSrc[3], transFirDerGate[3], transFirDerDrain[3] = self.tModels[3].jacobian(self.Vdd, self.inputVoltage, V[2])
		transFirDerSrc[4], transFirDerGate[4], transFirDerDrain[4]  = self.tModels[4].jacobian(V[2], self.inputVoltage, V[0])
		transFirDerSrc[5], transFirDerGate[5], transFirDerDrain[5]  = self.tModels[5].jacobian(V[2], V[0], 0.0)

		#print ("transFirDerDrain[4]", transFirDerDrain[4])
		#print ("transFirDerDrain[1]", transFirDerDrain[1])
		#print ("transFirDerSrc[1]", transFirDerSrc[1])
		#print ("transFirDerSrc[4]", transFirDerSrc[4])
		#print ("transFirDerDrain[1]", transFirDerDrain[1])
		#print ("transFirDerGate[2]", transFirDerGate[2])
		#print ("transFirDerDrain[0]", transFirDerDrain[0])
		#print ("transFirDerSrc[1]", transFirDerSrc[1])
		#print ("transFirDerSrc[2]", transFirDerSrc[2])
		#print ("transFirDerDrain[3]", transFirDerDrain[3])
		#print ("transFirDerGate[5]", transFirDerGate[5])
		#print ("transFirDerDrain[4]", transFirDerDrain[4])
		#print ("transFirDerSrc[5]", transFirDerSrc[5])
		#print ("transFirDerSrc[4]", transFirDerSrc[4])
		jac[0,0] = -transFirDerDrain[4] - transFirDerDrain[1]
		jac[0,1] = -transFirDerSrc[1]
		jac[0,2] = -transFirDerSrc[4]

		jac[1,0] = transFirDerDrain[1] + transFirDerGate[2]
		jac[1,1] = -transFirDerDrain[0] + transFirDerSrc[1] + transFirDerSrc[2]
		jac[1,2] = 0.0

		jac[2,0] = transFirDerDrain[4] + transFirDerGate[5]
		jac[2,1] = 0.0
		jac[2,2] = -transFirDerDrain[3] + transFirDerSrc[4] + transFirDerSrc[5]

		#print ("jac", jac)

		return jac


	def jacobianInterval(self,bounds):
		#print ("calculating jacobian interval")
		lowerBound = bounds[:,0]
		upperBound = bounds[:,1]
		lenV = len(lowerBound)
		jacSamples = np.zeros((lenV, lenV,8))

		jacSamples[:,:,0] = self.jacobian([lowerBound[0], lowerBound[1], lowerBound[2]])
		jacSamples[:,:,1] = self.jacobian([lowerBound[0], lowerBound[1], upperBound[2]])
		jacSamples[:,:,2] = self.jacobian([lowerBound[0], upperBound[1], lowerBound[2]])
		jacSamples[:,:,3] = self.jacobian([upperBound[0], lowerBound[1], lowerBound[2]])

		jacSamples[:,:,4] = self.jacobian([lowerBound[0], upperBound[1], upperBound[2]])
		jacSamples[:,:,5] = self.jacobian([upperBound[0], lowerBound[1], upperBound[2]])
		jacSamples[:,:,6] = self.jacobian([upperBound[0], upperBound[1], lowerBound[2]])
		jacSamples[:,:,7] = self.jacobian([upperBound[0], upperBound[1], upperBound[2]])

		jac = np.zeros((lenV, lenV, 2))
		jac[:,:,0] = np.ones((lenV, lenV))*float("inf")
		jac[:,:,1] = -np.ones((lenV, lenV))*float("inf")

		for ji in range(8):
			for i in range(lenV):
				for j in range(lenV):
					jac[i,j,0] = min(jac[i,j,0], jacSamples[i,j,ji])
					jac[i,j,1] = max(jac[i,j,1], jacSamples[i,j,ji])

		#print ("jac")
		#print (jac)
		return jac

	def linearConstraints(self, hyperRectangle):
		#print ("linearConstraints hyperRectangle")
		#print (hyperRectangle)
		solvers.options["show_progress"] = False
		allConstraints = ""

		# transistor currents
		for i in range(len(self.tModels)):
			srcGateDrainHyper = [None]*3
			if i == 0:
				srcGateDrainHyper[0] = [0.0]
				srcGateDrainHyper[1] = [self.inputVoltage]
				srcGateDrainHyper[2] = [hyperRectangle[1,0], hyperRectangle[1,1]]
			if i == 1:
				srcGateDrainHyper[0] = [hyperRectangle[1,0], hyperRectangle[1,1]]
				srcGateDrainHyper[1] = [self.inputVoltage]
				srcGateDrainHyper[2] = [hyperRectangle[0,0], hyperRectangle[0,1]]
			if i == 2:
				srcGateDrainHyper[0] = [hyperRectangle[1,0], hyperRectangle[1,1]]
				srcGateDrainHyper[1] = [hyperRectangle[0,0], hyperRectangle[0,1]]
				srcGateDrainHyper[2] = [self.Vdd]

			if i == 3:
				srcGateDrainHyper[0] = [self.Vdd]
				srcGateDrainHyper[1] = [self.inputVoltage]
				srcGateDrainHyper[2] = [hyperRectangle[2,0], hyperRectangle[2,1]]
			if i == 4:
				srcGateDrainHyper[0] = [hyperRectangle[2,0], hyperRectangle[2,1]]
				srcGateDrainHyper[1] = [self.inputVoltage]
				srcGateDrainHyper[2] = [hyperRectangle[0,0], hyperRectangle[0,1]]
			if i == 5:
				srcGateDrainHyper[0] = [hyperRectangle[2,0], hyperRectangle[2,1]]
				srcGateDrainHyper[1] = [hyperRectangle[0,0], hyperRectangle[0,1]]
				srcGateDrainHyper[2] = [0.0]


			allConstraints += self.tModels[i].linearConstraints(srcGateDrainHyper)
			
		allConstraints += "1 " + self.nIs[0] + " + " + "1 "+self.tIs[4] + " + " + "1 " + self.tIs[1] + " >= 0\n"
		allConstraints += "1 " + self.nIs[0] + " + " + "1 "+self.tIs[4] + " + " + "1 " + self.tIs[1] + " <= 0\n"

		allConstraints += "1 " + self.nIs[1] + " + " + "1 "+self.tIs[0] + " + " + "-1 " + self.tIs[1] + " + " + "-1 " + self.tIs[2]+" >= 0\n"
		allConstraints += "1 " + self.nIs[1] + " + " + "1 "+self.tIs[0] + " + " + "-1 " + self.tIs[1] + " + " + "-1 " + self.tIs[2]+" <= 0\n"

		allConstraints += "1 " + self.nIs[2] + " + " + "1 "+self.tIs[3] + " + " + "-1 " + self.tIs[5] + " + " + "-1 " + self.tIs[4]+" >= 0\n"
		allConstraints += "1 " + self.nIs[2] + " + " + "1 "+self.tIs[3] + " + " + "-1 " + self.tIs[5] + " + " + "-1 " + self.tIs[4]+" <= 0\n"

		for i in range(3):
			allConstraints += "1 " + self.nIs[i] + " >= 0\n"
			allConstraints += "1 " + self.nIs[i] + " <= 0\n"

		'''allConstraints += "1 " + self.xs[0] + " >= 0.5515\n"
		allConstraints += "1 " + self.xs[0] + " <= 0.5516\n"
		allConstraints += "1 " + self.xs[1] + " >= 0.1185\n"
		allConstraints += "1 " + self.xs[1] + " <= 0.1186\n"
		allConstraints += "1 " + self.xs[2] + " >= 1.4812\n"
		allConstraints += "1 " + self.xs[2] + " <= 1.4813\n"

		#allConstraints += "1 " + self.tIs[0] + " >= 0.1615\n"
		#allConstraints += "1 " + self.tIs[0] + " <= 0.1616\n"
		allConstraints += "1 " + self.tIs[1] + " >= 0.1586\n"
		allConstraints += "1 " + self.tIs[1] + " <= 0.1587\n"
		allConstraints += "1 " + self.tIs[2] + " >= 0.0029\n"
		allConstraints += "1 " + self.tIs[2] + " <= 0.0030\n"
		allConstraints += "1 " + self.tIs[3] + " >= -0.7209\n"
		allConstraints += "1 " + self.tIs[3] + " <= -0.7204\n"
		allConstraints += "1 " + self.tIs[4] + " >= -0.1587\n"
		allConstraints += "1 " + self.tIs[4] + " <= -0.1585\n"
		allConstraints += "1 " + self.tIs[5] + " >= -0.5620\n"
		allConstraints += "1 " + self.tIs[5] + " <= -0.5618\n"'''
		'''allConstraintList = allConstraints.splitlines()
		allConstraints = ""
		for i in range(len(allConstraintList)):
			#if i > 33 or i < 28:
			#if i <= 29:
			allConstraints += allConstraintList[i] + "\n"
		print ("numConstraints ", len(allConstraintList))'''
		#print ("allConstraints")
		#print (allConstraints)
		variableDict, A, B = lpUtils.constructCoeffMatrices(allConstraints)
		#print ("Amat", A)
		#print ("Bmat", B)
		newHyperRectangle = np.copy(hyperRectangle)
		feasible = True
		for i in range(len(self.xs)):
			#print ("min max ", i)
			minObjConstraint = "min 1 " + self.xs[i]
			maxObjConstraint = "max 1 " + self.xs[i]
			Cmin = lpUtils.constructObjMatrix(minObjConstraint,variableDict)
			Cmax = lpUtils.constructObjMatrix(maxObjConstraint,variableDict)
			minSol, maxSol = None, None
			try:
				minSol = solvers.lp(Cmin,A,B)
			except ValueError:
				#print ("weird constraints", allConstraints)
				pass
			try:
				maxSol = solvers.lp(Cmax,A,B)
			except ValueError:
				pass
			#print ("minSol Status", minSol["status"])
			#print ("maxSol Status", maxSol["status"])
			#print ("minSol", float(minSol["x"][variableDict[self.xs[0]]]))
			#print ("maxSol", float(maxSol["x"][variableDict[self.xs[0]]]))
			if minSol is not None and maxSol is not None:
				if (minSol["status"] == "primal infeasible"  and maxSol["status"] == "primal infeasible"):
					feasible = False
					break
				else:
					if minSol["status"] == "optimal":
						try:
							newHyperRectangle[i,0] = minSol['x'][variableDict[self.xs[i]]] - 1e-5
						except KeyError:
							pass
					if maxSol["status"] == "optimal":
						try:
							newHyperRectangle[i,1] = maxSol['x'][variableDict[self.xs[i]]] + 1e-5
						except KeyError:
							pass
			#print ("newVals", newHyperRectangle[i,:])
			if newHyperRectangle[i,1] < newHyperRectangle[i,0] or newHyperRectangle[i,0] < 0.0 or newHyperRectangle[i,1] > 1.0:
				#print ("old hyper", hyperRectangle[i,:])
				newHyperRectangle[i,:] = hyperRectangle[i,:]
				#print ("new hyper", newHyperRectangle[i,:])


		return [feasible, newHyperRectangle]






