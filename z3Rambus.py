from z3 import *
import rambusUtils as rUtils
import numpy as np
from tanhModel import TanhModel
from mosfetModel import MosfetModel
import intervalUtils
import time

def mosfetCurrent(Vtp, Vtn, Vdd, Kn, Kp, Sn, Sp, IVarN, IVarP, VinVar, VoutVar):
	constraints = []

	constraints.append(Or(And(VinVar <= Vtn, IVarN == 0.0),
							And(Vtn <= VinVar, VinVar <= VoutVar + Vtn, IVarN == Sn*(Kn/2.0)*(VinVar - Vtn)*(VinVar - Vtn)),
							And(Vtn <= VinVar, VinVar >= VoutVar + Vtn, IVarN == Sn*(Kn)*(VinVar - Vtn - VoutVar/2.0)*VoutVar)))

	constraints.append(Or(And(VinVar - Vtp >= Vdd, IVarP == 0.0),
							And(VinVar - Vtp <= Vdd, VoutVar <= VinVar - Vtp, IVarP == Sp*(Kp/2.0)*(VinVar - Vtp - Vdd)*(VinVar - Vtp - Vdd)),
							And(VinVar - Vtp <= Vdd, VoutVar >= VinVar - Vtp, IVarP == Sp*Kp*((VinVar - Vtp - Vdd) - (VoutVar - Vdd)/2.0)*(VoutVar - Vdd))))
	

	return constraints


 #a <= 0
def tanhCurrent(a, VinVar, VoutVar):
	constraints = []
	constraints.append(Or(And(VinVar >= -2.0/a, VoutVar == -1-(a*VinVar)**5),
							And(VinVar < -2.0/a, VinVar > 2.0/a, VoutVar == -(13/256.0)*(a*VinVar)**3 + (11/16.0)*(a*VinVar)),
							And(VinVar <= 2.0/a, VoutVar == 1-(a*VinVar)**5)))
	return constraints

def rambusOscillatorTanh(a, numStages, numSolutions = "all", g_cc = 0.5):
	model = TanhModel(modelParam = a, g_cc = g_cc, g_fwd = 1.0, numStages=numStages)
	start = time.time()
	g_fwd = 1.0
	lenV = numStages*2
	vs = RealVector("v", lenV)
	vfwds = RealVector("vfwd", lenV)
	vccs = RealVector("vcc", lenV)

	s = Solver()
	allSolutions = []
	while True:
		if numSolutions != "all" and len(allSolutions) == numSolutions:
			break
		allConstraints = []
		for i in range(lenV):
			allConstraints.append(vs[i] >= -1)
			allConstraints.append(vs[i] <= 1)
			fwdInd = (i-1)%lenV
			ccInd = (i+lenV//2)%lenV
			allConstraints += tanhCurrent(a, vs[fwdInd], vfwds[i])
			allConstraints += tanhCurrent(a, vs[ccInd], vccs[i])
			allConstraints.append(g_fwd*vfwds[i] + (-g_fwd-g_cc)*vs[i] + g_cc*vccs[i] == 0)

		excludingConstraints = []
		for solution in allSolutions:
			singleExcludingConstraints = []
			for i in range(lenV):
				singleExcludingConstraints.append(vs[i] != solution[i])
			excludingConstraints.append(singleExcludingConstraints)
		
		f_sat = And(*allConstraints)
		if len(excludingConstraints) > 0:
			for constraints in excludingConstraints:
				f_sat = And(f_sat, Or(*constraints))
		
		#print ("f_sat")
		#print (f_sat)
		s.push()
		s.add(f_sat)
		#print ("s")
		#print (s)
		result = s.check()
		print (result)
		if result != sat:
			break
		
		m = s.model()
		sol = np.zeros((lenV))
		for d in m.decls():
			dName = str(d.name())
			firstLetter = dName[0]
			if (dName[0] == "v" and dName[1] == "_"):
				index = int(dName[len(dName) - 1])
				val = float(Fraction(str(m[d])))
				sol[index] = val

		print ("sol", sol)
		s.pop()

		allSolutions.append(sol)
		
		print ("num solutions found", len(allSolutions))
		'''if len(allSolutions) == 3:
			break'''
		
	print ("all solutions")
	for solution in allSolutions:
		print (solution)
	
	print ("num solutions", len(allSolutions))
	end = time.time()
	print ("time taken", end - start)




def rambusOscillatorMosfet(Vtp, Vtn, Vdd, Kn, Kp, Sn, numStages, numSolutions = "all", g_cc = 0.5):
	start = time.time()
	print ("Vtp", Vtp, "Vtn", Vtn, "Vdd", Vdd, "Kn", Kn, "Kp", Kp, "Sn", Sn)
	g_fwd = 1.0
	Sp = Sn *2.0
	model = MosfetModel(modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn], g_cc = g_cc, g_fwd = g_fwd, numStages = numStages)
	boundMap = model.boundMap
	lenV = numStages*2

	vs = RealVector('v', lenV)
	ifwdNs = RealVector('ifwdN', lenV)
	ifwdPs = RealVector('ifwdP', lenV)
	iccNs = RealVector('iccN', lenV)
	iccPs = RealVector('iccP', lenV)

	s = Solver()

	allSolutions = []
	while True:
		if numSolutions != "all" and len(allSolutions) == numSolutions:
			break
		allConstraints = []	
		for i in range(lenV):
			allConstraints.append(vs[i] >= boundMap[i][0][0])
			allConstraints.append(vs[i] <= boundMap[i][1][1])
			allConstraints.append(g_fwd*(-ifwdNs[i]-ifwdPs[i]) + g_cc*(-iccNs[i]-iccPs[i]) == 0)
			fwdInd = (i-1)%lenV
			ccInd = (i+lenV//2)%lenV
			fwdConstraints = mosfetCurrent(Vtp = Vtp, Vtn = Vtn, Vdd = Vdd, Kn = Kn, Kp = Kp, Sn = Sn, Sp = Sp, IVarN = ifwdNs[i], IVarP = ifwdPs[i], VinVar = vs[fwdInd], VoutVar = vs[i])
			ccConstraints = mosfetCurrent(Vtp = Vtp, Vtn = Vtn, Vdd = Vdd, Kn = Kn, Kp = Kp, Sn = Sn, Sp = Sp, IVarN = iccNs[i], IVarP = iccPs[i], VinVar = vs[ccInd], VoutVar = vs[i])
			allConstraints += fwdConstraints + ccConstraints
		
		excludingConstraints = []
		for solution in allSolutions:
			singleExcludingConstraints = []
			for i in range(lenV):
				singleExcludingConstraints.append(vs[i] != solution[i])
			excludingConstraints.append(singleExcludingConstraints)
		
		#print ("allConstraints")
		#print (allConstraints)
		#print ("numConstraints", len(allConstraints))
		f_sat = And(*allConstraints)
		if len(excludingConstraints) > 0:
			for constraints in excludingConstraints:
				f_sat = And(f_sat, Or(*constraints))
		
		#print ("f_sat")
		#print (f_sat)
		s.push()
		s.add(f_sat)
		print ("s")
		print (s)
		result = s.check()
		print (result)
		if result != sat:
			break
		
		m = s.model()
		sol = np.zeros((lenV))
		for d in m.decls():
			dName = str(d.name())
			firstLetter = dName[0]
			if (dName[0] == "v" and dName[1] == "_"):
				index = int(dName[len(dName) - 1])
				val = float(Fraction(str(m[d])))
				sol[index] = val

		print ("sol", sol)
		s.pop()
		
		allSolutions.append(sol)

		print ("num solutions found", len(allSolutions))
		'''if len(allSolutions) == 1:
			break'''

	
	print ("all solutions")
	for solution in allSolutions:
		print (solution)

	end = time.time()
	print ("time taken", end - start)

#rambusOscillatorMosfet(Vtp = -0.25, Vtn = 0.25, Vdd = 1.0, Kn = 1.0, Kp = -0.5, Sn = 1.0, numStages = 2, numSolutions = 1, g_cc = 0.5)
rambusOscillatorTanh(a = -5.0, numStages = 2, numSolutions = "all", g_cc = 0.5)
