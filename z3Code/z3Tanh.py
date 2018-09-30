# @author: Itrat Ahmed Akhter
# Find dc equilibrium points for rambus oscillator with the inverter
# modeled as a tanh function with Z3

from z3 import *
import numpy as np
import time



# Return list of constraints approximating tanh with piecewise polynomial function
# a <= 0 -- gain for tanh
# VinVar is a Z3 variable representing input voltage
# VoutVar is a Z3 variable representing output voltage
def tanhCurrent(a, VinVar, VoutVar):
	constraints = []
	constraints.append(Or(And(VinVar >= -2.0/a, VoutVar == -1-(a*VinVar)**5),
							And(VinVar < -2.0/a, VinVar > 2.0/a, VoutVar == -(13/256.0)*(a*VinVar)**3 + (11/16.0)*(a*VinVar)),
							And(VinVar <= 2.0/a, VoutVar == 1-(a*VinVar)**5)))
	return constraints

# Try and find dc equilibrium points for rambus oscillator with the inverter
# being modeled by tanh function
# numStages indicates the number of stages in the rambus oscillator
# g_cc is the strength of the cross coupled inverter as compared to that of forward (g_fwd = 1.0)
# numSolutions indicates the number of solutions we want Z3 to find
# a is the gain of the inverter. Should be negative
def rambusOscillatorTanh(numStages, g_cc = 0.5, numSolutions = "all", a = -5.0):
	start = time.time()
	g_fwd = 1.0
	lenV = numStages*2
	vs = RealVector("v", lenV)
	vfwds = RealVector("vfwd", lenV)
	vccs = RealVector("vcc", lenV)

	allConstraints = []
		
	# Store rambus oscillator constraints
	for i in range(lenV):
		allConstraints.append(vs[i] >= -1)
		allConstraints.append(vs[i] <= 1)
		fwdInd = (i-1)%lenV
		ccInd = (i+lenV//2)%lenV
		allConstraints += tanhCurrent(a, vs[fwdInd], vfwds[i])
		allConstraints += tanhCurrent(a, vs[ccInd], vccs[i])
		allConstraints.append(g_fwd*vfwds[i] + (-g_fwd-g_cc)*vs[i] + g_cc*vccs[i] == 0)

	s = Solver()
	allSolutions = []
	while True:
		if numSolutions != "all" and len(allSolutions) == numSolutions:
			break

		# Store constraints pruning search space so that
		# old solutions are not considered
		excludingConstraints = []
		for solution in allSolutions:
			singleExcludingConstraints = []
			for i in range(lenV):
				singleExcludingConstraints.append(vs[i] != solution[i])
			excludingConstraints.append(singleExcludingConstraints)
		
		#print ("numConstraints", len(allConstraints))
		# Add all the rambus oscillator constraints
		f_sat = And(*allConstraints)
		# Add constraints so that old solutions are not considered
		if len(excludingConstraints) > 0:
			for constraints in excludingConstraints:
				f_sat = And(f_sat, Or(*constraints))
		
		# Add the constraints to Z3 with a push and pop operation
		#print ("f_sat")
		#print (f_sat)
		s.push()
		s.add(f_sat)
		#print ("s")
		#print (s)
		result = s.check()
		#print (result)
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
		
		
	print ("all solutions")
	for solution in allSolutions:
		print (solution)
	
	print ("num solutions", len(allSolutions))
	end = time.time()
	print ("time taken", end - start)
	return allSolutions


# Try and find dc equilibrium points for inverter modeled by a tanh function
# for a specific input voltage
# a indicates the gain by the tanh function
# numSolutions indicates the number of solutions we want Z3 to find
def inverterTanh(inputVoltage, a, numSolutions = "all"):
	start = time.time()

	outputVolt = Real("outputVolt")
	vRes = Real("vRes")

	allConstraints = []	
	allConstraints.append(outputVolt >= -1.0)
	allConstraints.append(outputVolt <= 1.0)
	tanhVal = np.tanh(a*inputVoltage)
	allConstraints.append(vRes == tanhVal)
	allConstraints.append(vRes - outputVolt == 0.0)
	
	s = Solver()
	allSolutions = []
	while True:
		if numSolutions != "all" and len(allSolutions) == numSolutions:
			break
		
		# Store constraints pruning search space so that
		# old solutions are not considered
		excludingConstraints = []
		for solution in allSolutions:
			singleExcludingConstraints = []
			singleExcludingConstraints.append(outputVolt <= solution[0][0])
			singleExcludingConstraints.append(outputVolt >= solution[0][1])
			excludingConstraints.append(singleExcludingConstraints)
		
		#print ("allConstraints")
		#print (allConstraints)
		#print ("numConstraints", len(allConstraints))
		f_sat = And(*allConstraints)
		if len(excludingConstraints) > 0:
			for constraints in excludingConstraints:
				f_sat = And(f_sat, Or(*constraints))
		
		# Add the constraints to Z3 with a push and pop operation
		#print ("f_sat")
		#print (f_sat)
		s.push()
		s.add(f_sat)
		#print ("s")
		#print (s)
		result = s.check()
		#print (result)
		if result != sat:
			break
		
		m = s.model()
		#print ("m", m)
		sol = None
		hyper = np.zeros((1,2))
		for d in m.decls():
			dName = str(d.name())
			if dName == "outputVolt":
				sol = float(Fraction(str(m[d])))
				'''if str(m[d])[-1] == "?":
					sol = float(str(m[d])[:-1])
				else:
					sol = float(str(m[d]))'''

		hyper[0,:] = np.array([sol - 0.1, sol + 0.1])
		#print ("sol", sol)
		s.pop()
		
		allSolutions.append(hyper)

		print ("num solutions found", len(allSolutions))

	
	'''print ("all solutions")
	for solution in allSolutions:
		print (solution)'''

	end = time.time()
	print ("time taken", end - start)
	return allSolutions





if __name__ == "__main__":
	#rambusOscillatorMosfet(Vtp = -0.25, Vtn = 0.25, Vdd = 1.0, Kn = 1.0, Kp = -0.5, Sn = 1.0, numStages = 2, numSolutions = 1, g_cc = 0.5)
	#rambusOscillatorTanh(a = -5.0, numStages = 2, numSolutions = "all", g_cc = 0.5)
	allSolutions = inverterTanh(-1.0, -5.0)
	print ("allSolutions")
	for solution in allSolutions:
		for i in range(solution.shape[0]):
			print (solution[i,0], solution[i,1])
