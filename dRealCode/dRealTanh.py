# @author Itrat Ahmed Akhter
# Find dc equilibrium points for rambus oscillator with the inverter
# modeled as a tanh function with dReal

from dreal.symbolic import Variable, logical_and, logical_or, tanh
from dreal.symbolic import logical_not
from dreal.symbolic import forall
from dreal.api import CheckSatisfiability, Minimize
import time
import numpy as np


# Try and find dc equilibrium points for rambus oscillator with the inverter
# being modeled by tanh function
# numStages indicates the number of stages in the rambus oscillator
# g_cc is the strength of the cross coupled inverter as compared to that of forward (g_fwd = 1.0)
# numSolutions indicates the number of solutions we want dReal to find
# a is the gain of the inverter. Should be negative
def rambusOscillatorTanh(numStages, g_cc = 0.5, numSolutions = "all", a= -5.0):
	epsilon = 1e-14
	start = time.time()
	g_fwd = 1.0
	lenV = numStages*2
	vs = []
	vfwds = []
	vccs = []
	for i in range(lenV):
		vs.append(Variable("v" + str(i)))
		vfwds.append(Variable("vfwd" + str(i)))
		vccs.append(Variable("vcc" + str(i)))

	allConstraints = []
		
	# Store rambus oscillator constraints
	for i in range(lenV):
		allConstraints.append(vs[i] >= -1)
		allConstraints.append(vs[i] <= 1)
		fwdInd = (i-1)%lenV
		ccInd = (i+lenV//2)%lenV
		allConstraints.append(vfwds[i] == tanh(a*vs[fwdInd]))
		allConstraints.append(vccs[i] == tanh(a*vs[ccInd]))
		allConstraints.append(g_fwd*vfwds[i] + (-g_fwd-g_cc)*vs[i] + g_cc*vccs[i] == 0)

	allSolutions = []
	while True:
		if numSolutions != "all" and len(allSolutions) == numSolutions:
			break

		# Store constraints pruning search space so that
		# old hyperrectangles are not considered
		excludingConstraints = []
		for solution in allSolutions:
			singleExcludingConstraints = []
			for i in range(lenV):
				singleExcludingConstraints.append(vs[i] <= solution[i][0])
				singleExcludingConstraints.append(vs[i] >= solution[i][1])
			excludingConstraints.append(singleExcludingConstraints)
		
		# Add all the rambus oscillator constraints
		f_sat = logical_and(*allConstraints)
		# Add constraints so that old hyperrectangles are not considered
		if len(excludingConstraints) > 0:
			for constraints in excludingConstraints:
				f_sat = logical_and(f_sat, logical_or(*constraints))
		
		#print ("f_sat")
		#print (f_sat)
		result = CheckSatisfiability(f_sat, epsilon)
		#print (result)
		if result is None:
			break
		hyper = np.zeros((lenV,2))
		for i in range(lenV):
			hyper[i,:] = [result[vs[i]].lb() - 2*epsilon, result[vs[i]].ub() + 2*epsilon]

		#print ("hyper", hyper)
		allSolutions.append(hyper)

		print ("num solutions found", len(allSolutions))
		

	end = time.time()
	print ("time taken", end - start)
	return allSolutions

