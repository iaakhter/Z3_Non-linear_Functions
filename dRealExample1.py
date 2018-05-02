from dreal.symbolic import Variable, logical_and, logical_or, asin, sin, cos
from dreal.symbolic import logical_not
from dreal.symbolic import forall
from dreal.api import CheckSatisfiability, Minimize
import math
import time
import numpy as np

def exampleFun(numSolutions = "all"):
	start = time.time()
	epsilon = 1e-12
	lenV = 1
	x = Variable('x')

	allSolutions = []
	while True:
		if numSolutions != "all" and len(allSolutions) == numSolutions:
			break
		allConstraints = []	
		allConstraints.append(2*x*asin(cos(0.797)*sin(math.pi/x)) - 0.0331*x >= 2*math.pi - 2.097)
		allConstraints.append(x >= -100.0)
		allConstraints.append(x <= -0.1)
		excludingConstraints = []
		for solution in allSolutions:
			singleExcludingConstraints = []
			singleExcludingConstraints.append(x <= solution[0][0])
			singleExcludingConstraints.append(x >= solution[0][1])
			excludingConstraints.append(singleExcludingConstraints)
		
		#print ("allConstraints")
		#print (allConstraints)
		#print ("numConstraints", len(allConstraints))
		f_sat = logical_and(*allConstraints)
		if len(excludingConstraints) > 0:
			for constraints in excludingConstraints:
				f_sat = logical_and(f_sat, logical_or(*constraints))
		
		#print ("f_sat")
		#print (f_sat)
		result = CheckSatisfiability(f_sat, epsilon)
		#print (result)
		if result is None:
			break
		hyper = np.zeros((1,2))
		hyper[0,:] = [result[x].lb() - 2*epsilon, result[x].ub() + 2*epsilon]

		print ("hyper", hyper)
		allSolutions.append(hyper)

		print ("num solutions found", len(allSolutions))


	'''constraints = []
	x = Variable('x')
	#constraints.append(x >= 0.0)
	#constraints.append(x <= 64.0)
	constraints.append(2*math.pi - 2*x*asin(cos(0.797)*sin(math.pi/x)) == 2.097 - 0.0331*x)
	f_sat = logical_and(*constraints)
	result = CheckSatisfiability(f_sat, epsilon)
	print (result)'''
	end = time.time()
	print ("time taken", end - start)

exampleFun()