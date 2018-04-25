from dreal.symbolic import Variable, logical_and, logical_or, tanh
from dreal.symbolic import logical_not
from dreal.symbolic import forall
from dreal.api import CheckSatisfiability, Minimize
import time
import rambusUtils as rUtils
import numpy as np
from schmittMosfet import SchmittMosfet
import intervalUtils


def nFet(Vtn, Vdd, Kn, Sn, gn, src, gate, drain, tI):
	constraints = []
	if type(src) == Variable or type(gate) == Variable:
		constraints.append(logical_or(logical_and(src > drain, gate - drain <= Vtn, tI == -(src - drain)*(2 + (gate - drain - Vtn)/Vdd)*(gn*1e-4)),
										logical_and(src > drain, gate - drain >= Vtn, src - drain >= gate - drain - Vtn, tI == -0.5*Sn*Kn*(gate - drain - Vtn)*(gate - drain - Vtn) - (src - drain)*(2 + (gate - drain - Vtn)/Vdd)*(gn*1e-4)),
										logical_and(src > drain, gate - drain >= Vtn, src - drain <= gate - drain - Vtn, tI == -Sn*Kn*(gate - drain - Vtn - (src - drain)/2.0)*(src - drain) - (src - drain)*(2 + (gate - drain - Vtn)/Vdd)*(gn*1e-4)),
										logical_and(src <= drain, gate - src <= Vtn, tI == (drain - src)*(2 + (gate - src - Vtn)/Vdd)*(gn*1e-4)),
										logical_and(src <= drain, gate - src >= Vtn, drain - src >= gate - src - Vtn, tI == 0.5*Sn*Kn*(gate - src - Vtn)*(gate - src - Vtn) + (drain - src)*(2 + (gate - src - Vtn)/Vdd)*(gn*1e-4)),
										logical_and(src <= drain, gate - src >= Vtn, drain - src <= gate - src - Vtn, tI == Sn*Kn*(gate - src - Vtn - (drain - src)/2.0)*(drain - src) + (drain - src)*(2 + (gate - src - Vtn)/Vdd)*(gn*1e-4))))
	else:
		if gate - src <= Vtn:
			constraints.append(logical_or(logical_and(src > drain, gate - drain <= Vtn, tI == -(src - drain)*(2 + (gate - drain - Vtn)/Vdd)*(gn*1e-4)),
											logical_and(src > drain, gate - drain >= Vtn, src - drain >= gate - drain - Vtn, tI == -0.5*Sn*Kn*(gate - drain - Vtn)*(gate - drain - Vtn) - (src - drain)*(2 + (gate - drain - Vtn)/Vdd)*(gn*1e-4)),
											logical_and(src > drain, gate - drain >= Vtn, src - drain <= gate - drain - Vtn, tI == -Sn*Kn*(gate - drain - Vtn - (src - drain)/2.0)*(src - drain) - (src - drain)*(2 + (gate - drain - Vtn)/Vdd)*(gn*1e-4)),
											logical_and(src <= drain, tI == (drain - src)*(2 + (gate - src - Vtn)/Vdd)*(gn*1e-4))))
		else:
			constraints.append(logical_or(logical_and(src > drain, gate - drain <= Vtn, tI == -(src - drain)*(2 + (gate - drain - Vtn)/Vdd)*(gn*1e-4)),
											logical_and(src > drain, gate - drain >= Vtn, src - drain >= gate - drain - Vtn, tI == -0.5*Sn*Kn*(gate - drain - Vtn)*(gate - drain - Vtn) - (src - drain)*(2 + (gate - drain - Vtn)/Vdd)*(gn*1e-4)),
											logical_and(src > drain, gate - drain >= Vtn, src - drain <= gate - drain - Vtn, tI == -Sn*Kn*(gate - drain - Vtn - (src - drain)/2.0)*(src - drain) - (src - drain)*(2 + (gate - drain - Vtn)/Vdd)*(gn*1e-4)),
											logical_and(src <= drain, drain - src >= gate - src - Vtn, tI == 0.5*Sn*Kn*(gate - src - Vtn)*(gate - src - Vtn) + (drain - src)*(2 + (gate - src - Vtn)/Vdd)*(gn*1e-4)),
											logical_and(src <= drain, drain - src <= gate - src - Vtn, tI == Sn*Kn*(gate - src - Vtn - (drain - src)/2.0)*(drain - src) + (drain - src)*(2 + (gate - src - Vtn)/Vdd)*(gn*1e-4))))
	
	return constraints

	return constraints
def pFet(Vtp, Vdd, Kp, Sp, gp, src, gate, drain, tI):
	constraints = []
	if type(src) == Variable or type(gate) == Variable:
		constraints.append(logical_or(logical_and(src < drain, gate - drain >= Vtp, tI == -(src - drain)*(2 - (gate - drain - Vtp)/Vdd)*(gp*1e-4)),
										logical_and(src < drain, gate - drain <= Vtp, src - drain <= gate - drain - Vtp, tI == -0.5*Sp*Kp*(gate - drain - Vtp)*(gate - drain - Vtp) - (src - drain)*(2 - (gate - drain - Vtp)/Vdd)*(gp*1e-4)),
										logical_and(src < drain, gate - drain <= Vtp, src - drain >= gate - drain - Vtp, tI == -Sp*Kp*(gate - drain - Vtp - (src - drain)/2.0)*(src - drain) - (src - drain)*(2 - (gate - drain - Vtp)/Vdd)*(gp*1e-4)),
										logical_and(src >= drain, gate - src >= Vtp, tI == (drain - src)*(2 - (gate - src - Vtp)/Vdd)*(gp*1e-4)),
										logical_and(src >= drain, gate - src <= Vtp, drain - src <= gate - src - Vtp, tI == 0.5*Sp*Kp*(gate - src - Vtp)*(gate - src - Vtp) + (drain - src)*(2 - (gate - src - Vtp)/Vdd)*(gp*1e-4)),
										logical_and(src >= drain, gate - src <= Vtp, drain - src >= gate - src - Vtp, tI == Sp*Kp*(gate - src - Vtp - (drain - src)/2.0)*(drain - src) + (drain - src)*(2 - (gate - src - Vtp)/Vdd)*(gp*1e-4))))
	else:
		if gate - src >= Vtp:
			constraints.append(logical_or(logical_and(src < drain, gate - drain >= Vtp, tI == -(src - drain)*(2 - (gate - drain - Vtp)/Vdd)*(gp*1e-4)),
											logical_and(src < drain, gate - drain <= Vtp, src - drain <= gate - drain - Vtp, tI == -0.5*Sp*Kp*(gate - drain - Vtp)*(gate - drain - Vtp) - (src - drain)*(2 - (gate - drain - Vtp)/Vdd)*(gp*1e-4)),
											logical_and(src < drain, gate - drain <= Vtp, src - drain >= gate - drain - Vtp, tI == -Sp*Kp*(gate - drain - Vtp - (src - drain)/2.0)*(src - drain) - (src - drain)*(2 - (gate - drain - Vtp)/Vdd)*(gp*1e-4)),
											logical_and(src >= drain, tI == (drain - src)*(2 - (gate - src - Vtp)/Vdd)*(gp*1e-4))))
		else:
			constraints.append(logical_or(logical_and(src < drain, gate - drain >= Vtp, tI == -(src - drain)*(2 - (gate - drain - Vtp)/Vdd)*(gp*1e-4)),
											logical_and(src < drain, gate - drain <= Vtp, src - drain <= gate - drain - Vtp, tI == -0.5*Sp*Kp*(gate - drain - Vtp)*(gate - drain - Vtp) - (src - drain)*(2 - (gate - drain - Vtp)/Vdd)*(gp*1e-4)),
											logical_and(src < drain, gate - drain <= Vtp, src - drain >= gate - drain - Vtp, tI == -Sp*Kp*(gate - drain - Vtp - (src - drain)/2.0)*(src - drain) - (src - drain)*(2 - (gate - drain - Vtp)/Vdd)*(gp*1e-4)),
											logical_and(src >= drain, drain - src <= gate - src - Vtp, tI == 0.5*Sp*Kp*(gate - src - Vtp)*(gate - src - Vtp) + (drain - src)*(2 - (gate - src - Vtp)/Vdd)*(gp*1e-4)),
											logical_and(src >= drain, drain - src >= gate - src - Vtp, tI == Sp*Kp*(gate - src - Vtp - (drain - src)/2.0)*(drain - src) + (drain - src)*(2 - (gate - src - Vtp)/Vdd)*(gp*1e-4))))
	
	return constraints

def schmittTrigger(inputVoltage, Vtp, Vtn, Vdd, Kn, Kp, Sn, numSolutions = "all"):
	epsilon = 1e-14
	start = time.time()
	print ("Vtp", Vtp, "Vtn", Vtn, "Vdd", Vdd, "Kn", Kn, "Kp", Kp, "Sn", Sn)
	Sp = Sn *2.0
	model = SchmittMosfet(modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn], inputVoltage=inputVoltage)
	gn = model.gn
	gp = model.gp
	boundMap = model.boundMap

	lenV = 3

	vs = []
	tIs = []
	nIs = []

	for i in range(lenV):
		vs.append(Variable("v" + str(i)))
		nIs.append(Variable("nI" + str(i)))
	
	for i in range(lenV*2):
		tIs.append(Variable("tI" + str(i)))

	allSolutions = []
	while True:
		if numSolutions != "all" and len(allSolutions) == numSolutions:
			break
		allConstraints = []	
		for i in range(lenV):
			allConstraints.append(vs[i] >= -0.2)
			allConstraints.append(vs[i] <= 2.0)
		allConstraints += nFet(Vtn, Vdd, Kn, Sn, gn, 0.0, inputVoltage, vs[1], tIs[0])
		allConstraints += nFet(Vtn, Vdd, Kn, Sn, gn, vs[1], inputVoltage, vs[0], tIs[1])
		allConstraints += nFet(Vtn, Vdd, Kn, Sn, gn, vs[1], vs[0], Vdd, tIs[2])
		allConstraints += pFet(Vtp, Vdd, Kp, Sp, gp, Vdd, inputVoltage, vs[0], tIs[3])
		allConstraints += pFet(Vtp, Vdd, Kp, Sp, gp, vs[2], inputVoltage, vs[0], tIs[4])
		allConstraints += pFet(Vtp, Vdd, Kp, Sp, gp, vs[2], vs[0], 0.0, tIs[5])
		allConstraints.append(nIs[0] == -tIs[4] - tIs[1])
		allConstraints.append(nIs[1] == -tIs[0] + tIs[1] + tIs[2])
		allConstraints.append(nIs[2] == -tIs[3] + tIs[5] + tIs[4])
		for i in range(lenV):
			allConstraints.append(nIs[i] == 0.0)
		
		excludingConstraints = []
		for solution in allSolutions:
			singleExcludingConstraints = []
			for i in range(lenV):
				singleExcludingConstraints.append(vs[i] <= solution[i][0])
				singleExcludingConstraints.append(vs[i] >= solution[i][1])
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
		print (result)
		if result is None:
			break
		hyper = np.zeros((lenV,2))
		for i in range(lenV):
			hyper[i,:] = [result[vs[i]].lb() - 1e+2*epsilon, result[vs[i]].ub() + 1e+2*epsilon]

		print ("hyper", hyper)
		allSolutions.append(hyper)

		print ("num solutions found", len(allSolutions))
		'''if len(allSolutions) == 1:
			break'''

	# categorize solutions found
	'''sampleSols, rotatedSols, stableSols, unstableSols = rUtils.categorizeSolutions(allSolutions,model)
	
	for solution in allSolutions:
		print (solution)

	for hi in range(len(sampleSols)):
		print ("equivalence class# ", hi)
		print ("main member ", sampleSols[hi])
		print ("number of other members ", len(rotatedSols[hi]))
		print ("other member rotationIndices: ")
		for mi in range(len(rotatedSols[hi])):
			print (rotatedSols[hi][mi])
		print ("")'''
	end = time.time()
	print ("time taken", end - start)

schmittTrigger(inputVoltage = 1.2, Vtp = -0.4, Vtn = 0.4, Vdd = 1.8, Kn = 1.5, Kp = -0.75, Sn = (8/3.0), numSolutions = "all")