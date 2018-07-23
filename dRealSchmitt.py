from dreal.symbolic import Variable, logical_and, logical_or, tanh
from dreal.symbolic import logical_not
from dreal.symbolic import forall
from dreal.api import CheckSatisfiability, Minimize
import time
import numpy as np


def nFetLeak(Vtn, Vdd, Kn, Sn, src, gate, drain, tI):
	constraints = []
	gds = 1e-8
	if type(src) == Variable or type(gate) == Variable:
		constraints.append(logical_or(logical_and(src > drain, gate - drain <= Vtn, tI == -(src - drain)*gds),
										logical_and(src > drain, gate - drain >= Vtn, src - drain >= gate - drain - Vtn, tI == -0.5*Sn*Kn*(gate - drain - Vtn)*(gate - drain - Vtn) -(src - drain)*gds),
										logical_and(src > drain, gate - drain >= Vtn, src - drain <= gate - drain - Vtn, tI == -Sn*Kn*(gate - drain - Vtn - (src - drain)/2.0)*(src - drain) -(src - drain)*gds),
										logical_and(src <= drain, gate - src <= Vtn, tI == (drain - src)*gds),
										logical_and(src <= drain, gate - src >= Vtn, drain - src >= gate - src - Vtn, tI == 0.5*Sn*Kn*(gate - src - Vtn)*(gate - src - Vtn) + (drain - src)*gds),
										logical_and(src <= drain, gate - src >= Vtn, drain - src <= gate - src - Vtn, tI == Sn*Kn*(gate - src - Vtn - (drain - src)/2.0)*(drain - src) + (drain - src)*gds)))
	else:
		if gate - src <= Vtn:
			constraints.append(logical_or(logical_and(src > drain, gate - drain <= Vtn, tI == -(src - drain)*gds),
											logical_and(src > drain, gate - drain >= Vtn, src - drain >= gate - drain - Vtn, tI == -0.5*Sn*Kn*(gate - drain - Vtn)*(gate - drain - Vtn) -(src - drain)*gds),
											logical_and(src > drain, gate - drain >= Vtn, src - drain <= gate - drain - Vtn, tI == -Sn*Kn*(gate - drain - Vtn - (src - drain)/2.0)*(src - drain) -(src - drain)*gds),
											logical_and(src <= drain, tI == (drain - src)*gds)))
		else:
			constraints.append(logical_or(logical_and(src > drain, gate - drain <= Vtn, tI == -(src - drain)*gds),
											logical_and(src > drain, gate - drain >= Vtn, src - drain >= gate - drain - Vtn, tI == -0.5*Sn*Kn*(gate - drain - Vtn)*(gate - drain - Vtn) -(src - drain)*gds),
											logical_and(src > drain, gate - drain >= Vtn, src - drain <= gate - drain - Vtn, tI == -Sn*Kn*(gate - drain - Vtn - (src - drain)/2.0)*(src - drain) -(src - drain)*gds),
											logical_and(src <= drain, drain - src >= gate - src - Vtn, tI == 0.5*Sn*Kn*(gate - src - Vtn)*(gate - src - Vtn) + (drain - src)*gds),
											logical_and(src <= drain, drain - src <= gate - src - Vtn, tI == Sn*Kn*(gate - src - Vtn - (drain - src)/2.0)*(drain - src) + (drain - src)*gds)))
	
	return constraints

def nFet(Vtn, Vdd, Kn, Sn, src, gate, drain, tI):
	constraints = []
	if type(src) == Variable or type(gate) == Variable:
		constraints.append(logical_or(logical_and(src > drain, gate - drain <= Vtn, tI == 0.0),
										logical_and(src > drain, gate - drain >= Vtn, src - drain >= gate - drain - Vtn, tI == -0.5*Sn*Kn*(gate - drain - Vtn)*(gate - drain - Vtn)),
										logical_and(src > drain, gate - drain >= Vtn, src - drain <= gate - drain - Vtn, tI == -Sn*Kn*(gate - drain - Vtn - (src - drain)/2.0)*(src - drain)),
										logical_and(src <= drain, gate - src <= Vtn, tI == 0.0),
										logical_and(src <= drain, gate - src >= Vtn, drain - src >= gate - src - Vtn, tI == 0.5*Sn*Kn*(gate - src - Vtn)*(gate - src - Vtn)),
										logical_and(src <= drain, gate - src >= Vtn, drain - src <= gate - src - Vtn, tI == Sn*Kn*(gate - src - Vtn - (drain - src)/2.0)*(drain - src))))
	else:
		if gate - src <= Vtn:
			constraints.append(logical_or(logical_and(src > drain, gate - drain <= Vtn, tI == 0.0),
											logical_and(src > drain, gate - drain >= Vtn, src - drain >= gate - drain - Vtn, tI == -0.5*Sn*Kn*(gate - drain - Vtn)*(gate - drain - Vtn)),
											logical_and(src > drain, gate - drain >= Vtn, src - drain <= gate - drain - Vtn, tI == -Sn*Kn*(gate - drain - Vtn - (src - drain)/2.0)*(src - drain)),
											logical_and(src <= drain, tI == 0.0)))
		else:
			constraints.append(logical_or(logical_and(src > drain, gate - drain <= Vtn, tI == 0.0),
											logical_and(src > drain, gate - drain >= Vtn, src - drain >= gate - drain - Vtn, tI == -0.5*Sn*Kn*(gate - drain - Vtn)*(gate - drain - Vtn)),
											logical_and(src > drain, gate - drain >= Vtn, src - drain <= gate - drain - Vtn, tI == -Sn*Kn*(gate - drain - Vtn - (src - drain)/2.0)*(src - drain)),
											logical_and(src <= drain, drain - src >= gate - src - Vtn, tI == 0.5*Sn*Kn*(gate - src - Vtn)*(gate - src - Vtn)),
											logical_and(src <= drain, drain - src <= gate - src - Vtn, tI == Sn*Kn*(gate - src - Vtn - (drain - src)/2.0)*(drain - src))))
	
	return constraints

def pFetLeak(Vtp, Vdd, Kp, Sp, src, gate, drain, tI):
	constraints = []
	gds = 1e-8
	if type(src) == Variable or type(gate) == Variable:
		constraints.append(logical_or(logical_and(src < drain, gate - drain >= Vtp, tI == -(src - drain)*gds),
										logical_and(src < drain, gate - drain <= Vtp, src - drain <= gate - drain - Vtp, tI == -0.5*Sp*Kp*(gate - drain - Vtp)*(gate - drain - Vtp) -(src - drain)*gds),
										logical_and(src < drain, gate - drain <= Vtp, src - drain >= gate - drain - Vtp, tI == -Sp*Kp*(gate - drain - Vtp - (src - drain)/2.0)*(src - drain) -(src - drain)*gds),
										logical_and(src >= drain, gate - src >= Vtp, tI == (drain - src)*gds),
										logical_and(src >= drain, gate - src <= Vtp, drain - src <= gate - src - Vtp, tI == 0.5*Sp*Kp*(gate - src - Vtp)*(gate - src - Vtp) + (drain - src)*gds),
										logical_and(src >= drain, gate - src <= Vtp, drain - src >= gate - src - Vtp, tI == Sp*Kp*(gate - src - Vtp - (drain - src)/2.0)*(drain - src) + (drain - src)*gds)))
	else:
		if gate - src >= Vtp:
			constraints.append(logical_or(logical_and(src < drain, gate - drain >= Vtp, tI == -(src - drain)*gds),
											logical_and(src < drain, gate - drain <= Vtp, src - drain <= gate - drain - Vtp, tI == -0.5*Sp*Kp*(gate - drain - Vtp)*(gate - drain - Vtp) -(src - drain)*gds),
											logical_and(src < drain, gate - drain <= Vtp, src - drain >= gate - drain - Vtp, tI == -Sp*Kp*(gate - drain - Vtp - (src - drain)/2.0)*(src - drain) -(src - drain)*gds),
											logical_and(src >= drain, tI == (drain - src)*gds)))
		else:
			constraints.append(logical_or(logical_and(src < drain, gate - drain >= Vtp, tI == -(src - drain)*gds),
											logical_and(src < drain, gate - drain <= Vtp, src - drain <= gate - drain - Vtp, tI == -0.5*Sp*Kp*(gate - drain - Vtp)*(gate - drain - Vtp) -(src - drain)*gds),
											logical_and(src < drain, gate - drain <= Vtp, src - drain >= gate - drain - Vtp, tI == -Sp*Kp*(gate - drain - Vtp - (src - drain)/2.0)*(src - drain) -(src - drain)*gds),
											logical_and(src >= drain, drain - src <= gate - src - Vtp, tI == 0.5*Sp*Kp*(gate - src - Vtp)*(gate - src - Vtp) + (drain - src)*gds),
											logical_and(src >= drain, drain - src >= gate - src - Vtp, tI == Sp*Kp*(gate - src - Vtp - (drain - src)/2.0)*(drain - src) + (drain - src)*gds)))
	
	return constraints

def pFet(Vtp, Vdd, Kp, Sp, src, gate, drain, tI):
	constraints = []
	if type(src) == Variable or type(gate) == Variable:
		constraints.append(logical_or(logical_and(src < drain, gate - drain >= Vtp, tI == 0.0),
										logical_and(src < drain, gate - drain <= Vtp, src - drain <= gate - drain - Vtp, tI == -0.5*Sp*Kp*(gate - drain - Vtp)*(gate - drain - Vtp)),
										logical_and(src < drain, gate - drain <= Vtp, src - drain >= gate - drain - Vtp, tI == -Sp*Kp*(gate - drain - Vtp - (src - drain)/2.0)*(src - drain)),
										logical_and(src >= drain, gate - src >= Vtp, tI == 0.0),
										logical_and(src >= drain, gate - src <= Vtp, drain - src <= gate - src - Vtp, tI == 0.5*Sp*Kp*(gate - src - Vtp)*(gate - src - Vtp)),
										logical_and(src >= drain, gate - src <= Vtp, drain - src >= gate - src - Vtp, tI == Sp*Kp*(gate - src - Vtp - (drain - src)/2.0)*(drain - src))))
	else:
		if gate - src >= Vtp:
			constraints.append(logical_or(logical_and(src < drain, gate - drain >= Vtp, tI == 0.0),
											logical_and(src < drain, gate - drain <= Vtp, src - drain <= gate - drain - Vtp, tI == -0.5*Sp*Kp*(gate - drain - Vtp)*(gate - drain - Vtp)),
											logical_and(src < drain, gate - drain <= Vtp, src - drain >= gate - drain - Vtp, tI == -Sp*Kp*(gate - drain - Vtp - (src - drain)/2.0)*(src - drain)),
											logical_and(src >= drain, tI == 0.0)))
		else:
			constraints.append(logical_or(logical_and(src < drain, gate - drain >= Vtp, tI == 0.0),
											logical_and(src < drain, gate - drain <= Vtp, src - drain <= gate - drain - Vtp, tI == -0.5*Sp*Kp*(gate - drain - Vtp)*(gate - drain - Vtp)),
											logical_and(src < drain, gate - drain <= Vtp, src - drain >= gate - drain - Vtp, tI == -Sp*Kp*(gate - drain - Vtp - (src - drain)/2.0)*(src - drain)),
											logical_and(src >= drain, drain - src <= gate - src - Vtp, tI == 0.5*Sp*Kp*(gate - src - Vtp)*(gate - src - Vtp)),
											logical_and(src >= drain, drain - src >= gate - src - Vtp, tI == Sp*Kp*(gate - src - Vtp - (drain - src)/2.0)*(drain - src))))
	
	return constraints

def schmittTrigger(inputVoltage, Vtp, Vtn, Vdd, Kn, Kp, Sn, numSolutions = "all"):
	epsilon = 1e-14
	start = time.time()
	print ("Vtp", Vtp, "Vtn", Vtn, "Vdd", Vdd, "Kn", Kn, "Kp", Kp, "Sn", Sn)
	Sp = Sn *2.0

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
			allConstraints.append(vs[i] >= 0.0)
			allConstraints.append(vs[i] <= 1.8)
		allConstraints += nFet(Vtn, Vdd, Kn, Sn, 0.0, inputVoltage, vs[1], tIs[0])
		allConstraints += nFet(Vtn, Vdd, Kn, Sn, vs[1], inputVoltage, vs[0], tIs[1])
		allConstraints += nFet(Vtn, Vdd, Kn, Sn, vs[1], vs[0], Vdd, tIs[2])
		allConstraints += pFet(Vtp, Vdd, Kp, Sp, Vdd, inputVoltage, vs[2], tIs[3])
		allConstraints += pFet(Vtp, Vdd, Kp, Sp, vs[2], inputVoltage, vs[0], tIs[4])
		allConstraints += pFet(Vtp, Vdd, Kp, Sp, vs[2], vs[0], 0.0, tIs[5])
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
		#print (result)
		if result is None:
			break
		hyper = np.zeros((lenV,2))
		for i in range(lenV):
			hyper[i,:] = [result[vs[i]].lb() - 1e+13*epsilon, result[vs[i]].ub() + 1e+13*epsilon]

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
	return allSolutions

if __name__ == "__main__":
	schmittTrigger(inputVoltage = 1.8, Vtp = -0.4, Vtn = 0.4, Vdd = 1.8, Kn = 270*1e-6, Kp = -90*1e-6, Sn = 3.0, numSolutions = "all")
