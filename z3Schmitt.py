from z3 import *
import rambusUtils as rUtils
import numpy as np
from schmittMosfet import SchmittMosfet
import intervalUtils
import time


def nFet(Vtn, Vdd, Kn, Sn, gn, src, gate, drain, tI):
	constraints = []
	constraints.append(Or(And(src > drain, gate - drain <= Vtn, tI == -(src - drain)*(2 + (gate - drain - Vtn)/Vdd)*(gn*1e-4)),
							And(src > drain, gate - drain >= Vtn, src - drain >= gate - drain - Vtn, tI == -0.5*Sn*Kn*(gate - drain - Vtn)*(gate - drain - Vtn) - (src - drain)*(2 + (gate - drain - Vtn)/Vdd)*(gn*1e-4)),
							And(src > drain, gate - drain >= Vtn, src - drain <= gate - drain - Vtn, tI == -Sn*Kn*(gate - drain - Vtn - (src - drain)/2.0)*(src - drain) - (src - drain)*(2 + (gate - drain - Vtn)/Vdd)*(gn*1e-4)),
							And(src <= drain, gate - src <= Vtn, tI == (drain - src)*(2 + (gate - src - Vtn)/Vdd)*(gn*1e-4)),
							And(src <= drain, gate - src >= Vtn, drain - src >= gate - src - Vtn, tI == 0.5*Sn*Kn*(gate - src - Vtn)*(gate - src - Vtn) + (drain - src)*(2 + (gate - src - Vtn)/Vdd)*(gn*1e-4)),
							And(src <= drain, gate - src >= Vtn, drain - src <= gate - src - Vtn, tI == Sn*Kn*(gate - src - Vtn - (drain - src)/2.0)*(drain - src) + (drain - src)*(2 + (gate - src - Vtn)/Vdd)*(gn*1e-4))))
	return constraints
def pFet(Vtp, Vdd, Kp, Sp, gp, src, gate, drain, tI):
	constraints = []
	constraints.append(Or(And(src < drain, gate - drain >= Vtp, tI == -(src - drain)*(2 - (gate - drain - Vtp)/Vdd)*(gp*1e-4)),
							And(src < drain, gate - drain <= Vtp, src - drain <= gate - drain - Vtp, tI == -0.5*Sp*Kp*(gate - drain - Vtp)*(gate - drain - Vtp) - (src - drain)*(2 - (gate - drain - Vtp)/Vdd)*(gp*1e-4)),
							And(src < drain, gate - drain <= Vtp, src - drain >= gate - drain - Vtp, tI == -Sp*Kp*(gate - drain - Vtp - (src - drain)/2.0)*(src - drain) - (src - drain)*(2 - (gate - drain - Vtp)/Vdd)*(gp*1e-4)),
							And(src >= drain, gate - src >= Vtp, tI == (drain - src)*(2 - (gate - src - Vtp)/Vdd)*(gp*1e-4)),
							And(src >= drain, gate - src <= Vtp, drain - src <= gate - src - Vtp, tI == 0.5*Sp*Kp*(gate - src - Vtp)*(gate - src - Vtp) + (drain - src)*(2 - (gate - src - Vtp)/Vdd)*(gp*1e-4)),
							And(src >= drain, gate - src <= Vtp, drain - src >= gate - src - Vtp, tI == Sp*Kp*(gate - src - Vtp - (drain - src)/2.0)*(drain - src) + (drain - src)*(2 - (gate - src - Vtp)/Vdd)*(gp*1e-4))))
	return constraints

def schmittTrigger(inputVoltage, Vtp, Vtn, Vdd, Kn, Kp, Sn, numSolutions = "all"):
	start = time.time()
	print ("Vtp", Vtp, "Vtn", Vtn, "Vdd", Vdd, "Kn", Kn, "Kp", Kp, "Sn", Sn)
	Sp = Sn *2.0
	model = SchmittMosfet(modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn], inputVoltage=inputVoltage)
	gn = model.gn
	gp = model.gp
	boundMap = model.boundMap

	vs = RealVector('v', 3)
	tIs = RealVector('tI', 6)
	nIs = RealVector('nI',3)
	lenV = 3
	s = Solver()

	allSolutions = []
	while True:
		if numSolutions != "all" and len(allSolutions) == numSolutions:
			break
		allConstraints = []
		for i in range(lenV):
			allConstraints.append(vs[i] >= 0.0)
			allConstraints.append(vs[i] <= Vdd)
			allConstraints.append(nIs[i] == 0.0)
		allConstraints += nFet(Vtn, Vdd, Kn, Sn, gn, 0.0, inputVoltage, vs[1], tIs[0])
		allConstraints += nFet(Vtn, Vdd, Kn, Sn, gn, vs[1], inputVoltage, vs[0], tIs[1])
		allConstraints += nFet(Vtn, Vdd, Kn, Sn, gn, vs[1], vs[0], Vdd, tIs[2])
		allConstraints += pFet(Vtp, Vdd, Kp, Sp, gp, Vdd, inputVoltage, vs[2], tIs[3])
		allConstraints += pFet(Vtp, Vdd, Kp, Sp, gp, vs[2], inputVoltage, vs[0], tIs[4])
		allConstraints += pFet(Vtp, Vdd, Kp, Sp, gp, vs[2], vs[0], 0.0, tIs[5])
		allConstraints.append(nIs[0] == -tIs[4] - tIs[1])
		allConstraints.append(nIs[1] == -tIs[0] + tIs[1] + tIs[2])
		allConstraints.append(nIs[2] == -tIs[3] + tIs[5] + tIs[4])

		
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
		#print ("s")
		#print (s)
		result = s.check()
		print (result)
		if result != sat:
			break
		
		m = s.model()
		sol = np.zeros((3))
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
	return allSolutions

if __name__ == "__main__":
	schmittTrigger(inputVoltage = 0.2, Vtp = -0.4, Vtn = 0.4, Vdd = 1.8, Kn = 270*1e-6, Kp = -90*1e-6, Sn = (8/3.0), numSolutions = "all")
