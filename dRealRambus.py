from dreal.symbolic import Variable, logical_and, logical_or, tanh
from dreal.symbolic import logical_not
from dreal.symbolic import forall
from dreal.api import CheckSatisfiability, Minimize
import time
import rambusUtils as rUtils
import numpy as np
from tanhModel import TanhModel
from mosfetModel import MosfetModel
import intervalUtils
import resource

# find the solution in hyper
def rambusOscillatorTanhDebug(a, numStages, hyper, g_cc = 0.5):
	model = TanhModel(modelParam = a, g_cc = g_cc, g_fwd = 1.0, numStages=numStages)
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

	allSolutions = []
	while True:
		allConstraints = []
		for i in range(lenV):
			allConstraints.append(vs[i] >= hyper[i][0])
			allConstraints.append(vs[i] <= hyper[i][1])
			fwdInd = (i-1)%lenV
			ccInd = (i+lenV//2)%lenV
			allConstraints.append(vfwds[i] == tanh(a*vs[fwdInd]))
			allConstraints.append(vccs[i] == tanh(a*vs[ccInd]))
			allConstraints.append(g_fwd*vfwds[i] + (-g_fwd-g_cc)*vs[i] + g_cc*vccs[i] == 0)

		excludingConstraints = []
		for solution in allSolutions:
			singleExcludingConstraints = []
			for i in range(lenV):
				singleExcludingConstraints.append(vs[i] <= solution[i][0])
				singleExcludingConstraints.append(vs[i] >= solution[i][1])
			excludingConstraints.append(singleExcludingConstraints)
		
		f_sat = logical_and(*allConstraints)
		if len(excludingConstraints) > 0:
			for constraints in excludingConstraints:
				f_sat = logical_and(f_sat, logical_or(*constraints))
		
		#print ("f_sat")
		#print (f_sat)
		result = CheckSatisfiability(f_sat, 1e-12)
		#print (result)
		if result is None:
			break
		hyper = np.zeros((lenV,2))
		for i in range(lenV):
			hyper[i,:] = [result[vs[i]].lb(), result[vs[i]].ub()]

		# find the biggest hyperrectangle containing the unique solution
		hyperWithUniqueSoln = np.zeros((lenV,2))
		diff = np.ones((lenV))*0.4
		startingIndex = 0
		while True:
			#print ("diff", diff)
			hyperWithUniqueSoln[:,0] = hyper[:,0] - diff
			hyperWithUniqueSoln[:,1] = hyper[:,1] + diff
			kResult = intervalUtils.checkExistenceOfSolutionGS(model,hyperWithUniqueSoln.transpose())
			if kResult[0] == False and kResult[1] is not None:
				diff[startingIndex] = diff[startingIndex]/2.0
				startingIndex = (startingIndex + 1)%lenV
			else:
				print ("found unique hyper", hyperWithUniqueSoln)
				allSolutions.append(hyperWithUniqueSoln)
				break

		print ("num solutions found", len(allSolutions))

	# categorize solutions found
	sampleSols, rotatedSols, stableSols, unstableSols = rUtils.categorizeSolutions(allSolutions,model)
	

	for hi in range(len(sampleSols)):
		print ("equivalence class# ", hi)
		print ("main member ", sampleSols[hi])
		print ("number of other members ", len(rotatedSols[hi]))
		print ("other member rotationIndices: ")
		for mi in range(len(rotatedSols[hi])):
			print (rotatedSols[hi][mi])
		print ("")

def rambusOscillatorTanh(a, numStages, numSolutions = "all", g_cc = 0.5):
	epsilon = 1e-14
	model = TanhModel(modelParam = a, g_cc = g_cc, g_fwd = 1.0, numStages=numStages)
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

	allSolutions = []
	while True:
		if numSolutions != "all" and len(allSolutions) == numSolutions:
			break
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
		

	# categorize solutions found
	'''sampleSols, rotatedSols, stableSols, unstableSols = rUtils.categorizeSolutions(allSolutions,model)
	

	for hi in range(len(sampleSols)):
		print ("equivalence class# ", hi)
		print ("main member ", sampleSols[hi])
		print ("number of other members ", len(rotatedSols[hi]))
		print ("other member rotationIndices: ")
		for mi in range(len(rotatedSols[hi])):
			print (rotatedSols[hi][mi])
		print ("")

	for hi in range(len(sampleSols)):
		if len(rotatedSols[hi]) > lenV - 1 or (len(rotatedSols[hi]) >= 1 and rotatedSols[hi][0] == 0):
			print ("problem equivalence class# ", hi)
			print ("main member ", sampleSols[hi])
			print ("num other Solutions ", len(rotatedSols[hi]))

	print ("")
	print ("num stable solutions ", len(stableSols))
	print ("num solutions", len(allSolutions))'''
	end = time.time()
	print ("time taken", end - start)
	return allSolutions


def mosfetCurrent(Vtp, Vtn, Vdd, Kn, Kp, Sn, Sp, IVarN, IVarP, VinVar, VoutVar):
	constraints = []

	constraints.append(logical_or(logical_and(VinVar <= Vtn, IVarN == 0.0),
									logical_and(Vtn <= VinVar, VinVar <= VoutVar + Vtn, IVarN == Sn*(Kn/2.0)*(VinVar - Vtn)*(VinVar - Vtn)),
									logical_and(Vtn <= VinVar, VinVar >= VoutVar + Vtn, IVarN == Sn*(Kn)*(VinVar - Vtn - VoutVar/2.0)*VoutVar)))

	constraints.append(logical_or(logical_and(VinVar - Vtp >= Vdd, IVarP == 0.0),
									logical_and(VinVar - Vtp <= Vdd, VoutVar <= VinVar - Vtp, IVarP == Sp*(Kp/2.0)*(VinVar - Vtp - Vdd)*(VinVar - Vtp - Vdd)),
									logical_and(VinVar - Vtp <= Vdd, VoutVar >= VinVar - Vtp, IVarP == Sp*Kp*((VinVar - Vtp - Vdd) - (VoutVar - Vdd)/2.0)*(VoutVar - Vdd))))
	

	return constraints


def rambusOscillatorMosfet(Vtp, Vtn, Vdd, Kn, Kp, Sn, numStages, numSolutions = "all", g_cc = 0.5):
	epsilon = 1e-14
	model = MosfetModel(modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn], g_cc = g_cc, g_fwd = 1.0, numStages = numStages)
	start = time.time()
	#print ("Vtp", Vtp, "Vtn", Vtn, "Vdd", Vdd, "Kn", Kn, "Kp", Kp, "Sn", Sn)
	g_fwd = 1.0
	Sp = Sn *2.0
	lenV = numStages*2

	vs = []
	ifwdNs = []
	ifwdPs = []
	iccNs = []
	iccPs = []
	for i in range(lenV):
		vs.append(Variable("v" + str(i)))
		ifwdNs.append(Variable("ifwdN" + str(i)))
		ifwdPs.append(Variable("ifwdP" + str(i)))
		iccNs.append(Variable("iccN" + str(i)))
		iccPs.append(Variable("iccP" + str(i)))

	allSolutions = []
	while True:
		if numSolutions != "all" and len(allSolutions) == numSolutions:
			break
		allConstraints = []	
		for i in range(lenV):
			allConstraints.append(vs[i] >= 0.0)
			allConstraints.append(vs[i] <= Vdd)
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
			hyper[i,:] = [result[vs[i]].lb() - 1000*epsilon, result[vs[i]].ub() + 1000*epsilon]

		print ("hyper", hyper)
		allSolutions.append(hyper)

		print ("num solutions found", len(allSolutions))

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


if __name__ =="__main__":
	#rambusOscillatorTanh(a = -5.0, numStages = 4, numSolutions = "all", g_cc = 4.0)
	rambusOscillatorMosfet(Vtp = -0.4, Vtn = 0.4, Vdd = 1.8, Kn = 270*1e-6, Kp = -90*1e-6, Sn = 3.0, numStages = 2, g_cc = 0.5)
	print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

	#check if you can easily find the solution in the given hyper
	'''hyper = np.array([[-0.74, -0.72],
		[-0.60, -0.58],
		[0.98, 1.0],
		[0.06, 0.08],
		[0.72, 0.74],
		[0.58, 0.60],
		[-1.0, -0.98],
		[-0.08, -0.06]])
	rambusOscillatorTanhDebug(a = -5.0, numStages = 4, hyper = hyper, g_cc = 4.0)'''
