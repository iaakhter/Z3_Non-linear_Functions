from dreal.symbolic import Variable, logical_and, logical_or, tanh
from dreal.symbolic import logical_not
from dreal.api import CheckSatisfiability, Minimize
import time
import rambusUtils as rUtils
import numpy as np
from tanhModel import TanhModel
from mosfetModel import MosfetModel
import intervalUtils

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
		result = CheckSatisfiability(f_sat, 1e-16)
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


def rambusOscillatorTanh(a, numStages, g_cc = 0.5):
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
			allConstraints.append(vs[i] >= -1)
			allConstraints.append(vs[i] <= 1)
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
		result = CheckSatisfiability(f_sat, 1e-16)
		#print (result)
		if result is None:
			break
		hyper = np.zeros((lenV,2))
		for i in range(lenV):
			hyper[i,:] = [result[vs[i]].lb(), result[vs[i]].ub()]

		#allSolutions.append(hyper)
		# find the biggest hyperrectangle containing the unique solution
		hyperWithUniqueSoln = np.zeros((lenV,2))
		diff = np.ones((lenV))*0.4
		startingIndex = 0
		while True:
			#print ("diff", diff)
			hyperWithUniqueSoln[:,0] = hyper[:,0] - diff
			hyperWithUniqueSoln[:,1] = hyper[:,1] + diff
			hyperWithUniqueSoln = np.clip(hyperWithUniqueSoln, -1.0, 1.0)
			kResult = intervalUtils.checkExistenceOfSolutionGS(model,hyperWithUniqueSoln.transpose())
			if kResult[0] == False and kResult[1] is not None:
				diff[startingIndex] = diff[startingIndex]/2.0
				startingIndex = (startingIndex + 1)%lenV
			else:
				print ("found unique hyper", hyperWithUniqueSoln)
				allSolutions.append(hyperWithUniqueSoln)
				break

		print ("num solutions found", len(allSolutions))
		'''if len(allSolutions) == 3:
			break'''
		

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

	for hi in range(len(sampleSols)):
		if len(rotatedSols[hi]) > lenV - 1 or (len(rotatedSols[hi]) >= 1 and rotatedSols[hi][0] == 0):
			print ("problem equivalence class# ", hi)
			print ("main member ", sampleSols[hi])
			print ("num other Solutions ", len(rotatedSols[hi]))

	print ("")
	print ("num stable solutions ", len(stableSols))
	print ("num solutions", len(allSolutions))
	end = time.time()
	print ("time taken", end - start)


def mosfetCurrent(Vtp, Vtn, Vdd, Kn, Kp, Sn, Sp, IVarN, IVarP, VinVar, VoutVar):
	constraints = []

	constraints.append(logical_or(logical_and(VinVar <= Vtn, IVarN == 0.0),
									logical_and(Vtn <= VinVar, VinVar <= VoutVar + Vtn, IVarN == Sn*(Kn/2.0)*(VinVar - Vtn)*(VinVar - Vtn)),
									logical_and(Vtn <= VinVar, VinVar >= VoutVar + Vtn, IVarN == Sn*(Kn)*(VinVar - Vtn - VoutVar/2.0)*VoutVar)))

	constraints.append(logical_or(logical_and(VinVar - Vtp >= Vdd, IVarP == 0.0),
									logical_and(VinVar - Vtp <= Vdd, VoutVar <= VinVar - Vtp, IVarP == Sp*(Kp/2.0)*(VinVar - Vtp - Vdd)*(VinVar - Vtp - Vdd)),
									logical_and(VinVar - Vtp <= Vdd, VoutVar >= VinVar - Vtp, IVarP == Sp*Kp*((VinVar - Vtp - Vdd) - (VoutVar - Vdd)/2.0)*(VoutVar - Vdd))))
	

	return constraints


def rambusOscillatorMosfet(Vtp, Vtn, Vdd, Kn, Kp, Sn, numStages, g_cc = 0.5):
	start = time.time()
	print ("Vtp", Vtp, "Vtn", Vtn, "Vdd", Vdd, "Kn", Kn, "Kp", Kp, "Sn", Sn)
	g_fwd = 1.0
	Sp = Sn *2.0
	model = MosfetModel(modelParam = [Vtp, Vtn, Vdd, Kn, Kp, Sn], g_cc = g_cc, g_fwd = g_fwd, numStages = numStages)
	boundMap = model.boundMap
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
		result = CheckSatisfiability(f_sat, 1e-16)
		#print (result)
		if result is None:
			break
		hyper = np.zeros((lenV,2))
		for i in range(lenV):
			hyper[i,:] = [result[vs[i]].lb(), result[vs[i]].ub()]

		#print ("hyper", hyper)
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
			elif kResult[0]:
				print ("found unique hyper", hyperWithUniqueSoln)
				allSolutions.append(hyperWithUniqueSoln)
				break
			else:
				print ("no unique hyper", hyperWithUniqueSoln)
				return

		print ("num solutions found", len(allSolutions))
		'''if len(allSolutions) == 1:
			break'''

	# categorize solutions found
	sampleSols, rotatedSols, stableSols, unstableSols = rUtils.categorizeSolutions(allSolutions,model)
	
	for solution in allSolutions:
		print (solution)

	for hi in range(len(sampleSols)):
		print ("equivalence class# ", hi)
		print ("main member ", sampleSols[hi])
		print ("number of other members ", len(rotatedSols[hi]))
		print ("other member rotationIndices: ")
		for mi in range(len(rotatedSols[hi])):
			print (rotatedSols[hi][mi])
		print ("")
	end = time.time()
	print ("time taken", end - start)


#rambusOscillatorTanh(a = -5.0, numStages = 2, g_cc = 0.5)

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

rambusOscillatorMosfet(Vtp = -0.25, Vtn = 0.25, Vdd = 1.0, Kn = 1.0, Kp = -0.5, Sn = 1.0, numStages = 4, g_cc = 0.5)
