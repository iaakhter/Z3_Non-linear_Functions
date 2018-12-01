import numpy as np
import matplotlib.pyplot as plt
from prototype import *
from intervalUtils import *
from circuitModels import InverterLoopMosfet
from funCompUtils import *

def analyzeSchmittTrigger():
	inputVoltages = np.linspace(0.35, 0.55, 50)
	#inputVoltages = np.linspace(0.0, 1.0, 50)
	posEigenValues = []
	outVoltages = []
	for i in range(len(inputVoltages)):
		statVars = {}
		inputVoltage = inputVoltages[i]
		modelParam = [1.0] #Vdd
		model = SchmittMosfet(modelType = "scMosfet", modelParam = modelParam, inputVoltage = inputVoltage)
		allHypers = []
		solverLoopNoLp(allHypers, model)
		if len(allHypers) != 3:
			raise Exception("analyzeSchmittTrigger inputVoltage "+str(inputVoltage)+
				": there should have been 3 solutions but got " + str(len(allHypers)) + " solutions ")
		
		foundPosEig = False
		hyperOutVolts = []
		for hyper in allHypers:
			exampleVolt = (hyper[:,0] + hyper[:,1])/2.0
			soln = newton(model,exampleVolt)
			if not(soln[0]):
				raise Exception("analyzeSchmittTrigger inputVoltage "+str(inputVoltage)+
				": newton's method should have found solution in  " + str(allHypers))
			soln = soln[1]
			hyperOutVolts.append(soln[0])
			jacAtSoln = model.jacobian(soln)
			eigVals,_ = np.linalg.eig(jacAtSoln)
			maxEig = np.amax(eigVals.real)
			if maxEig > 0:
				foundPosEig = True
				posEigenValues.append(maxEig)

		outVoltages.append(hyperOutVolts)
		if not(foundPosEig):
			raise Exception("analyzeSchmittTrigger inputVoltage "+str(inputVoltage)+
				": there should have been one equilibrium point with pos eig value  ")

	posEigenValues = np.array(posEigenValues)
	print ("Vin")
	print (inputVoltages)
	#print ("Vouts")
	#print (outVoltages)
	print ("Pos Eig")
	print (posEigenValues)
	#plt.plot(inputVoltages, posEigenValues, "*")
	#plt.xlabel("Vin")
	#plt.ylabel("Real Part of Eigen Values")
	#plt.show()

def schmittTriggerHysteresis():
	inputVoltages = np.linspace(0.0, 1.8, 50)
	inDrawVoltages = []
	outDrawVoltages = []
	for i in range(len(inputVoltages)):
		statVars = {}
		inputVoltage = inputVoltages[i]
		print ("inputVoltage", inputVoltage)
		#modelParam = [1.0] #Vdd
		#model = SchmittMosfet(modelType = "scMosfet", modelParam = modelParam, inputVoltage = inputVoltage)
		modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6, 8/3.0]
		model = SchmittMosfet(modelType = "lcMosfet", modelParam = modelParam, inputVoltage = inputVoltage)
		allHypers = []
		solverLoopNoLp(allHypers, model)
	
		for hyper in allHypers:
			exampleVolt = (hyper[:,0] + hyper[:,1])/2.0
			soln = newton(model,exampleVolt)
			if not(soln[0]):
				raise Exception("analyzeSchmittTrigger inputVoltage "+str(inputVoltage)+
				": newton's method should have found solution in  " + str(allHypers))
			soln = soln[1]
			inDrawVoltages.append(inputVoltage)
			outDrawVoltages.append(soln[0])
			

	inDrawVoltages = np.array(inDrawVoltages)
	outDrawVoltages = np.array(outDrawVoltages)
	#print ("num inDrawVoltages", inDrawVoltages.shape)
	#print ("num outDrawVoltages", outDrawVoltages.shape)
	plt.scatter(np.array(inDrawVoltages), np.array(outDrawVoltages), marker='*')
	plt.xlabel("Vin", fontsize=15)
	plt.ylabel("Vout", fontsize=15)
	plt.xlim(-0.2, 2.0)
	plt.ylim(-0.2, 2.0)
	plt.show()

# Assume a is negative
def tanhApprox(Vin, a):
	if Vin > -2.0/a:
		return -1-(a*Vin)**(-5)
	elif Vin <= -2.0/a and Vin >= 2.0/a:
		return -(13/256.0)*(a*Vin)**3 + (11/16.0)*(a*Vin)
	elif Vin < 2.0/a:
		return 1-(a*Vin)**(-5)

def compareTanhWithApprox():
	a = -5
	Vins = np.linspace(-1.0, 1.0, 50)
	VoutApprox = []
	VoutTanh = []
	for i in range(len(Vins)):
		Vin = Vins[i]
		VoutApprox.append(tanhApprox(Vin, a))
		VoutTanh.append((np.tanh(a*Vin)))

	VoutApprox = np.array(VoutApprox)
	VoutTanh = np.array(VoutTanh)
	#print (Vins.shape)
	#print (np.vstack((VoutApprox, VoutTanh)).shape)

	plt.plot(Vins, np.vstack((VoutApprox, VoutTanh)).transpose())
	plt.xlabel("x")
	plt.ylabel("y")
	plt.ylim([-1.5, 1.5])
	plt.legend(['tanh Approximation', 'tanh'])
	plt.show()

def tanhWithLinearConstraints():
	a = -5.0
	Vins = np.linspace(-1.0, 1.0, 100)
	lowGrad = tanhFunder(-1.0, a, 0.0)
	cLow = np.tanh(a*-1.0) - lowGrad*-1.0
	highGrad = tanhFunder(0.0, a, 0.0)
	cHigh = np.tanh(a*0.0) - highGrad*0.0

	secGrad = (np.tanh(a*-1.0) - np.tanh(a*0.0))/-1.0
	cSec = np.tanh(a*-1.0) - secGrad*-1.0
	VoutTanh = []
	leftTang = []
	rightTang = []
	sec = []

	for i in range(len(Vins)):
		Vin = Vins[i]
		VoutTanh.append(np.tanh(a*Vin))
		leftTang.append(lowGrad*Vin + cLow)
		rightTang.append(highGrad*Vin + cHigh)
		sec.append(secGrad*Vin + cSec)

	plt.plot(Vins, VoutTanh, 'b-', Vins, leftTang, 'r--', Vins, rightTang, 'r--', Vins, sec, 'g--')
	plt.xlabel("x")
	plt.ylabel("y")
	plt.ylim([-1.5, 1.5])
	plt.show()


def analyzeInverterLoop():
	posEigenValues = []
	statVars = {}
	modelParam = [1.0] #Vdd
	model = InverterLoopMosfet(modelType = "scMosfet", modelParam = modelParam)
	allHypers = []
	solverLoopNoLp(allHypers, model)
	#if len(allHypers) != 3:
	#	raise Exception("analyzeInverterLoop inputVoltage: there should have been 3 solutions but got " + str(len(allHypers)) + " solutions ")
	
	#print ("len(allHypers)", len(allHypers))
	foundPosEig = False
	for hyper in allHypers:
		exampleVolt = (hyper[:,0] + hyper[:,1])/2.0
		soln = newton(model,exampleVolt)
		#if not(soln[0]):
		#	raise Exception("analyzeInverterLoop: newton's method should have found solution in  " + str(allHypers))
		soln = soln[1]
		#print ("soln", soln)
		#print ("current", model.f(soln))
		jacAtSoln = model.jacobian(soln)
		eigVals,_ = np.linalg.eig(jacAtSoln)
		maxEig = np.amax(eigVals.real)
		if maxEig > 0:
			#print ("pos eig value at", hyper)
			foundPosEig = True
			posEigenValues.append(maxEig)

	#if not(foundPosEig):
	#	raise Exception("analyzeInverter inputVoltage : there should have been one equilibrium point with pos eig value  ")

	posEigenValues = np.array(posEigenValues)
	print ("Pos Eig")
	print (posEigenValues)


analyzeSchmittTrigger()
#analyzeInverterLoop()
#schmittTriggerHysteresis()
#compareTanhWithApprox()
#tanhWithLinearConstraints()
