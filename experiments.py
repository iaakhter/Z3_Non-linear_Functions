import time
import csv
#from memory_profiler import memory_usage
from prototype import *

import sys
sys.path.insert(0,'dRealCode/')

def solverRambusExperiments():
	numTrials = 1

	numStagesArray = [6]
	gccArray = [4.0]
	modelTypeArray = ["tanh"]
	useLp = False

	
	timingFilename = "data/solver_time_DATE2.csv"
	#timingFilename = "data/solver_time_kAlphaVals.csv"
	headers = ["Problem", "NumSolutions", "numBisection", "numLP", "numK", "numSingleKill", "numDoubleKill", "totalLPTime", "totalKTime", "avgLPTime", "avgKTime"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	headers.append("UseLp")
	csvTimeData = [headers]

	for modelType in modelTypeArray:
		for numStages in numStagesArray:
			for gcc in gccArray:
				print ("modelType", modelType, "numStages", numStages, "gcc", gcc)
				problemName = "rambus_"+modelType+"_stgs_"+str(numStages)+"_gcc_"+str(gcc)
				
				timeData = [problemName]
				if not(useLp):
					problemName += "_useLP_false"

				for n in range(numTrials):
					start = time.time()
					statVars = {}
					allHypers = rambusOscillator(modelType=modelType, numStages=numStages, g_cc=gcc, statVars=statVars, numSolutions="all", useLp=useLp)
					end = time.time()
					numSolutions = len(allHypers)
					timeTaken = end - start
					print ("run ", n, "timeTaken", timeTaken)

					if n == 0:
						timeData.append(numSolutions)
						timeData.append(statVars["numBisection"])
						timeData.append(statVars["numLp"])
						timeData.append(statVars["numK"])
						timeData.append(statVars["numSingleKill"])
						timeData.append(statVars["numDoubleKill"])
						timeData.append(statVars["totalLPTime"])
						timeData.append(statVars["totalKTime"])
						timeData.append(statVars["avgLPTime"])
						timeData.append(statVars["avgKTime"])

					timeData.append(timeTaken)
				
				if not(useLp):
					timeData.append("False")
				else:
					timeData.append("True")

				csvTimeData.append(timeData)
				
	timeFile = open(timingFilename, 'a')
	with timeFile:
		writer = csv.writer(timeFile)
		writer.writerows(csvTimeData)


def solverSchmittExperiments():
	numTrials = 1
	inputVoltageArray = [0.0, 0.5, 1.0]

	modelTypeArray = ["scMosfet"]
	useLp = False
	
	timingFilename = "data/solver_time_DATE2.csv"
	headers = ["Problem", "NumSolutions", "numBisection", "numLP", "numK", "numSingleKill", "numDoubleKill", "totalLPTime", "totalKTime", "avgLPTime", "avgKTime"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	headers.append("UseLp")
	csvTimeData = [headers]

	for modelType in modelTypeArray:
		for inputVoltage in inputVoltageArray:
			print ("inputVoltage", inputVoltage)
			problemName = "schmitt_"+modelType+"_input_voltage_"+str(inputVoltage)
			timeData = [problemName]

			for n in range(numTrials):
				statVars = {}
				start = time.time()
				allHypers = schmittTrigger(modelType=modelType, inputVoltage = inputVoltage, statVars=statVars, numSolutions = "all", useLp = useLp)
				end = time.time()
				numSolutions = len(allHypers)
				timeTaken = end - start
				print ("run ", n, "timeTaken", timeTaken)

				if n == 0:
					timeData.append(numSolutions)
					timeData.append(statVars["numBisection"])
					timeData.append(statVars["numLp"])
					timeData.append(statVars["numK"])
					timeData.append(statVars["numSingleKill"])
					timeData.append(statVars["numDoubleKill"])
					timeData.append(statVars["totalLPTime"])
					timeData.append(statVars["totalKTime"])
					timeData.append(statVars["avgLPTime"])
					timeData.append(statVars["avgKTime"])

				timeData.append(timeTaken)
			
			if not(useLp):
				timeData.append("False")
			else:
				timeData.append("True")
			csvTimeData.append(timeData)
					
	timeFile = open(timingFilename, 'a')
	with timeFile:
		writer = csv.writer(timeFile)
		writer.writerows(csvTimeData)

def solverInverterExperiments():
	numTrials = 1
	inputVoltageArray = [0.0, 1.0]

	modelTypeArray = ["scMosfet"]
	useLp = False
	useNewtonHypers = False

	
	timingFilename = "data/solver_time_DATE2.csv"
	headers = ["Problem", "NumSolutions", "numBisection", "numLP", "numK", "numSingleKill", "numDoubleKill", "totalLPTime", "totalKTime", "avgLPTime", "avgKTime"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	headers.append("UseLp")
	csvTimeData = [headers]

	for modelType in modelTypeArray:
		for inputVoltage in inputVoltageArray:
			print ("inputVoltage", inputVoltage)
			problemName = "inverter_"+modelType+"_input_voltage_"+str(inputVoltage)
			timeData = [problemName]

			for n in range(numTrials):
				statVars = {}
				start = time.time()
				allHypers = inverter(modelType=modelType, inputVoltage=inputVoltage, statVars=statVars, numSolutions="all" , useLp=useLp)
				end = time.time()
				numSolutions = len(allHypers)
				timeTaken = end - start
				print ("run ", n, "timeTaken", timeTaken)

				if n == 0:
					timeData.append(numSolutions)
					timeData.append(statVars["numBisection"])
					timeData.append(statVars["numLp"])
					timeData.append(statVars["numK"])
					timeData.append(statVars["numSingleKill"])
					timeData.append(statVars["numDoubleKill"])
					timeData.append(statVars["totalLPTime"])
					timeData.append(statVars["totalKTime"])
					timeData.append(statVars["avgLPTime"])
					timeData.append(statVars["avgKTime"])

				timeData.append(timeTaken)
			
			if not(useLp):
				timeData.append("False")
			else:
				timeData.append("True")
			csvTimeData.append(timeData)
					
	timeFile = open(timingFilename, 'a')
	with timeFile:
		writer = csv.writer(timeFile)
		writer.writerows(csvTimeData)


#solverInverterExperiments()
solverRambusExperiments()
#solverSchmittExperiments()
