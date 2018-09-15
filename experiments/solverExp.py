import sys
import csv
import multiprocessing

sys.path.append('../')
from prototype import *

def runRambusExperiment(timingFilename, modelType, numStages, gcc):
	useLp = False
	numTrials = 1

	headers = ["Problem", "NumSolutions", "numBisection", "numLP", "numK", "numSingleKill", "numDoubleKill", "totalLPTime", "totalKTime", "avgLPTime", "avgKTime"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	headers.append("UseLp")
	csvTimeData = [headers]

	print ("rambus modelType", modelType, "numStages", numStages, "gcc", gcc)
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

def runSchmittExperiment(timingFilename, modelType, inputVoltage):
	numTrials = 1
	useLp = False
	
	headers = ["Problem", "NumSolutions", "numBisection", "numLP", "numK", "numSingleKill", "numDoubleKill", "totalLPTime", "totalKTime", "avgLPTime", "avgKTime"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	headers.append("UseLp")
	csvTimeData = [headers]

	print ("schmitt modelType", modelType, "inputVoltage", inputVoltage)
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

def runInverterExperiment(timingFilename, modelType, inputVoltage):
	numTrials = 1
	useLp = False
	
	headers = ["Problem", "NumSolutions", "numBisection", "numLP", "numK", "numSingleKill", "numDoubleKill", "totalLPTime", "totalKTime", "avgLPTime", "avgKTime"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	headers.append("UseLp")
	csvTimeData = [headers]

	print ("inverter modelType", modelType, "inputVoltage", inputVoltage)
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



if __name__ == '__main__':
	timeout = 36000 # in seconds (10 hours)
	timingFilename = "../data/solver_time_DATE2.csv"

	# Run rambus experiments
	modelTypesL = ["tanh", "lcMosfet", "scMosfet"]
	#modelTypesL = ["tanh"]
	numStagesL = [2, 4, 6]
	#numStagesL = [2]
	gccL = [0.5, 4.0]
	for modelType in modelTypesL:
		for numStages in numStagesL:
			for gcc in gccL:	
				p = multiprocessing.Process(target=runRambusExperiment, name="RunRambusExperiment", args=(timingFilename, modelType, numStages, gcc))
				
				p.start()

				# Wait a maximum of timeout seconds for process
				# Usage: join([timeout in seconds])
				p.join(timeout)

				# If thread is active
				if p.is_alive():
					print "After 10 hours... let's kill it..."

					# Terminate process
					p.terminate()
					p.join()

	# Run schmitt trigger experiments
	modelTypesL = ["lcMosfet", "scMosfet"]
	inputVoltages = []
	for modelType in modelTypesL:
		if modelType == "lcMosfet":
			inputVoltages = [0.0, 0.9, 1.8]
		elif modelType == "scMosfet":
			inputVoltages = [0.0, 0.5, 1.0]
		for inputVoltage in inputVoltages:
			p = multiprocessing.Process(target=runSchmittExperiment, name="RunSchmittExperiment", args=(timingFilename, modelType, inputVoltage))
			p.start()

			# Wait a maximum of timeout seconds for process
			# Usage: join([timeout in seconds])
			p.join(timeout)

			# If thread is active
			if p.is_alive():
				print "After 10 hours... let's kill it..."

				# Terminate process
				p.terminate()
				p.join()

	modelTypesL = ["lcMosfet", "scMosfet"]
	inputVoltages = []
	for modelType in modelTypesL:
		if modelType == "lcMosfet":
			inputVoltages = [0.0, 1.8]
		elif modelType == "scMosfet":
			inputVoltages = [0.0, 1.0]
		for inputVoltage in inputVoltages:
			p = multiprocessing.Process(target=runInverterExperiment, name="RunSchmittExperiment", args=(timingFilename, modelType, inputVoltage))
			p.start()


			# Wait a maximum of timeout seconds for process
			# Usage: join([timeout in seconds])
			p.join(timeout)

			# If thread is active
			if p.is_alive():
				print "After 10 hours... let's kill it..."

				# Terminate process
				p.terminate()
				p.join()


