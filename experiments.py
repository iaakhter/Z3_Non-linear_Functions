import time
import csv
#from memory_profiler import memory_usage
#from prototype2 import *

def solverRambusExperiments():
	from prototype2 import *
	numTrials = 1
	numStagesArray = [2, 4]
	gccArray = [0.5, 4.0]
	modelTypeArray = ["tanh", "mosfet"]
	volThresholdArray = [0.9]

	numStagesArray = [6]
	gccArray = [0.5, 4.0]
	modelTypeArray = ["mosfet"]
	volThresholdArray = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
						0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 
						0.1, 0.3, 0.5, 0.7, 0.9, 
						0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
						0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]
	volThresholdArray = [1.0]
	useLp = True

	
	timingFilename = "data/solver_time_det2.csv"
	headers = ["Problem", "NumSolutions", "volRedThreshold", "numBisection", "numLP", "numGS", "numSingleKill", "numDoubleKill", "totalLPTime", "totalGSTime", "avgLPTime", "avgGSTime"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	#headers.append("UseLp")
	csvTimeData = [headers]

	for volThreshold in volThresholdArray:
		for modelType in modelTypeArray:
			for numStages in numStagesArray:
				for gcc in gccArray:
					print ("volThreshold", volThreshold, "modelType", modelType, "numStages", numStages, "gcc", gcc)
					problemName = "rambus_"+modelType+"_stgs_"+str(numStages)+"_gcc_"+str(gcc)
					timeData = [problemName]
					if not(useLp):
						problemName += "_useLP_false"
	
					for n in range(numTrials):
						start = time.time()
						allHypers, globalVars = rambusOscillator(modelType=modelType, numStages=numStages, g_cc=gcc, lpThreshold=volThreshold, numSolutions="all" , newtonHypers=True, useLp=useLp)
						end = time.time()
						[numBisection, numLp, numGs, numSingleKill, numDoubleKill, totalLPTime, totalGSTime, avgLPTime, avgGSTime, stringHyperList] = globalVars
						numSolutions = len(allHypers)
						timeTaken = end - start
						print ("run ", n, "timeTaken", timeTaken)

						if n == 0:
							timeData.append(numSolutions)
							timeData.append(volThreshold)
							timeData.append(numBisection)
							timeData.append(numLp)
							timeData.append(numGs)
							timeData.append(numSingleKill)
							timeData.append(numDoubleKill)
							timeData.append(totalLPTime)
							timeData.append(totalGSTime)
							timeData.append(avgLPTime)
							timeData.append(avgGSTime)
							stringTextFilename = problemName + "_volRedThreshold_"+str(volThreshold)+".txt"
							theFile = open("data/textFiles/" + stringTextFilename, 'w')
							for item in stringHyperList:
								#print (item, type(item[1]))
								stringItem = ""
								if type(item[1]) != list:
									stringItem += item[0] + ": " + str(item[1]) + "\n"
									#theFile.write("%s, %s%\n" % item[0], str(item[1]))
								else:
									stringItem += item[0] + ": (" + str(item[1][0]) +  ", " + str(item[1][0]) + ")\n"
									#theFile.write("%s, [%s, %s]%\n" % item[0], str(item[1][0]), str(item[1][1]))
								theFile.write(stringItem)
							theFile.close()

						timeData.append(timeTaken)
					
					#timeData.append("False")

					csvTimeData.append(timeData)
					
	timeFile = open(timingFilename, 'a')
	with timeFile:
	    writer = csv.writer(timeFile)
	    writer.writerows(csvTimeData)


def solverSchmittExperiments():
	from prototype2 import *
	numTrials = 1
	inputVoltageArray = [0.0, 1.0, 1.8]
	volThresholdArray = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
						0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 
						0.1, 0.3, 0.5, 0.7, 0.9, 
						0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
						0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]

	volThresholdArray = [1.0]
	useLp = True

	
	timingFilename = "data/solver_time_det2.csv"
	headers = ["Problem", "NumSolutions", "volRedThreshold", "numBisection", "numLP", "numGS", "numSingleKill", "numDoubleKill"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	csvTimeData = [headers]

	for inputVoltage in inputVoltageArray:
		for volThreshold in volThresholdArray:
			print ("inputVoltage", inputVoltage, "volThreshold", volThreshold)
			problemName = "schmitt_input_voltage_"+str(inputVoltage)
			timeData = [problemName]

			for n in range(numTrials):
				start = time.time()
				allHypers, globalVars = schmittTrigger(inputVoltage, volThreshold, numSolutions = "all", newtonHypers = True, useLp = useLp)
				end = time.time()
				[numBisection, numLp, numGs, numSingleKill, numDoubleKill, stringHyperList] = globalVars
				numSolutions = len(allHypers)
				timeTaken = end - start
				print ("run ", n, "timeTaken", timeTaken)

				if n == 0:
					timeData.append(numSolutions)
					timeData.append(volThreshold)
					timeData.append(numBisection)
					timeData.append(numLp)
					timeData.append(numGs)
					timeData.append(numSingleKill)
					timeData.append(numDoubleKill)
					stringTextFilename = problemName + "_volRedThreshold_"+str(volThreshold)+".txt"
					if not(useLp):
						stringTextFilename += "_useLP_false"
					theFile = open("data/textFiles/" + stringTextFilename, 'w')
					for item in stringHyperList:
						#print (item, type(item[1]))
						stringItem = ""
						if type(item[1]) != list:
							stringItem += item[0] + ": " + str(item[1]) + "\n"
							#theFile.write("%s, %s%\n" % item[0], str(item[1]))
						else:
							stringItem += item[0] + ": (" + str(item[1][0]) +  ", " + str(item[1][0]) + ")\n"
							#theFile.write("%s, [%s, %s]%\n" % item[0], str(item[1][0]), str(item[1][1]))
						theFile.write(stringItem)
					theFile.close()

				timeData.append(timeTaken)
			
			if not(useLp):
				timeData.append("False")
			csvTimeData.append(timeData)
					
	timeFile = open(timingFilename, 'a')
	with timeFile:
	    writer = csv.writer(timeFile)
	    writer.writerows(csvTimeData)

def solverBenchmarkExperiments():
	from prototype2 import *
	numTrials = 1
	problemTypeArray = ["dRealExample", "meti25", "meti18"]
	volThresholdArray = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
						0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 
						0.1, 0.3, 0.5, 0.7, 0.9, 
						0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
						0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]

	volThresholdArray = [1.0]
	useLp = True

	
	timingFilename = "data/solver_time_det2.csv"
	headers = ["Problem", "NumSolutions", "volRedThreshold", "numBisection", "numLP", "numGS", "numSingleKill", "numDoubleKill", "totalLPTime", "totalGSTime", "avgLPTime", "avgGSTime"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	if not(useLp):
		headers.append("UseLp")
	csvTimeData = [headers]

	for problemType in problemTypeArray:
		for volThreshold in volThresholdArray:
			print ("problemType", problemType)
			timeData = ["benchmark_problem_"+problemType]

			for n in range(numTrials):
				start = time.time()
				allHypers, globalVars = singleVariableInequalities(problemType, volThreshold, numSolutions="all", newtonHypers=True, useLp=useLp)
				end = time.time()
				timeTaken = end - start
				print ("run ", n, "timeTaken", timeTaken)
				[numBisection, numLp, numGs, numSingleKill, numDoubleKill, totalLPTime, totalGSTime, avgLPTime, avgGSTime, stringHyperList] = globalVars
				numSolutions = len(allHypers)

				if n == 0:
					timeData.append(numSolutions)
					timeData.append(volThreshold)
					timeData.append(numBisection)
					timeData.append(numLp)
					timeData.append(numGs)
					timeData.append(numSingleKill)
					timeData.append(numDoubleKill)
					timeData.append(totalLPTime)
					timeData.append(totalGSTime)
					timeData.append(avgLPTime)
					timeData.append(avgGSTime)
					stringTextFilename = problemType + "_volRedThreshold_"+str(volThreshold)+".txt"
					if not(useLp):
						stringTextFilename += "_useLP_false"
					theFile = open("data/textFiles/" + stringTextFilename, 'w')
					for item in stringHyperList:
						#print (item, type(item[1]))
						stringItem = ""
						if type(item[1]) != list:
							stringItem += item[0] + ": " + str(item[1]) + "\n"
							#theFile.write("%s, %s%\n" % item[0], str(item[1]))
						else:
							stringItem += item[0] + ": (" + str(item[1][0]) +  ", " + str(item[1][0]) + ")\n"
							#theFile.write("%s, [%s, %s]%\n" % item[0], str(item[1][0]), str(item[1][1]))
						theFile.write(stringItem)
					theFile.close()

				timeData.append(timeTaken)
			
			if not(useLp):
				timeData.append("False")
			csvTimeData.append(timeData)
					
	timeFile = open(timingFilename, 'a')
	with timeFile:
	    writer = csv.writer(timeFile)
	    writer.writerows(csvTimeData)


# started at 3:21
def dRealRambusExperiments():
	from dRealRambus import *
	numTrials = 1
	numStagesArray = [2, 4]
	gccArray = [0.5, 4.0]
	modelTypeArray = ["tanh", "mosfet"]

	numStagesArray = [4]
	gccArray = [0.5]
	modelTypeArray = ["mosfet"]

	
	timingFilename = "data/dreal_time.csv"
	headers = ["Problem", "NumSolutions"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	csvTimeData = [headers]

	for modelType in modelTypeArray:
		for numStages in numStagesArray:
			for gcc in gccArray:
				print ("modelType", modelType, "numStages", numStages, "gcc", gcc)
				timeData = ["rambus_"+modelType+"_stgs_"+str(numStages)+"_gcc_"+str(gcc)]

				for n in range(numTrials):
					start = time.time()
					if modelType == "tanh":
						allHypers = rambusOscillatorTanh(-5.0, numStages=numStages, numSolutions = "all", g_cc = gcc)
					elif modelType == "mosfet":
						allHypers = rambusOscillatorMosfet(Vtp = -0.4, Vtn = 0.4, Vdd = 1.8, Kn = 270*1e-6, Kp = -90*1e-6, Sn = 8/3.0, numStages = numStages, numSolutions = "all", g_cc = gcc)
					end = time.time()
					numSolutions = len(allHypers)
					timeTaken = end - start
					print ("run ", n, "timeTaken", timeTaken)

					if n == 0:
						timeData.append(numSolutions)

					timeData.append(timeTaken)
				
				csvTimeData.append(timeData)
					
	timeFile = open(timingFilename, 'a')
	with timeFile:
	    writer = csv.writer(timeFile)
	    writer.writerows(csvTimeData)


def dRealSchmittExperiments():
	from dRealSchmitt import *
	numTrials = 3
	inputVoltageArray = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
	inputVoltageArray = [0.0, 1.0, 1.8]
	timingFilename = "data/dReal_time.csv"
	headers = ["Problem", "NumSolutions"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	csvTimeData = [headers]

	for inputVoltage in inputVoltageArray:
		print ("inputVoltage", inputVoltage)
		timeData = ["schmitt_input_voltage_"+str(inputVoltage)]

		for n in range(numTrials):
			start = time.time()
			allHypers = schmittTrigger(inputVoltage = inputVoltage, Vtp = -0.4, Vtn = 0.4, Vdd = 1.8, Kn = 270*1e-6, Kp = -90*1e-6, Sn = (8/3.0), numSolutions = "all")
			end = time.time()
			numSolutions = len(allHypers)
			timeTaken = end - start
			print ("run ", n, "timeTaken", timeTaken)

			if n == 0:
				timeData.append(numSolutions)

			timeData.append(timeTaken)
		

		csvTimeData.append(timeData)
					
	timeFile = open(timingFilename, 'a')
	with timeFile:
	    writer = csv.writer(timeFile)
	    writer.writerows(csvTimeData)


def dRealBenchmarkExperiments():
	from dRealBenchmarks import *
	numTrials = 3
	problemTypeArray = ["dRealExample", "meti25", "meti18"]
	
	
	timingFilename = "data/dreal_time.csv"
	headers = ["Problem"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	csvTimeData = [headers]

	for problemType in problemTypeArray:
		print ("problemType", problemType)
		timeData = ["benchmark_problem_"+problemType]

		for n in range(numTrials):
			start = time.time()
			if problemType == "dRealExample":
				exampleFun(numSolutions = 1)
			if problemType == "meti25":
				meti25(numSolutions = 1)
			if problemType == "meti18":
				meti18(numSolutions = 1)
			end = time.time()
			timeTaken = end - start
			print ("run ", n, "timeTaken", timeTaken)


			timeData.append(timeTaken)
		

		csvTimeData.append(timeData)
					
	timeFile = open(timingFilename, 'a')
	with timeFile:
	    writer = csv.writer(timeFile)
	    writer.writerows(csvTimeData)


#1:27
def z3RambusExperiments():
	from z3Rambus import *
	numTrials = 1
	numStagesArray = [2, 4]
	gccArray = [0.5, 4.0]
	modelTypeArray = ["tanh", "mosfet"]

	numStagesArray = [2]
	gccArray = [0.5]
	modelTypeArray = ["tanh"]

	
	timingFilename = "data/z3_rambus.csv"
	headers = ["Problem", "NumSolutions"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	csvTimeData = [headers]

	for modelType in modelTypeArray:
		for numStages in numStagesArray:
			for gcc in gccArray:
				print ("modelType", modelType, "numStages", numStages, "gcc", gcc)
				timeData = ["rambus_"+modelType+"_stgs_"+str(numStages)+"_gcc_"+str(gcc)]

				for n in range(numTrials):
					start = time.time()
					if modelType == "tanh":
						allHypers = rambusOscillatorTanh(-5.0, numStages=numStages, numSolutions = "all", g_cc = gcc)
					elif modelType == "mosfet":
						allHypers = rambusOscillatorMosfet(Vtp = -0.4, Vtn = 0.4, Vdd = 1.8, Kn = 270*1e-6, Kp = -90*1e-6, Sn = 8/3.0, numStages = numStages, numSolutions = "all", g_cc = gcc)
					end = time.time()
					numSolutions = len(allHypers)
					timeTaken = end - start
					print ("run ", n, "timeTaken", timeTaken)

					if n == 0:
						timeData.append(numSolutions)

					timeData.append(timeTaken)
				
				csvTimeData.append(timeData)
					
	timeFile = open(timingFilename, 'a')
	with timeFile:
	    writer = csv.writer(timeFile)
	    writer.writerows(csvTimeData)


def z3SchmittExperiments():
	from z3Schmitt import *
	numTrials = 3
	inputVoltageArray = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
	
	timingFilename = "z3_time_schmitt.csv"
	headers = ["Problem", "NumSolutions"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	csvTimeData = [headers]

	for inputVoltage in inputVoltageArray:
		print ("inputVoltage", inputVoltage)
		timeData = ["schmitt_input_voltage_"+str(inputVoltage)]

		for n in range(numTrials):
			start = time.time()
			allHypers = schmittTrigger(inputVoltage = inputVoltage, Vtp = -0.4, Vtn = 0.4, Vdd = 1.8, Kn = 270*1e-6, Kp = -90*1e-6, Sn = (8/3.0), numSolutions = "all")
			end = time.time()
			numSolutions = len(allHypers)
			timeTaken = end - start
			print ("run ", n, "timeTaken", timeTaken)

			if n == 0:
				timeData.append(numSolutions)

			timeData.append(timeTaken)
		

		csvTimeData.append(timeData)
					
	timeFile = open(timingFilename, 'a')
	with timeFile:
	    writer = csv.writer(timeFile)
	    writer.writerows(csvTimeData)


def solverRambusExperimentsMemory():
	numTrials = 3
	numStagesArray = [2, 4]
	gccArray = [0.5, 4.0]
	modelTypeArray = ["tanh", "mosfet"]
	lpThresholdArray = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

	numStagesArray = [2, 4]
	gccArray = [0.5, 4.0]
	modelTypeArray = ["tanh"]
	lpThresholdArray = [0.3]

	
	memoryFilename = "solver_mem.csv"
	headers = ["Problem", "NumSolutions", "LPThreshold"]
	for i in range(numTrials):
		headers.append("Run_"+str(i))
	csvMemData = [headers]

	for lpThreshold in lpThresholdArray:
		for modelType in modelTypeArray:
			for numStages in numStagesArray:
				for gcc in gccArray:
					print ("lpThreshold", lpThreshold, "modelType", modelType, "numStages", numStages, "gcc", gcc)
					memData = ["rambus_"+modelType+"_stgs_"+str(numStages)+"_gcc_"+str(gcc)]
					allHypers = rambusOscillator(modelType=modelType, numStages=numStages, g_cc=gcc, lpThreshold=lpThreshold, numSolutions="all" , newtonHypers=True, useLp=True)
					numSolutions = len(allHypers)
					memData.append(numSolutions)
					memData.append(lpThreshold)
					mem = max(memory_usag((rambusOscillator, {'modelType':modelType}, {'numStages':numStages}, {'g_cc':gcc}, {'lpThreshold':lpThreshold}, 
						{'numSolutions': "all"}, {'newtonHypers':True}, {'useLp':True})))

					memData.append(mem)
					csvMemData.append(memData)
					
	memFile = open(memoryFilename, 'a')
	with memFile:
	    writer = csv.writer(memFile)
	    writer.writerows(csvMemData)


solverRambusExperiments()
#solverSchmittExperiments()
#solverBenchmarkExperiments()
#dRealRambusExperiments()
#dRealSchmittExperiments()
#dRealBenchmarkExperiments()
#z3RambusExperiments()

