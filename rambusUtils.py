import numpy as np
import intervalUtils

'''
Return true if stable and false otherwise
'''
def determineStability(equilibrium,model):
	jac = model.jacobian(equilibrium)
	print ("equilibrium", equilibrium)
	print ("jac", jac)
	eigVals,_ = np.linalg.eig(jac)
	maxEig = np.amax(eigVals.real)
	if maxEig > 0:
		return False
	return True


'''
Divide solutions according to equivalence classes.
Categorize solutions as stable or unstable
'''
def categorizeSolutions(allHypers,model):
	sampleSols = []
	rotatedSols = {}
	stableSols = []
	unstableSols = []
	allSols = []
	for hyper in allHypers:
		exampleSoln = (hyper[:,0] + hyper[:,1])/2.0
		lenV = len(exampleSoln)
		finalSoln = intervalUtils.newton(model,exampleSoln)
		#print "exampleSoln ", exampleSoln
		#print "finalSoln ", finalSoln
		stable = determineStability(finalSoln[1],model)
		if stable:
			stableSols.append(finalSoln[1])
		else:
			unstableSols.append(finalSoln[1])
		allSols.append(finalSoln[1])
		
		# Classify the solutions into equivalence classes
		if len(sampleSols) == 0:
			sampleSols.append(finalSoln[1])
			rotatedSols[0] = []
		else:
			foundSample = False
			for si in range(len(sampleSols)):
				sample = sampleSols[si]
				for ii in range(lenV):
					if abs(finalSoln[1][0] - sample[ii]) < 1e-8:
						rotatedSample = np.zeros_like(finalSoln[1])
						for ri in range(lenV):
							rotatedSample[ri] = sample[(ii+ri)%lenV]
						if np.less_equal(np.absolute(rotatedSample - finalSoln[1]), np.ones((lenV))*1e-8 ).all():
							foundSample = True
							rotatedSols[si].append(ii)
							break
				if foundSample:
					break

			if foundSample == False:
				sampleSols.append(finalSoln[1])
				rotatedSols[len(sampleSols)-1] = []


	return sampleSols, rotatedSols, stableSols, unstableSols

