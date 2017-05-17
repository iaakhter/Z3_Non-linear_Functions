def checkIfSolnUniqueInHyperrect(a,g_fwd,g_cc,hyperRectangle,sol,alpha=0.1,r=1.0):
	lowerBound = array(hyperRectangle[0])
	upperBound = array(hyperRectangle[1])
	distFromLowerBound = linalg.norm(lowerBound - sol)
	distFromUpperBound = linalg.norm(upperBound - sol)
	maxDist = max(distFromLowerBound,distFromUpperBound)
	jac = getJacobian(sol,a,g_cc,g_fwd)
	#get the smallest singular value of
	_,sigma,_ = linalg.svd(jac)
	print "sigma Array: ", sigma
	sigma = sigma[len(sigma)-1]

	r = 1.0

	# is this correct?
	alpha = linalg.norm((exp(1) - 2)*(exp(sol)))
	ratio = sigma/alpha

	print "sigma ", sigma
	print "alpha ", alpha
	print "ratio ", ratio
	print "maxDist ", maxDist
	if maxDist < ratio:
		return True
	return False

def ifFoundAllSolutions(VlowVhighs,a,g_fwd,g_cc,sols,alpha=0.1,r=1.0):
	I = RealVector('I',len(sols[0]))
	V = RealVector('V',len(sols[0]))
	s = Solver()

	for sol in sols:
		jac = getJacobian(sol,a,g_cc,g_fwd)
		#get the smallest singular value of
		_,sigma,_ = linalg.svd(jac)
		sigma = sigma[len(sigma)-1]
		r = 1.0

		# is this correct?
		alpha = linalg.norm((exp(1) - 2)*(exp(sol)))
		ratio = sigma/alpha
		s.add(Or(*[Or(V[i] < sol[i] - ratio, 
							V[i] > sol[i] + ratio) for i in range(len(V))]))

	s.push()
	oscl(s,I,V,a,VlowVhighs,g_cc,g_fwd)
	is_equilibrium = And(*[I[i]==0 for i in range(len(V))])
	s.add(is_equilibrium)
	ch = s.check()

	if ch==sat:
		print "Found solutions not previously found"
		return True
	else:
		print "All solutions have been found"
		return False
					
def plotSolutionsWithinHyperRectangles(I,V,a,g_fwd,g_cc,allHyperRectangles,voltageNumber):
	solv = Solver()

	for hR in range(len(allHyperRectangles)):
		print "Visualizing hyperrectangle ", hR
		hyperRectangle = allHyperRectangles[hR]
		print "lower bounds ", hyperRectangle[0]
		print "upper bounds ",hyperRectangle[1]

		#print "starting HyperRectangle: "
		#print hyperRectangle
		solv.push()
		distance = [(hyperRectangle[1][i] - hyperRectangle[0][i]) for i in range(len(V))]
		osclRefine(solv,I,V,a,hyperRectangle,g_cc,g_fwd)
		solv.add(And(*[And(V[i]>=hyperRectangle[0][i], V[i]<=hyperRectangle[1][i]) for i in range(len(V))]))
		is_equilibrium = And(*[I[i]==0 for i in range(len(V))])
		solv.add(is_equilibrium)
		ch = solv.check()

		#print "ch ", ch
		if ch == sat:
			solVoltArray = range(0,len(V))
			m = solv.model()
			for d in m.decls():
				dName = str(d.name())
				firstLetter = dName[0]
				if(dName[0]=="V" and dName[1]=="_"):
					index = int(dName[len(dName) - 1])
					val = float(Fraction(str(m[d])))
					#print "index: ", index, " val: ", val
					solVoltArray[index] = val
			inpt = arange(-2.0,2.0,0.01)
			output = tanhFun(a,inpt)
			lowVolt = hyperRectangle[0][voltageNumber]
			highVolt = hyperRectangle[1][voltageNumber]
			tanhFunVlow = tanhFun(a,lowVolt)
			tanhFunVhigh = tanhFun(a,highVolt)
			dLow = tanhFunder(a,lowVolt)
			dHigh = tanhFunder(a,highVolt)
			diff = highVolt-lowVolt
			if(diff == 0):
				diff = 1e-10
			dThird = (tanhFunVhigh - tanhFunVlow)/diff
			cLow = tanhFunVlow - dLow*lowVolt
			cHigh = tanhFunVhigh - dHigh*highVolt
			cThird = tanhFunVlow - dThird*lowVolt

			triInp1 = arange(lowVolt,highVolt,0.0001)
			triOutputLow = dLow*triInp1 + cLow
			triOutputHigh = dHigh*triInp1 + cHigh
			triOutputThird = dThird*triInp1 + cThird

			print "solVoltArray[0]: ", solVoltArray[voltageNumber]
			
			plt.figure(hR)
			plt.plot(inpt,output,'b',
				triInp1,triOutputLow,'r',
				triInp1,triOutputHigh,'r',
				triInp1,triOutputThird,'r',
				solVoltArray[voltageNumber],tanhFun(a,solVoltArray[voltageNumber]),'k*')
			solv.pop()
	plt.show()

'''def checkExistenceOfSolution(a,g_fwd,g_cc,hyperRectangle):
	print "lower bounds ", hyperRectangle[0]
	print "upper bounds ",hyperRectangle[1]

	#TODO need to take care/check if it matters that the jacobian has
	#zero elements
	jac = getJacobianInterval(hyperRectangle,a,g_cc,g_fwd)
	jacIv = iv.matrix(len(hyperRectangle[0]),len(hyperRectangle[1]))
	for i in range(len(hyperRectangle[0])):
		for j in range(len(hyperRectangle[1])):
			jacIv[i,j] = iv.mpf([jac[i,j,0],jac[i,j,1]])


	#inverseJac = getInverseOfIntervalMatrix(jac,0.1)
	#print "inverseJac"
	#print inverseJac

	print "jacIv"
	print jacIv
	midPoint = array([(hyperRectangle[0][i]+hyperRectangle[1][i])/2.0 for i in range(len(hyperRectangle[0]))])
	_,_,IMidPoint = array(oscNum(midPoint,a,g_cc,g_fwd))
	#nInterval1 = midPoint - dot(inverseJac[:,:,0],IMidPoint)
	#nInterval2 = midPoint - dot(inverseJac[:,:,1],IMidPoint)

	nInterval1 = midPoint - iv.lu_solve(jacIv[:,:,0],IMidPoint)
	nInterval2 = midPoint - iv.lu_solve(jacIv[:,:,1],IMidPoint)
	newtonInterval = zeros((len(nInterval1),2))
	for i in range(len(nInterval1)):
		newtonInterval[i][0] = min(nInterval1[i],nInterval2[i])
		newtonInterval[i][1] = max(nInterval1[i],nInterval2[i])

	print "newtonLowerBounds ", newtonInterval[:,0]
	print "newtonUpperBounds ", newtonInterval[:,1]

	for i in range(len(hyperRectangle[0])):
		if newtonInterval[i][0] < hyperRectangle[0][i] or newtonInterval[i][0] > hyperRectangle[1][i]:
			return False
		if newtonInterval[i][1] < hyperRectangle[0][i] or newtonInterval[i][1] > hyperRectangle[1][i]:
			return False
	return True'''

def findSolWithNewtons(a,g_fwd,g_cc,hyperRectangle):
	print "lower bounds ", hyperRectangle[0]
	print "upper bounds ",hyperRectangle[1]
	sol = [(hyperRectangle[1][i] + hyperRectangle[0][i])/2.0 for i in range(len(hyperRectangle[0]))]
	print "starting solution ", sol
	Inum = array(oscNum(sol,a,g_cc,g_fwd))
	oldInum = copy.deepcopy(Inum)
	iter = 0
	while(all(abs(Inum) > 1.0e-6)):
		jac = getJacobian(sol,a,g_cc,g_fwd)
		sol = sol - linalg.solve(jac,Inum)
		oldInum = copy.deepcopy(Inum)
		Inum = array(oscNum(sol,a,g_cc,g_fwd))
		print "jac ", jac
		print "sol ", sol
		print "Inum ", Inum
		print ""
		iter+=1
		if iter>7:
			sol = None
			break
	return sol

# taken from http://repository.uwyo.edu/cgi/viewcontent.cgi?article=1468&context=ela
# Inverse Interval Matrix: A Survey by Jiri Rohn and Raena Farhadsefat
def getInverseOfIntervalMatrix(mat,bound):
	'''numVolts = mat.shape[0]
	inverse = zeros((numVolts,numVolts,2))
	upperBounds = mat[:,:,1]
	lowerBounds = mat[:,:,0]
	midMat = (upperBounds + lowerBounds)/2.0
	delta = midMat - lowerBounds
	midMatInverse = linalg.inv(midMat)
	midMatInverseAbs = absolute(midMatInverse)
	I = identity(numVolts)
	M = linalg.inv(I - dot(midMatInverseAbs,delta))
	print "midMat"
	print midMat
	print "midMatInverse"
	print midMatInverse
	print "delta"
	print delta
	print "M"
	print M
	Tu = zeros((numVolts,numVolts))
	for i in range(numVolts):
		Tu[i,i] = M[i,i]
	Tv = linalg.inv(2*Tu - I)

	BlowTilda = -dot(M,midMatInverseAbs)+dot(Tu,(midMatInverse + midMatInverseAbs))
	BhighTilda = dot(M,midMatInverseAbs)+dot(Tu,(midMatInverse - midMatInverseAbs))
	print "BlowTilda"
	print BlowTilda
	print "multBlow"
	print dot(Tv,BlowTilda)

	print "BhighTilda"
	print BhighTilda
	print "multhigh"
	print dot(Tv,BhighTilda)
	inverse[:,:,0] = minimum(BlowTilda,dot(Tv,BlowTilda))
	inverse[:,:,1] = maximum(BhighTilda,dot(Tv,BhighTilda))
	spectralRadius = linalg.eigvals(dot(midMatInverseAbs,delta))
	print "spectralRadius: ", spectralRadius
	return inverse'''
	return iv.inverse(mat)