# Itrat Ahmed Akhter
# CPSC 538G
# Final Project
# z3lib.py

'''
A library of functions that model the Rambus mobius ring oscillator in
Z3 so that Z3 can find the DC quilibrium points.
'''

from z3 import *
from numpy import *
import copy
import matplotlib.pyplot as plt

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

'''
A function that returns the absolute value of x.
'''
def myAbs(x):
	return If(x >= 0,x,-x)


'''
takes in non-symbolic python values
calculates the tanhFun of val
'''
def tanhFun(a,val):
	return -tanh(a*val)
	#return -(exp(a*val) - exp(-a*val))/(exp(a*val) + exp(-a*val))

'''
takes in non-symbolic python values
calculates the derivative of tanhFun of val
'''
def tanhFunder(a,val):
	den = cosh(a*val)*cosh(a*val)
	#print "den ", den
	return -a/(cosh(a*val)*cosh(a*val))
	#return (-4.0*a)/((exp(a*val) + exp(-a*val)) * (exp(a*val) + exp(-a*val)))


'''simulate oscillation with python numberical values to check if the
z3 solutions make sense'''
def oscNum(V,a,g_cc,g_fwd = 1):
	lenV = len(V)
	Vin = [V[i % lenV] for i in range(-1,lenV-1)]
	Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
	return [((tanhFun(a,Vin[i])-V[i])*g_fwd
			+(tanhFun(a,Vcc[i])-V[i])*g_cc) for i in range(lenV)]


'''Get jacobian of rambus oscillator at V
'''
def getJacobian(V,a,g_cc,g_fwd = 1):
	lenV = len(V)
	Vin = [V[i % lenV] for i in range(-1,lenV-1)]
	Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
	jacobian = zeros((lenV, lenV))
	for i in range(lenV):
		jacobian[i,i] = -(g_fwd+g_cc)
		jacobian[i,(i-1)%lenV] = g_fwd*tanhFunder(a,V[i%lenV])
		jacobian[i,(i + lenV/2) % lenV] = g_cc*tanhFunder(a,V[(i + lenV/2) % lenV])

	return jacobian

def triangleBounds(a, Vin, Vout, Vlow, Vhigh):
	tanhFunVlow = tanhFun(a,Vlow)
	tanhFunVhigh = tanhFun(a,Vhigh)
	dLow = tanhFunder(a,Vlow)
	dHigh = tanhFunder(a,Vhigh)
	diff = Vhigh - Vlow
	if(diff == 0):
		diff = 1e-10
	dThird = (tanhFunVhigh - tanhFunVlow)/diff
	cLow = tanhFunVlow - dLow*Vlow
	cHigh = tanhFunVhigh - dHigh*Vhigh
	cThird = tanhFunVlow - dThird*Vlow

	if Vlow <= 0 and Vhigh <=0:
		return Implies(And(Vin >= Vlow, Vin <= Vhigh),
						And(Vout >= dThird*Vin + cThird,
							Vout <= dLow*Vin + cLow,
							Vout <= dHigh*Vin + cHigh))

	elif Vlow >=0 and Vhigh >=0:
		return Implies(And(Vin >= Vlow, Vin <= Vhigh),
						And(Vout <= dThird*Vin + cThird,
							Vout >= dLow*Vin + cLow,
							Vout >= dHigh*Vin + cHigh))

def trapezoidBounds(a,Vin,Vout, Vlow, Vhigh):
	tanhFunVlow = tanhFun(a,Vlow)
	tanhFunVhigh = tanhFun(a,Vhigh)
	dLow = tanhFunder(a,Vlow)
	dHigh = tanhFunder(a,Vhigh)
	diff = Vhigh - Vlow
	if(diff == 0):
		diff = 1e-10
	dThird = (tanhFunVhigh - tanhFunVlow)/diff
	cLow = tanhFunVlow - dLow*Vlow
	cHigh = tanhFunVhigh - dHigh*Vhigh
	cThird = tanhFunVlow - dThird*Vlow

	if Vlow <= 0 and Vhigh <= 0:
		return Implies(Vin < Vlow, 
						And(Vout < dLow*Vin + cLow,
							Vout <= 1,
							Vout > tanhFunVlow))
	elif Vlow >= 0 and Vhigh >=0:
		return Implies(Vin > Vhigh, 
						And(Vout > dHigh*Vin + cHigh,
							Vout >= - 1,
							Vout <  tanhFunVhigh))



# s is solver. I and V are an array of Z3 symbolic  values.
# Vlow and Vhigh are an array of python numberical ints.
# This function simulates mobius oscillator
'''def oscl(s,I,V,a,Vlow,Vhigh,g_cc,g_fwd = 1):
	lenV = len(V)
	VoutFwd = RealVector('VoutFwd',lenV)
	VoutCc = RealVector('VoutCc',lenV)
	Vin = [V[i % lenV] for i in range(-1,lenV-1)]
	Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
	for i in range(len(Vin)):
		inverter(s,a,Vin[i],VoutFwd[i],Vlow[i],Vhigh[i])
	for i in range(len(Vcc)):
		inverter(s,a,Vcc[i],VoutCc[i],Vlow[i],Vhigh[i])
	s.add(And(*[I[i] == (VoutFwd[i] - V[i])*g_fwd + (VoutCc[i] - V[i])*g_cc for i in range(lenV)]))'''

def oscl(s,I,V,a,VlowVhighs,g_cc,g_fwd = 1):
	lenV = len(V)
	VoutFwd = RealVector('VoutFwd',lenV)
	VoutCc = RealVector('VoutCc',lenV)
	Vin = [V[i % lenV] for i in range(-1,lenV-1)]
	Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
	allVlowVhighs = []
	for i in range(lenV):
		allVlowVhighs.append(VlowVhighs)
	for i in range(lenV):
		for j in range(len(allVlowVhighs[i])):
			for k in range(len(allVlowVhighs[i])):
				boundin = allVlowVhighs[(i-1)%lenV][j]
				boundcc = allVlowVhighs[(i+lenV/2)%lenV][k]
				claimFwd = triangleBounds(a,Vin[i],VoutFwd[i],boundin[0][i],boundin[1][i])
				claimCc = triangleBounds(a,Vcc[i],VoutCc[i],boundcc[0][i],boundcc[1][i])
				s.add(claimFwd)
				s.add(claimCc)
				if j==0:
					claimTrapFwd1 = trapezoidBounds(a, Vin[i], VoutFwd[i], boundin[0][i], boundin[1][i])
					s.add(claimTrapFwd1)
				if k==0:
					claimTrapCc1 = trapezoidBounds(a, Vcc[i], VoutCc[i], boundcc[0][i], boundcc[1][i])
					s.add(claimTrapCc1)
				if j==len(allVlowVhighs[i])-1:
					claimTrapFwd2 = trapezoidBounds(a, Vin[i], VoutFwd[i], boundin[0][i], boundin[1][i])
					s.add(claimTrapFwd2)
				if k==len(allVlowVhighs[i])-1:
					claimTrapCc2 = trapezoidBounds(a, Vcc[i], VoutCc[i], boundcc[0][i], boundcc[1][i])
					s.add(claimTrapCc2)

		s.add(I[i] == (VoutFwd[i] - V[i])*g_fwd + (VoutCc[i] - V[i])*g_cc)

def osclRefine(s,I,V,a,hyperRectangles,g_cc,g_fwd = 1):
	lenV = len(V)
	VoutFwd = RealVector('VoutFwd',lenV)
	VoutCc = RealVector('VoutCc',lenV)
	Vin = [V[i % lenV] for i in range(-1,lenV-1)]
	Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
	for i in range(lenV):
		for j in range(len(hyperRectangles)):
			hyperRectangle = hyperRectangles[j]
			boundin = [hyperRectangle[0][(i-1) % lenV],hyperRectangle[1][(i-1) % lenV]]
			boundcc = [hyperRectangle[0][(i+lenV/2)%lenV],hyperRectangle[1][(i+lenV/2)%lenV]]
			#print "boundin ", boundin
			claimFwd = triangleBounds(a,Vin[i],VoutFwd[i],boundin[0],boundin[1])
			claimCc = triangleBounds(a,Vcc[i],VoutCc[i],boundcc[0],boundcc[1])
			s.add(claimFwd)
			s.add(claimCc)

		s.add(I[i] == (VoutFwd[i] - V[i])*g_fwd + (VoutCc[i] - V[i])*g_cc)

def findScale(I,V,a,VlowVhighs,g_fwd,g_cc):
	print "Finding Scale"
	opt = Optimize()
	oscl(opt,I,V,a,VlowVhighs,g_cc,g_fwd)
	claim = Or(*[I[i]!=0 for i in range(len(V))])
	opt.add(Not(claim))
	lowerBounds = []
	upperBounds = []
	for i in range(len(V)):
		print "i: ", i
		opt.push()
		optMin = opt.minimize(V[i])
		opt.check()
		minVal = float(Fraction(str(opt.lower(optMin))))
		opt.pop()
		opt.push()
		optMax = opt.maximize(V[i])
		opt.check()
		maxVal = float(Fraction(str(opt.upper(optMax))))
		opt.pop()
		lowerBounds.append(minVal)
		upperBounds.append(maxVal)
	print "lowerBounds ", lowerBounds
	print "upperbounds ", upperBounds
	print ""
	return [lowerBounds,upperBounds]
	

# s is solver. I and V are an array of Z3 symbolic  values.
# Vlow and Vhigh are an array of python numberical ints.
# This function finds hyper rectangles containing DC equilibrium points 
# for our oscillator model. Each hyper rectangle has length and width
# equalling distance and is stored in allHyperRectangles 
def findHyper(I,V,a,VlowVhighs,g_fwd,g_cc,distances):
	allHyperRectangles = []
	s = Solver()
	print "Finding HyperRectangles"
	count = 0
	while(True):
		#simulate the oscillator and check for solution
		print "count: ", count
		count += 1
		s.push()
		oscl(s,I,V,a,VlowVhighs,g_cc,g_fwd)
		is_equilibrium = And(*[I[i]==0 for i in range(len(V))])
		s.add(is_equilibrium)
		'''s.push()
		python_syntax_is_silly = True
		if(python_syntax_is_silly):
			s.add(And(*[Or(V[i] == -0.3024*(1 - 2*(i % 2)),V[i] == 0.3024*(1 - 2*(i % 2))) for i in range(len(V))]))
			ch = s.check()
			if(ch == sat):
				print "s accepts the expected solution"
			else:
				print "s rejects the expected solution"
				print s
				return s
		s.pop()
		if count==2:
			return'''

		#print "solver "
		#print s
		ch = s.check()
		if(ch == sat):
			VoutFwd = range(0,len(V))
			VoutCc = range(0,len(V))
			lowVoltArray = range(0,len(V))
			highVoltArray = range(0,len(V))
			solVoltArray = range(0,len(V))
			m = s.model()
			#print "refuted, here's a counter-example"
			#print "solution number ", count
			#print "  " + str(m)
			for d in m.decls():
				dName = str(d.name())
				firstLetter = dName[0]
				if(dName[0]=="V" and dName[1]=="_"):
					index = int(dName[len(dName) - 1])
					val = float(Fraction(str(m[d])))
					#print "index: ", index, " val: ", val
					solVoltArray[index] = val
					lowVoltArray[index] = val-distances[index]
					highVoltArray[index] = val+distances[index]
					if(lowVoltArray[index] < 0 and highVoltArray[index]>0):
						if(val >= 0):
							lowVoltArray[index] = 0.0
						elif(val < 0):
							highVoltArray[index] = 0.0

				elif(dName[0]=="V" and dName[4]=="F"):
					index = int(dName[len(dName) - 1])
					val = float(Fraction(str(m[d])))
					#print "index: ", index, " val: ", val
					VoutFwd[index] = val

				elif(dName[0]=="V" and dName[4]=="C"):
					index = int(dName[len(dName) - 1])
					val = float(Fraction(str(m[d])))
					#print "index: ", index, " val: ", val
					VoutCc[index] = val
			print "VoutFwd: "
			print VoutFwd
			print "VoutCC: "
			print VoutCc
			print "sol: "
			print solVoltArray
			print "Check solution "
			Inum = oscNum(solVoltArray,a,g_cc,g_fwd)
			print "I should be close to 0"
			print Inum
			#create hyperrectangle around the solution formed and add it to 
			#allHyperRectangles
			newHyperRectangle = [lowVoltArray,highVoltArray]

		else:
			newHyperRectangle = None
			if(ch == unsat):
				print "no more solutions"
				print "allHyperRectangles = " + str(allHyperRectangles)
			elif(ch == unknown):
				print "solver failed"
			else:
				print "INTERNAL ERROR -- unrecognized return code"

		s.pop()
		if newHyperRectangle is None:
			return allHyperRectangles
		else:
			print "newHyperRectangle"
			print newHyperRectangle
			allHyperRectangles.append(newHyperRectangle)
			# Add the constraint so that Z3 can find solutions outside the hyper rectangle just constructed
			s.add(Or(*[Or(V[i] < newHyperRectangle[0][i] - distances[i], 
							V[i] > newHyperRectangle[1][i] + distances[i]) for i in range(len(V))]))
			#return


def refine(I,V,a,hyperrectangle,g_fwd,g_cc):
	print "Finding solutions within hyperrectangle"
	print "low bounds: ", hyperrectangle[0]
	print "upper bounds: ", hyperrectangle[1]
	s = Solver()
	s.add(And(*[And(V[i]>=hyperrectangle[0][i], V[i]<=hyperrectangle[1][i]) for i in range(len(V))]))
	oldSol = array([(hyperrectangle[0][i]+hyperrectangle[1][i])/2.0 for i in range(len(V))])
	hyperrectangles = [hyperrectangle]
	count = 0
	InumNorms = []
	while True:
		s.push()
		osclRefine(s,I,V,a,hyperrectangles,g_cc,g_fwd)
		is_equilibrium = And(*[I[i]==0 for i in range(len(V))])
		s.add(is_equilibrium)
		ch = s.check()
		if ch==sat:
			VoutFwd = range(0,len(V))
			VoutCc = range(0,len(V))
			solVoltArray = zeros((len(V)))
			m = s.model()
			for d in m.decls():
				dName = str(d.name())
				firstLetter = dName[0]
				if(dName[0]=="V" and dName[1]=="_"):
					index = int(dName[len(dName) - 1])
					val = float(Fraction(str(m[d])))
					#print "index: ", index, " val: ", val
					solVoltArray[index] = val
					'''if(lowVoltArray[index] < 0 and highVoltArray[index]>0):
						if(val >= 0):
							lowVoltArray[index] = 0.0
						elif(val < 0):
							highVoltArray[index] = 0.0'''

				elif(dName[0]=="V" and dName[4]=="F"):
					index = int(dName[len(dName) - 1])
					val = float(Fraction(str(m[d])))
					#print "index: ", index, " val: ", val
					VoutFwd[index] = val

				elif(dName[0]=="V" and dName[4]=="C"):
					index = int(dName[len(dName) - 1])
					val = float(Fraction(str(m[d])))
					#print "index: ", index, " val: ", val
					VoutCc[index] = val
			#print "VoutFwd: "
			#print VoutFwd
			#print "VoutCC: "
			#print VoutCc
			print "sol: "
			print solVoltArray
			print "Check solution "
			Inum = oscNum(solVoltArray,a,g_cc,g_fwd)
			print "I should be close to 0"
			print Inum
			s.pop()

			hyperFirst = hyperrectangles[0]
			hyperLast = hyperrectangles[len(hyperrectangles)-1]
			leftHyperrectangle = [hyperFirst[0],solVoltArray]
			rightHyperrectangle = [solVoltArray,hyperLast[1]]
			hyperrectangles = [leftHyperrectangle,rightHyperrectangle]

			diffBetweenSoln = solVoltArray - oldSol
			InumNorms.append(linalg.norm(diffBetweenSoln))
			if linalg.norm(diffBetweenSoln) < 1e-6:
				plt.plot(arange(len(InumNorms)),InumNorms)
				return solVoltArray

			oldSol = solVoltArray
			count+=1
		
		else:
			s.pop()
			plt.plot(arange(len(InumNorms)),InumNorms)
			return None

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


def testInvRegion(g_cc):
	a = 5
	g_fwd = 1.0
	lenV = 8
	V = RealVector('V',lenV)
	I = RealVector('I',lenV)
	allHyperRectangles = []

	V = RealVector('V',4)
	I = RealVector('I',4)
	allHyperRectangles = []
	VlowVhighs = [[[-3.0,-3.0,-3.0,-3.0],[-2.0,-2.0,-2.0,-2.0]],
				  [[-2.0,-2.0,-2.0,-2.0],[-1.0,-1.0,-1.0,-1.0]],
				  [[-1.0,-1.0,-1.0,-1.0],[0.0, 0.0, 0.0, 0.0]],
				  [[0.0, 0.0, 0.0, 0.0],[1.0, 1.0, 1.0, 1.0]],
				  [[1.0, 1.0, 1.0, 1.0],[2,0, 2.0, 2.0, 2.0]],
				  [[2.0, 2.0, 2.0, 2.0],[3.0, 3.0, 3.0, 3.0]]]

	overallHyperRectangle = findScale(I,V,a,VlowVhighs,g_fwd,g_cc)
	minOptSol = overallHyperRectangle[0]
	maxOptSol = overallHyperRectangle[1]
	#minOptSol = [-0.6429559706071334, -0.6429559706071334, -0.6429559706071334, -0.6429559706071334]
	#maxOptSol = [0.6429559706071334, 0.6429559706071334, 0.6429559706071334, 0.6429559706071334]
	VlowVhighs = [[minOptSol,[minOptSol[0]/2.0,minOptSol[1]/2.0,minOptSol[2]/2.0,minOptSol[3]/2.0]],
				  [[minOptSol[0]/2.0,minOptSol[1]/2.0,minOptSol[2]/2.0,minOptSol[3]/2.0],[0.0, 0.0, 0.0, 0.0]],
				  [[0.0, 0.0, 0.0, 0.0],[maxOptSol[0]/2.0,maxOptSol[1]/2.0,maxOptSol[2]/2.0,maxOptSol[3]/2.0]],
				  [[maxOptSol[0]/2.0,maxOptSol[1]/2.0,maxOptSol[2]/2.0,maxOptSol[3]/2.0],maxOptSol]]


	distances = [(maxOptSol[i] - minOptSol[i])/8.0 for i in range(len(V))]
	allHyperRectangles = findHyper(I,V,a,VlowVhighs,g_fwd,g_cc,distances)
	#plotSolutionsWithinHyperRectangles(I,V,a,g_fwd,g_cc,allHyperRectangles,0)
	
	print "total number of hyperrectangles: ", len(allHyperRectangles)
	print ""
	print "refining hyperrectangles"
	sols = []
	for i in range(len(allHyperRectangles)):
		print "Refining hyper rectangle ", i
		plt.figure(i)
		sol = refine(I,V,a,allHyperRectangles[i],g_fwd,g_cc)
		if sol!=None:
			print "Refined solution: "
			print sol
			sols.append(sol)
		else:
			print "No solution found"
		print ""
	plt.show()
	print "All refined solutions "
	print sols
	
	'''ifFoundAllSolutions(VlowVhighs,a,g_fwd,g_cc,sols)
	print ""

	print "Number of refined solutions ", len(sols)
	print "Checking if refined solutions are correct"
	# Checking the refined hyperrectangle with numerical computation
	for i in range(len(sols)):
		Vtest = sols[i]
		Inum = oscNum(Vtest,a,g_cc,g_fwd)
		print "Example voltage in hyperrectangle"
		print Vtest
		print "Inum should be zero"
		print Inum
		print ""'''


testInvRegion(0.5)
