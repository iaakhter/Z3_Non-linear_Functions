#@author: Mark Greenstreet

import numpy as np
import lpUtils
from cvxopt import matrix,solvers
from scipy.spatial import ConvexHull
from intervalBasics import *


class LP:
	# We'll store the A and Aeq matrices by rows: each row corresponds
	#   to a single constraint.  Each matrix is a python list of rows,
	#   each row is a list of numbers.
	#   Note that cvxopt expects the matrices to stored by colums -- I
	#   guess there's some Fortran in the genealogy of cvxopt.  They also
	#   use scipy matrices.  We'll have to convert when calling the cvxopt
	#   routines.
	def __init__(self, c=None, A=None, b=None, Aeq=None, beq=None):
		if(c is None):
			self.c = None
		else:
			self.c = [ e for e in c ]
		if(A is None):
			self.A = []
		else:
			self.A = [ [ e for e in row ] for row in A]
		if(b is None):
			self.b = []
		else:
			self.b = [ e for e in b ]
		if(Aeq is None):
			self.Aeq = []
		else:
			self.Aeq = [ [ e for e in row ] for row in Aeq]
		if(beq is None):
			self.beq = []
		else:
			self.beq = [ e for e in beq ]

	def num_constraints(self):
		return len(self.A) + len(self.Aeq)

	def __str__(self):
		return "ineqA\n" + str(np.array(self.A)) + "\n" + "ineqB\n" + str(np.array(self.b)) + "\n" + \
				"eqA\n" + str(np.array(self.Aeq)) + "\n" + "eqB\n" + str(np.array(self.beq)) + "\n" + \
				"cost\n" + str(np.array(self.c)) + "\n"

	# concat: add the constraints from LP2 to our constraints.
	# We keep our cost vector and ignore LP2.c.  We should probably
	# check to makes sure that LP2 has the same number of variables as
	# we do, but that will be a later enhancement.
	# concat modifies self.
	def concat(self, LP2):
		self.A = self.A + LP2.A
		self.b = self.b + LP2.b
		self.Aeq = self.Aeq + LP2.Aeq
		self.beq = self.beq + LP2.beq
		return(self)

	# sometimes, we want to replace a variable with its negation (i.e.
	#   when using nfet code to model pfet currents).  That means we need
	#   to negate the elements of A and Aeq.
	def neg_A(self):
		nA = [[-e for e in row] for row in self.A]
		nAeq = [[-e for e in row] for row in self.Aeq]
		return LP(self.c, nA, self.b, nAeq, self.beq)

	# add an equality constraint
	# eq_constraint modifies self.
	def eq_constraint(self, aeq, beq):
		self.Aeq.append(aeq)
		self.beq.append(beq)

	# add an inequality constraint
	# ineq_constraint modifies self.
	def ineq_constraint(self, a, b):
		self.A.append(a)
		self.b.append(b)

	# remove inequality constraint at index, index
	def remove_ineq_constraint(self, index):
		aRemoved = self.A.pop(index)
		bRemoved = self.b.pop(index)
		return (aRemoved, bRemoved)

	
	# Use inequality constraints of 
	# one LP as costs (both min and max) of other LP
	# to construct constraints that satisfy both LP
	# return a new LP from the new constraints
	def constraint_as_cost(self, otherLp):
		#print ("start constraint_as_cost")
		newLp = LP()
		# Use the normals of inequalities of
		# self's as cost (minimize and maximize) 
		# for other LP, solve the other LP to
		# find the constraint (from self's, minimization and maximization)
		# that satisfies both the LPs
		validConstraints = []
		for i in range(len(self.A)):
			#print ("constraint being considered")
			#print (self.A[i])
			#print (self.b[i])
			minCost = [ val*1.0 for val in self.A[i]]
			maxCost = [-val*1.0 for val in self.A[i]]
			possibleValidConstraints = []
			possibleBs = []
			
			# list containint [x, y] where x and y
			# represent a constraint to be added to
			# A and b respectively
			possibleValidConstraints.append([[ val for val in self.A[i]], self.b[i]])
			possibleBs.append(abs(self.b[i]))
			
			# Minimize
			otherLp.add_cost(minCost)
			#print ("otherLp")
			#print (otherLp)
			minSol = otherLp.solve()
			if minSol is not None:
				#print ("minCost", minCost)
				#print ("minSol['x']", np.array(minSol['x']))
				minB = np.dot(np.array(minCost), np.array(minSol['x']))[0]
				#print ("minB", minB)
				possibleValidConstraints.append([maxCost, -minB])
				possibleBs.append(abs(minB))
				#print ("constraint being added", possibleValidConstraints[-1])

			# Maximize
			otherLp.add_cost(maxCost)
			maxSol = otherLp.solve()
			if maxSol is not None:
				#print ("maxCost", maxCost)
				#print ("maxSol['x']", np.array(maxSol['x']))
				maxB = np.dot(np.array(maxCost), np.array(maxSol['x']))[0]
				#print ("maxB", maxB)
				possibleValidConstraints.append([minCost, -maxB])
				possibleBs.append(abs(maxB))
				#print ("constraint being added", possibleValidConstraints[-1])

			# Add the constraint with the highest absolute constant to the newLp
			maxB = max(possibleBs)
			maxBindex = possibleBs.index(maxB)
			newLp.ineq_constraint(possibleValidConstraints[maxBindex][0], possibleValidConstraints[maxBindex][1])

		
		return newLp


	
	# Create a union of self and another LP
	# The union of LPs should satisfy both
	# the LPs
	def union(self, otherLp):
		newLp = LP()

		# Use inequality constraints of self as 
		# costs of otherLp to find new constraints
		# that satisfy self and otherLP
		newLp.concat(self.constraint_as_cost(otherLp))

		# Use inequality constraints of otherLp as 
		# costs of self to find new constraints
		# that satisfy self and otherLP
		newLp.concat(otherLp.constraint_as_cost(self))

		# add the equality constraints
		for i in range(len(self.Aeq)):
			newLp.eq_constraint([x for x in self.Aeq[i]], self.beq[i])
		for i in range(len(otherLp.Aeq)):
			newLp.eq_constraint([x for x in otherLp.Aeq[i]], otherLp.beq[i])

		return newLp

	# This will replace the current cost
	# function. 
	def add_cost(self, c):
		self.c = c

	# return cvxopt solution after solving lp
	def solve(self):
		if self.num_constraints() == 0:
			return None
		cocantenatedA = [ e for e in self.A ]
		for eqConstraint in self.Aeq:
			cocantenatedA.append(eqConstraint)
			negConstraint = [ -x for x in eqConstraint]
			cocantenatedA.append(negConstraint)

		cocantenatedb = [ e for e in self.b ]
		for eqb in self.beq:
			cocantenatedb.append(eqb)
			cocantenatedb.append(-eqb)

		AMatrix = matrix(np.array(cocantenatedA))
		bMatrix = matrix(cocantenatedb)
		cMatrix = matrix(self.c)
		solvers.options["show_progress"] = False

		'''print ("self lp")
		print (self)

		print ("AMatrix")
		print (AMatrix)
		print ("bMatrix")
		print (bMatrix)
		print ("cMatrix")
		print (cMatrix)'''
		
		try:
			sol = solvers.lp(cMatrix, AMatrix, bMatrix)
			return sol
		except ValueError:
			return None

	# Calculate the slack for each inequality constraint
	def slack(self):
		sol = self.solve()
		if sol is None:
			raise Exception('LP:slack - could not solve LP')

		if sol["status"] == "primal infeasible":
			raise Exception('LP:slack - LP infeasible')

		calculatedB = np.dot(np.array(self.A), np.array(sol['x']))
		slack = np.array(self.b) - np.transpose(calculatedB)
		return slack


	def varmap(self, nvar, vmap):
		n_ineq = len(self.A)
		A = []
		for i in range(n_ineq):
			r = [0 for j in range(nvar)]
			for k in range(len(vmap)):
				r[vmap[k]] = self.A[i][k]
			A.append(r)
		n_eq = len(self.Aeq)
		Aeq = []
		for i in range(n_eq):
			r = [0 for j in range(nvar)]
			for k in range(len(vmap)):
				r[vmap[k]] = self.Aeq[i][k]
			Aeq.append(r)
		return LP(self.c, A, self.b, Aeq, self.beq)


class MosfetModel:
	def __init__(self, channelType, Vt, k, gds=0.0):
		self.channelType = channelType   # 'pfet' or 'nfet'
		self.Vt = Vt                     # threshold voltage
		self.k = k                       # carrier mobility
		if(gds == 'default'):
			self.gds = 1.0e-8  # for "leakage" -- help keep the Jacobians non-singular
		else: self.gds = gds

	def __str__(self):
		return "MosfetModel(" + str(self.channelType) + ", " + str(self.Vt) + ", " + str(self.k) + ", " + str(self.s) + ")"

class Mosfet:
	def __init__(self, s, g, d, model, shape=3.0):
		self.s = s
		self.g = g
		self.d = d
		self.shape = shape
		self.model = model

	def ids_help(self, Vs, Vg, Vd, channelType, Vt, ks):
		if(interval_p(Vs) or interval_p(Vg) or interval_p(Vd)):
			# at least one of Vs, Vg, or Vd is an interval, we should return an interval
			return np.array([
				self.ids_help(my_max(Vs), my_min(Vg), my_min(Vd), channelType, Vt, ks),
				self.ids_help(my_min(Vs), my_max(Vg), my_max(Vd), channelType, Vt, ks)])
		elif(channelType == 'pfet'):
			return -self.ids_help(-Vs, -Vg, -Vd, 'nfet', -Vt, -ks)
		elif(Vd < Vs):
			return -self.ids_help(Vd, Vg, Vs, channelType, Vt, ks)
		Vgse = (Vg - Vs) - Vt
		Vds = Vd - Vs
		i_leak = Vds*self.model.gds
		if(Vgse < 0):  # cut-off
			i0 = 0
		elif(Vgse < Vds): # saturation
			i0 = (ks/2.0)*Vgse*Vgse
		else: # linear
			i0 = ks*(Vgse - Vds/2.0)*Vds
		return(i0 + i_leak)

	def ids(self, V):
		model = self.model
		return(self.ids_help(V[self.s], V[self.g], V[self.d], model.channelType, model.Vt, model.k*self.shape))


	# grad_ids: compute the partials of ids wrt. Vs, Vg, and Vd
	#   This function is rather dense.  I would be happier if I could think of
	#    a way to make it more obvious.
	def dg_fun(self, Vs, Vg, Vd, Vt, ks):
		if(Vs[0] > Vd[1]): return None
		Vgse = interval_sub(interval_sub(Vg, np.array([Vs[0], min(Vs[1], Vd[1])])), Vt)
		Vgse[0] = max(Vgse[0], 0)
		Vgse[1] = max(Vgse[1], 0)
		Vds = interval_sub(Vd, Vs)
		Vds[0] = max(Vds[0], 0)
		Vx = np.array([Vg[0] - Vt[1] - Vd[1], Vg[1] - Vt[0] - max(Vs[0], Vd[0])])
		Vx[0] = max(Vx[0], 0)
		Vx[1] = max(Vx[1], 0)
		dg = interval_mult(ks, np.array([min(Vgse[0], Vds[0]), min(Vgse[1], Vds[1])]))
		dd = interval_add(interval_mult(ks, Vx), self.model.gds)
		# print "ks = " + str(ks) + ", gds = " + str(self.model.gds)
		# print "Vgse = " + str(Vgse) + ", Vds = " + str(Vds) + ", Vx = " + str(Vx)
		# print "dg = " + str(dg) + ", dd = " + str(dd)
		return np.array([interval_neg(interval_add(dg, dd)), dg, dd])

	def grad_ids_help(self, Vs, Vg, Vd, channelType, Vt, ks):
		if(channelType == 'pfet'):
			# self.grad_ids_help(-Vs, -Vg, -Vd, 'nfet', -Vt, -ks)
			# returns the partials of -Ids wrt. -Vs, -Vg, and -Vd,
			# e.g. (d -Ids)/(d -Vs).  The negations cancel out; so
			# we can just return that gradient.
			return self.grad_ids_help(interval_neg(Vs), interval_neg(Vg), interval_neg(Vd), 'nfet', -Vt, -ks)
		elif(interval_p(Vs) or interval_p(Vg) or interval_p(Vd)):
			Vs = interval_fix(Vs)
			Vg = interval_fix(Vg)
			Vd = interval_fix(Vd)
			Vt = interval_fix(Vt)
			g0 = self.dg_fun(Vs, Vg, Vd, Vt, ks)
			g1x = self.dg_fun(Vd, Vg, Vs, Vt, ks)
			if(g1x is None): g1 = None
			else: g1 = np.array([interval_neg(g1x[2]), interval_neg(g1x[1]), interval_neg(g1x[0])])
			if g0 is None: return g1
			elif g1 is None: return g0
			else: return np.array([interval_union(g0[i], g1[i]) for i in range(len(g0))])
		elif(Vd < Vs):
			gx = self.grad_ids_help(Vd, Vg, Vs, channelType, Vt, ks)
			return np.array([-gx[2], -gx[1], -gx[0]])
		Vgse = (Vg - Vs) - Vt
		Vds = Vd - Vs
		if(Vgse < 0):  # cut-off: Ids = 0
			return np.array([-self.model.gds, 0.0, self.model.gds])
		elif(Vgse < Vds): # saturation: Ids = (ks/2.0)*Vgse*Vgse
			return np.array([-ks*Vgse - self.model.gds, ks*Vgse, self.model.gds])
		else: # linear: ks*(Vgse - Vds/2.0)*Vds
			dg = ks*Vds
			dd = ks*(Vgse - Vds) + self.model.gds
			return np.array([-(dg + dd), dg, dd])

	def grad_ids(self, V):
		model = self.model
		return(self.grad_ids_help(V[self.s], V[self.g], V[self.d], model.channelType, model.Vt, model.k*self.shape))
			
	def lp_grind(self, Vs, Vg, Vd, Vt, ks, cvx_flag):
		if(not interval_p(Vs)): Vs = [Vs]
		elif(Vs[0] == Vs[1]): Vs = [Vs[0]]
		if(not interval_p(Vg)): Vg = [Vg]
		elif(Vg[0] == Vg[1]): Vg = [Vg[0]]
		if(not interval_p(Vd)): Vd = [Vd]
		elif(Vd[0] == Vd[1]): Vd = [Vd[0]]

		#print ("lp_grind", "Vs", Vs, "Vg", Vg, "Vd", Vd)
		lp = LP()

		# The tangent bounds are pretty easy.
		#   Calculate the current and gradient at each corner of the Vs, Vg, Vd hyperrectangle.
		#   If the ids is convex in this region, the tangent is a lower bound for ids
		#   If the ids is anti-convex in this region, the tangent is an upper bound.
		ids = np.zeros([len(Vs), len(Vg), len(Vd)])
		for i_s in range(len(Vs)):
			vvs = Vs[i_s]
			for i_g in range(len(Vg)):
				vvg = Vg[i_g]
				for i_d in range(len(Vd)):
					vvd = Vd[i_d]
					v = np.array([vvs, vvg, vvd])
					#print ("v", v)
					ids[i_s, i_g, i_d] = self.ids_help(vvs, vvg, vvd, 'nfet', Vt, ks)
					#print ("ids[i_s, i_g, i_d]", ids[i_s, i_g, i_d])
					g = self.grad_ids_help(vvs, vvg, vvd, 'nfet', Vt, ks)
					#print ("g", g)
					d = ids[i_s, i_g, i_d] - np.dot(v,g)
					#print ("d", d)
					if(cvx_flag):
						#print ("ineq: ", [g[0], g[1], g[2], -1], -d)
						lp.ineq_constraint([g[0], g[1], g[2], -1.0], -d)
					else:
						#print ("ineq: ", [-g[0], -g[1], -g[2], 1], d)
						lp.ineq_constraint([-g[0], -g[1], -g[2], 1.0], d)

		#print ("Vs", Vs, "Vg", Vg, "Vd", Vd)
		# To compute the secant constraint, we estimate (d Ids)/(d V) for V in [Vs, Vg, Vd]
		#   using numerical differencing based on the ids values we calculated above.
		#   We then figure out the additive constant to make this an upper bound if
		#   ids is convex, and a lower bound if ids is anti-convex
		dv = np.zeros(3)

		# estimate (d Ids)/(d Vs)
		if(len(Vs) == 1): dv[0] = 0.0
		else:
			sum = 0.0
			n = 0
			for i_g in range(len(Vg)):
				for i_d in range(len(Vd)):
					sum += ids[1, i_g, i_d] - ids[0, i_g, i_d]
					n += 1
			dv[0] = (sum+0.0)/(n*(Vs[1] - Vs[0]))

		# estimate (d Ids)/(d Vg)
		if(len(Vg) == 1): dv[1] = 0.0
		else:
			sum = 0.0
			n = 0
			for i_s in range(len(Vs)):
				for i_d in range(len(Vd)):
					sum += ids[i_s, 1, i_d] - ids[i_s, 0, i_d]
					n += 1
			dv[1] = (sum+0.0)/(n*(Vg[1] - Vg[0]))

		# estimate (d Ids)/(d Vd)
		if(len(Vd) == 1): dv[2] = 0.0
		else:
			sum = 0.0
			n = 0
			for i_s in range(len(Vs)):
				for i_g in range(len(Vg)):
					sum += ids[i_s, i_g, 1] - ids[i_s, i_g, 0]
					n += 1
			dv[2] = (sum+0.0)/(n*(Vd[1] - Vd[0]))

		# determine the additive constant
		b = None
		for i_s in range(len(Vs)):
			for i_g in range(len(Vg)):
				for i_d in range(len(Vd)):
					ix = np.dot(dv, np.array([Vs[i_s], Vg[i_g], Vd[i_d]]))
					bb = ids[i_s, i_g, i_d] - ix
					if(b is None): b = bb
					elif(cvx_flag): b = max(b, bb)
					else: b = min(b, bb)

		if(cvx_flag):
			#print ("ineq", [-dv[0], -dv[1], -dv[2], 1], b)
			lp.ineq_constraint([-dv[0], -dv[1], -dv[2], 1.0], b)
		else:
			#print ("ineq", [dv[0], dv[1], dv[2], -1], -b)
			lp.ineq_constraint([dv[0], dv[1], dv[2], -1.0], -b)

		return lp
	# end lp_grind

	# This function constructs linear program
	# in terms of src, gate, drain and Ids given the model
	# representing the linear or saturation region.
	# Amatrix represents model for quadratic function
	# The quadratic function should be with respect
	# to at most 2 variables - for example, for linear region
	# the function is with respect to 2 variables and for saturation
	# or cutoff it is with respect to 1 variable.
	# For the mosfet case, if it is linear region, the variables
	# are Vg - Vs - Vt and Vd - Vs. For saturation or cutoff, 
	def quad_lin_constraints(self, Amatrix, vertList, Vt, ks):
		#print ("ks", ks)
		if Amatrix.shape[0] > 2:
			raise Exception("quad_lin_constraints: can only accept functions of at most 2 variables")

		# The costs used to find the extreme points
		# in terms of variables for Amatrix should be 
		# derived from the tangents. 
		costs = []
		ds = []
		for vert in vertList:
			#print ("vert", vert)
			grad = 2*np.dot(Amatrix, vert)
			#print ("grad", grad)
			costs.append(list(-grad) + [1])
			dVal = np.dot(np.transpose(vert), np.dot(Amatrix, vert)) - np.dot(grad, vert)
			#print ("currentVal", np.dot(np.transpose(vert), np.dot(Amatrix, vert)))
			#print ("dVal", dVal)
			ds.append(dVal)
		
		lp = LP()

		# handle the case where A is neither 
		# positive semidefinite or negative semidefinite
		if Amatrix.shape[0] > 1 and np.linalg.det(Amatrix) < 0:
			# We need to sandwitch the model by the tangents
			# on both sides so add negation of existing costs
			# to existing costs
			#costs = [[0,0,1]]
			allCosts = [[grad for grad in cost] for cost in costs]
			#allCosts = []
			for cost in costs:
				allCosts.append([-grad for grad in cost])

			for cost in allCosts:
				#print ("cost", cost)
				# Multiply A with coefficient for Ids from cost
				# In all cases, the cost for Ids is 
				cA = cost[-1]*Amatrix

				eigVals, eigVectors = np.linalg.eig(cA)
				#print ("eigVals", eigVals)
				#print ("eigVectors")
				#print (eigVectors)

				# sort the eigVals in descending order
				# order eigVectors in the same way
				sortedIndices = np.fliplr([np.argsort(eigVals)])[0]
				sortedEigVals = eigVals[sortedIndices]
				sortedEigVectors = eigVectors[:,sortedIndices]
				#print ("sortedEigVals", sortedEigVals)
				#print ("sortedEigVectors")
				#print (sortedEigVectors)

				v0 = sortedEigVectors[:,0]
				#print ("v0")
				#print (v0)
				# Find the intersection of eigen vector corresponding to positive
				# eigen value with the hyperrectangle
				intersectionPoints = []
				for vi in range(len(vertList)):
					vert1 = vertList[vi]
					vert2 = vertList[(vi + 1)%len(vertList)]
					#print ("vert1", vert1)
					#print ("vert2", vert2)
					if vert2[0] - vert1[0] == 0:
						x0 = vert2[0]
						x1 = (-v0[0]*x0)/v0[1]
						#print ("x0", x0, "x1", x1)
						if x0 >= vert1[0] and x0 <= vert2[0]:
							intersectionPoints.append(np.array([x0, x1]))
					else:
						m = (vert2[1] - vert1[1])/(vert2[0] - vert1[0])
						c = vert1[1] - m*vert1[0]
						#print ("m", m, "c", c)
						x0 = (-v0[1]*c)/(v0[0] + v0[1]*m)
						x1 = m*x0 + c
						#print ("x0", x0, "x1", x1)
						if x0 >= vert1[0] and x0 <= vert2[0] and x1 >= vert1[1] and x1 <= vert2[1]:
							intersectionPoints.append(np.array([x0, x1]))

				# Now test all the corners and all the intersection points to
				# check which one is the minimum
				pointsToTest = [point for point in vertList]
				pointsToTest += intersectionPoints

				valToCompare = float("inf")
				
				for point in pointsToTest:
					currentAtPoint = np.dot(np.transpose(point), np.dot(Amatrix, point))
					#print ("3dpoint", np.array([point[0], point[1], currentAtPoint]))
					funVal = np.dot(cost, np.array([point[0], point[1], currentAtPoint]))
					#print ("funVal", funVal)
					
					valToCompare = min(funVal, valToCompare)

				#print ("valToCompare", valToCompare)
				# transform expression in terms of vgse and vds, to Vs, Vg and Vd
				dConst =  -(valToCompare + cost[0]*Vt)

				lp.ineq_constraint([cost[0] + cost[1], -cost[0], -cost[1], -cost[2]], dConst)
			#print ("lp in reg")
			#print(lp)
		
		else:
			# handle the case where A is positive semi definite or negative semidefinite
			eigVals, eigVectors = np.linalg.eig(Amatrix)

			# positive semidefinite - convex
			cvx_flag = all([eigVals[ei] >= 0 for ei in range(len(eigVals))])

			# add the tangent constraints
			for ci in range(len(costs)):
				cost = costs[ci]
				if len(cost) == 3:
					gradgsdICons = [-cost[0] - cost[1], cost[0], cost[1], cost[2], ds[ci] + cost[0]*Vt]
				elif len(cost) == 2:
					gradgsdICons = [-cost[0], cost[0], 0.0, cost[1], ds[ci] + cost[0]*Vt]
				#print ("gradgsdICons before", gradgsdICons)
				if(cvx_flag):
					gradgsdICons = [-grad for grad in gradgsdICons]
				
				#print ("gradgsdICons", gradgsdICons)
				lp.ineq_constraint(gradgsdICons[:-1], gradgsdICons[-1])


			# take average of cost this is needed for cap constraint
			avgCost = np.zeros((len(costs[0])))
			for cost in costs:
				avgCost += np.array(cost)
			avgCost = avgCost * (1.0/len(costs))
			#print ("avgCost", avgCost)

			# now find the additive constant for the cap constraint
			d = None
			for vert in vertList:
				#print ("vert", vert)
				IVal = np.dot(np.transpose(vert), np.dot(Amatrix, vert))
				#print ("IVal", IVal)
				#print ("np.dot(-avgCost[:-1], vert)", np.dot(-avgCost[:-1], vert))
				bb = IVal - np.dot(-avgCost[:-1], vert)
				#print ("bb", bb)
				if(d is None): d = bb
				elif(cvx_flag): d = max(d, bb)
				else: d = min(d, bb)

			if len(cost) == 3:
				gradgsdICons = [-avgCost[0] - avgCost[1], avgCost[0], avgCost[1], avgCost[2], d + avgCost[0]*Vt]
			elif len(cost) == 2:
				gradgsdICons = [-avgCost[0], avgCost[0], 0.0, avgCost[1], d + avgCost[0]*Vt]
			#print ("gradgsdICons before", gradgsdICons)
			if not(cvx_flag):
				gradgsdICons = [-grad for grad in gradgsdICons]

			lp.ineq_constraint(gradgsdICons[:-1], gradgsdICons[-1])
			
		#print ("lp in regConstraints")
		#print (lp)
		return lp

	
	def lp_ids_help(self, Vs, Vg, Vd, channelType, Vt, ks):
		#print ("Vs", Vs, "Vg", Vg, "Vd", Vd)
		Vgs = interval_sub(Vg, Vs)
		Vgse = Vgs - Vt
		Vds = interval_sub(Vd, Vs)
		if(not(interval_p(Vs) or interval_p(Vg) or interval_p(Vd))):
			#print ("if1")
			# Vs, Vg, and Vd are non-intervals -- they define a point
			# Add an inequality that our Ids is the value for this point
			return(LP(None, None, None, [[0,0,0,1.0]], self.ids_help(Vs, Vg, Vd, channelType, Vt, ks)))
		elif(channelType == 'pfet'):
			#print ("if2")
			LPnfet = self.lp_ids_help(interval_neg(Vs), interval_neg(Vg), interval_neg(Vd), 'nfet', -Vt, -ks)
			return LPnfet.neg_A()
		elif((interval_lo(Vs) <= interval_hi(Vd)) and (interval_hi(Vs) >= interval_lo(Vd))):
			#print ("if3")
			# If the Vs interval overlaps the Vd interval, then Vds can change sign.
			# That means we've got a saddle.  We won't try to generated LP constraints
			# for the saddle.  So, we just return an empty LP.
			return(LP())
		elif(interval_lo(Vs) > interval_hi(Vd)):
			#print ("if4")
			LPswap = self.lp_ids_help(Vd, Vg, Vs, channelType, Vt, ks)
			A = []
			for i in range(len(LPswap.A)):
				row = LPswap.A[i]
				A.append([row[2], row[1], row[0], -row[3]])
			Aeq = []
			for i in range(len(LPswap.Aeq)):
				row = LPswap.Aeq[i]
				Aeq.append([row[2], row[1], row[0], -row[3]])
			return LP(LPswap.c, A, LPswap.b, Aeq, LPswap.beq)
		
		elif(interval_hi(Vg) - Vt <= interval_lo(Vd)):  # cut-off and/or saturation everywhere in the hyperrectangle
			#print ("if5")
			if(not(interval_p(Vs) or interval_p(Vg))):
				idsVal = self.ids_help(Vs, Vg, Vd, channelType, Vt, ks)
				return(LP(None, [[0,0,0,-1.0], [0,0,0,1.0]], [-idsVal[0], idsVal[1]], None, None))
			else: 
				A = (ks/2.0)*np.array([[1.0]])

				# Change A matrix for cutogg
				if(interval_hi(Vgse) <= 0.0):
					#print ("cutoff")
					A = (ks/2.0)*np.array([[0.0]])

				vertList = [np.array([Vgse[0]]), \
						np.array([Vgse[1]])]
				return self.quad_lin_constraints(A, vertList, Vt, ks)
				#return self.lp_grind(Vs, Vg, interval_hi(Vd), Vt, ks, True)
		elif(interval_lo(Vg) - Vt >= interval_hi(Vd)):  # linear everywhere in the hyperrectangle
			#print ("if6")
			#return self.lp_grind(Vs, Vg, Vd, Vt, ks, False)
			#TODO: Need to incorporate the leakage term in A here somehow
			A = (ks/2.0)*np.array([[0.0, 1.0], [1.0, -1.0]])
			#print ("Vgse", Vgse, "Vds", Vds)
			vertList = [np.array([Vgse[0], Vds[0]]), \
						np.array([Vgse[1], Vds[0]]), \
						np.array([Vgse[1], Vds[1]]), \
						np.array([Vgse[0], Vds[1]])]
			return self.quad_lin_constraints(A, vertList, Vt, ks)
		else:
			#print ("if7")
			return(LP())

	def lp_ids(self, V):
		model = self.model
		idsLp = self.lp_ids_help(V[self.s], V[self.g], V[self.d], model.channelType, model.Vt, model.k*self.shape)
		'''if idsLp.num_constraints() == 0:
			return idsLp'''

		#print ("idsLp")
		#print (idsLp)
		lp = LP()
		indicesList = [self.s, self.g, self.d]
		# add the hyper constraints
		for i in range(len(indicesList)):
			constraint = [0, 0, 0, 0]
			voltage = V[indicesList[i]]
			if interval_p(voltage):
				constraint[i] = -1.0
				#print ("ineq", [ e for e in constraint ], -voltage[0])
				lp.ineq_constraint([ e for e in constraint ], -voltage[0])
				constraint[i] = 1.0
				#print ("ineq", [ e for e in constraint ], voltage[1])
				lp.ineq_constraint([ e for e in constraint ], voltage[1])
			else:
				constraint[i] = 1.0
				#print ("eq", [ e for e in constraint ], voltage)
				lp.eq_constraint([ e for e in constraint ], voltage)

		'''current = self.ids(V)
		constraint = [0, 0, 0, 0]
		#print ("current", current)
		if interval_p(current):
			constraint[3] = -1.0
			#print ("ineq", [ e for e in constraint ], -voltage[0])
			lp.ineq_constraint([ e for e in constraint ], -current[0])
			constraint[3] = 1.0
			#print ("ineq", [ e for e in constraint ], voltage[1])
			lp.ineq_constraint([ e for e in constraint ], current[1])
		else:
			constraint[3] = 1.0
			#print ("eq", [ e for e in constraint ], voltage)
			lp.eq_constraint([ e for e in constraint ], current)'''
		
		lp.concat(idsLp)
		return lp

class Circuit:
	def __init__(self, tr):
		self.tr = tr

	def f(self, V):
		intervalVal = any([interval_p(x) for x in V])
		if intervalVal:
			I_node = np.zeros((len(V),2))
		else:
			I_node = np.zeros(len(V))
		for i in range(len(self.tr)):
			tr = self.tr[i]
			Ids = tr.ids(V)
			#print "Circuit.f: i = " + str(i) + ", tr.s = " + str(tr.s) + "(" + str(V[tr.s]) + "), tr.g = " + str(tr.g) + "(" + str(V[tr.g]) + "), tr.d = " + str(tr.d) + "(" + str(V[tr.d]) + "), ids = " + str(Ids)
			I_node[tr.s] = interval_add(I_node[tr.s], Ids)
			I_node[tr.d] = interval_sub(I_node[tr.d], Ids)
		return I_node

	def jacobian(self, V):
		intervalVal = any([interval_p(x) for x in V])
		
		
		if intervalVal:
			J = np.zeros([len(V), len(V), 2])
		else:
			J = np.zeros([len(V), len(V)])
		for i in range(len(self.tr)):
			tr = self.tr[i]
			g = tr.grad_ids(V)
			# print 'i = ' + str(i) + ', tr.s = ' + str(tr.s) + ', tr.g = ' + str(tr.g) + ', tr.d = ' + str(tr.d) + ', g = ' + str(g)
			sgd = [tr.s, tr.g, tr.d]
			for i in range(len(sgd)):
				J[tr.s, sgd[i]] = interval_add(J[tr.s, sgd[i]], g[i])
				J[tr.d, sgd[i]] = interval_sub(J[tr.d, sgd[i]], g[i])
		return J

	

	# The lp we return has one variable for each node and one variable for each transistor.
	# For 0 <= i < #nodes, variable[i] is the voltage on node i.
	# For 0 <= j < #transistors, variable[#nodes+j] is Ids for transistor j
	# This function collects all the linear constraints related to the 
	# transistors and set the node currents to 0
	def lp(self, V, grndPowerIndex):
		lp = LP()
		n_nodes = len(V)
		n_tr = len(self.tr)
		nvars = len(V) + n_tr

		#print ("nvars", nvars)

		eqCoeffs = np.zeros((n_nodes, nvars))
		for i in range(n_tr):
			#print ("transistor number", i)
			tr = self.tr[i]
			lptr = tr.lp_ids(V)
			lp.concat(lptr.varmap(nvars, [tr.s, tr.g, tr.d, n_nodes+i]))
			eqCoeffs[tr.s, n_nodes + i] += 1.0
			eqCoeffs[tr.d, n_nodes + i] += -1.0

		#print ("eqCoeffs")
		#print (eqCoeffs)

		'''testHyper = np.array([ [ 0.0435,  0.0437 ],
       							[ 0.5698,  0.5700],
       							[ 0.8018,  0.8020],
      			 				[ 0.8328,  0.8330],
       							[ 1.7766,  1.7768],
       							[ 1.0396,  1.0398],
      	 						[ 1.7455,  1.7457],
       							[ 1.0310,  1.0312],
       							[ 0.9144,  0.9146],
       							[ 0.8611,  0.8613],
       							[ 0.0162,  0.0164],
       							[ 0.5562,  0.5564]])

		idsCurrents = np.array([[0.0, 4.76e-06],
								[-0.00020, -0.00018],
								[0.00018741, 0.00018743],
								[-1e-10, 1e-10],
								[-1e-10, 1e-10],
								[-0.000492377684838, -0.000492377684838],
								[0.000639294744512, 0.000639294744512],
								[-0.000146917052105, -0.000146917052105],
								[1.16939696965e-05, 1.16939696965e-05],
								[-0.000186037284398, -0.000186037284398],
								[0.000428879582271, 0.000428879582271]])'''

		# need to add equality constraints that the sum of the currents into each node is zero
		for i in range(n_nodes):
			if all([i != gpi for gpi in grndPowerIndex]):
				#if i < 2:
				lp.ineq_constraint(list(-eqCoeffs[i]), 1e-3)
				lp.ineq_constraint(list(eqCoeffs[i]), 1e-3)
				#lp.eq_constraint(list(eqCoeffs[i]), 0.0)
				'''if i < n_nodes:
					negIneqConstraint = [0]*nvars
					negIneqConstraint[i] = -1.0
					posIneqConstraint = [0]*nvars
					posIneqConstraint[i] = 1.0
					lp.ineq_constraint(negIneqConstraint, -testHyper[i,0])
					lp.ineq_constraint(posIneqConstraint, testHyper[i,1])'''

		'''for ti in range(5):
			negIneqConstraint = [0]*nvars
			negIneqConstraint[n_nodes + ti] = -1.0
			posIneqConstraint = [0]*nvars
			posIneqConstraint[n_nodes + ti] = 1.0
			lp.ineq_constraint(negIneqConstraint, -idsCurrents[ti,0])
			lp.ineq_constraint(posIneqConstraint, idsCurrents[ti,1])'''

		return lp



	# This function solves the linear program 
	# returns a tighter hyperrectangle if feasible
	def linearConstraints(self, V, grndPowerIndex):
		lp = self.lp(V, grndPowerIndex)
		n_nodes = len(V)
		n_tr = len(self.tr)
		nvars = len(V) + n_tr
		#print ("nvars", nvars)
	
		tighterHyper = [x for x in V]
		feasible = True
		for i in range(n_nodes):
			#print ("i", i)
			cost = np.zeros((nvars))

			#minimize variable i
			cost[i] = 1.0
			lp.add_cost(list(cost))
			minSol = lp.solve()

			#maximize variable i
			cost[i] = -1.0
			lp.add_cost(list(cost))
			maxSol = lp.solve()

			if minSol is None or maxSol is None:
				continue

			#print ("minSol status", minSol["status"])
			#print ("maxSol status", maxSol["status"])


			if minSol["status"] == "primal infeasible" and maxSol["status"] == "primal infeasible":
				feasible = False
				break
			if interval_p(tighterHyper[i]):
				if minSol["status"] == "optimal":
					tighterHyper[i][0] = minSol['x'][i] - 1e-6
				if maxSol["status"] == "optimal":
					tighterHyper[i][1] = maxSol['x'][i] + 1e-6


		return [feasible, tighterHyper]




