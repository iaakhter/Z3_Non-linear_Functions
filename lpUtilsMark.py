#@author: Mark Greenstreet

import numpy as np
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
		'''print ("start constraint_as_cost")
		print ("selfLp")
		print (self)
		print ("otherLp")
		print (otherLp)'''
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
			minSol = otherLp.solve()
			if minSol is not None and minSol["status"] == "optimal":
				#print ("minSol['status']", minSol["status"])
				minB = np.dot(np.array(minCost), np.array(minSol['x']))[0]
				#print ("minB", minB)
				possibleValidConstraints.append([maxCost, -minB])
				possibleBs.append(abs(minB))
				#print ("constraint being added", possibleValidConstraints[-1])

			# Maximize
			otherLp.add_cost(maxCost)
			maxSol = otherLp.solve()
			if maxSol is not None and maxSol["status"] == "optimal":
				#print ("maxSol['status']", maxSol["status"])
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
			if minSol is not None and maxSol is not None and minSol["status"] == "optimal" and maxSol["status"] == "optimal":
				newLp.ineq_constraint(possibleValidConstraints[maxBindex][0], possibleValidConstraints[maxBindex][1])

		
		#print ("newLp")
		#print (newLp)
		return newLp


	
	# Create a union of self and another LP
	# The union of LPs should satisfy both
	# the LPs
	def union(self, otherLp):
		'''print ("selfLp before", self.num_constraints())
		print (self)
		print ("otherLp before", otherLp.num_constraints())
		print (otherLp)'''
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

