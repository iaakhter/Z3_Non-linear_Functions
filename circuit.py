#@author: Mark Greenstreet
import pickle
import math
import numpy as np
from cvxopt import matrix,solvers
from scipy.spatial import ConvexHull
from intervalBasics import *
from lpUtilsMark import *


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

	# Calculate ids given Vgse (Vg - Vs - Vt) and Vds (Vd - Vs)
	def ids_help_var2(self, Vgse, Vds, channelType, ks):
		if(channelType == 'pfet'):
			return -self.ids_help_var2(-Vgse, -Vds, 'nfet', -ks)
		elif(Vds < 0.0):
			return -self.ids_help_var2(Vgse - Vds, -Vds, channelType, ks)
		i_leak = Vds*self.model.gds
		if(Vgse < 0):  # cut-off
			i0 = 0
		elif(Vgse < Vds): # saturation
			i0 = (ks/2.0)*Vgse*Vgse
		else: # linear
			i0 = ks*(Vgse - Vds/2.0)*Vds
		return(i0 + i_leak)

	#TODO: Need a better way to frame this - more like ids
	def ids_var2(self, Vgs, Vds):
		model = self.model
		Vgse = Vgs - model.Vt
		return self.ids_help_var2(Vgse, Vds, model.channelType, model.k*self.shape )

	
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
			
	
	# This function constructs linear program
	# in terms of src, gate, drain and Ids given the model
	# representing the cutoff, linear or saturation region.
	# Amatrix represents model for quadratic function
	# The quadratic function should be with respect
	# to at most 2 variables - for example, for linear region
	# the function is with respect to 2 variables and for saturation
	# or cutoff it is with respect to 1 variable.
	# For the mosfet case, if it is linear region, the variables
	# are Vg - Vs - Vt and Vd - Vs. For saturation or cutoff, 
	def quad_lin_constraints(self, Amatrix, vertList, Vt):
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
		
		lp2d = LP()
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
				intersectionPoints = self.intersection([v0[0], v0[1], 0.0], vertList)

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

				lp2d.ineq_constraint([-cost[0], -cost[1], -cost[2]], -valToCompare)
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
					gradgsdICons2d = [cost[0], cost[1], cost[2], ds[ci]]
				elif len(cost) == 2:
					gradgsdICons = [-cost[0], cost[0], 0.0, cost[1], ds[ci] + cost[0]*Vt]
					gradgsdICons2d = [cost[0], 0.0, cost[1], ds[ci]]
				#print ("gradgsdICons before", gradgsdICons)
				if(cvx_flag):
					gradgsdICons = [-grad for grad in gradgsdICons]
					gradgsdICons2d = [-grad for grad in gradgsdICons2d]
				
				#print ("gradgsdICons", gradgsdICons)
				lp.ineq_constraint(gradgsdICons[:-1], gradgsdICons[-1])
				lp2d.ineq_constraint(gradgsdICons2d[:-1], gradgsdICons2d[-1])


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
				gradgsdICons2d = [avgCost[0], avgCost[1], avgCost[2], d]
			elif len(cost) == 2:
				gradgsdICons = [-avgCost[0], avgCost[0], 0.0, avgCost[1], d + avgCost[0]*Vt]
				gradgsdICons2d = [avgCost[0], 0.0, avgCost[1], d]
			#print ("gradgsdICons before", gradgsdICons)
			if not(cvx_flag):
				gradgsdICons = [-grad for grad in gradgsdICons]
				gradgsdICons2d = [-grad for grad in gradgsdICons2d]

			lp.ineq_constraint(gradgsdICons[:-1], gradgsdICons[-1])
			lp2d.ineq_constraint(gradgsdICons2d[:-1], gradgsdICons2d[-1])


		if self.model.gds != 0.0:
			leakTermMat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -self.model.gds, 1.0]])
			newAList = list(np.dot(np.array(lp2d.A), leakTermMat))
			lp2d.A = newAList

		lp3d = LP()
		for i in range(len(lp2d.A)):
			lp3d.ineq_constraint([-lp2d.A[i][0] - lp2d.A[i][1], lp2d.A[i][0], lp2d.A[i][1], lp2d.A[i][2]], 
				lp2d.b[i] + lp2d.A[i][0]*Vt)
			
		'''print ("lp3d")
		print (lp3d)

		print ("lp")
		print (lp)'''


		#print ("lp in regConstraints")
		#print (lp)
		#return lp
		return lp3d

	# Construct linear constraints for the mosfet model
	# depending on whatever region they belong to - cutoff,
	# saturation or linear - Assume nfet at the point when
	# function is called
	def lp_grind(self, Vgse, Vds, Vt, ks, hyperLp):
		#TODO: Need to incorporate the leakage term in A here somehow
		#print ("lp_grind: Vgse", Vgse, "Vds", Vds)
		#print ("lp_grind: hyperLp")
		#print (hyperLp)
		Vgse = interval_fix(Vgse)
		Vds = interval_fix(Vds)

		satA = (ks/2.0)*np.array([[1.0]])
		linA = (ks/2.0)*np.array([[0.0, 1.0], [1.0, -1.0]])

		if interval_hi(Vgse) <= 0.0: # cutoff everywhere in the hyperrectangle
			#print ("lp_grind: if1")
			lp = LP()
			#lp.ineq_constraint([0.0, 0.0, 0.0, -1.0], 0.0)
			#lp.ineq_constraint([0.0, 0.0, 0.0, 1.0], 0.0)
			return lp.concat(hyperLp)

		elif(interval_lo(Vgse) >= 0 and interval_lo(Vds) >= interval_hi(Vgse)):  # saturation everywhere in the hyperrectangle
			#print ("lp_grind: if2")
			vertList = [np.array([Vgse[0]]), \
					np.array([Vgse[1]])]
			return self.quad_lin_constraints(satA, vertList, Vt).concat(hyperLp)

		elif(interval_lo(Vgse) >= 0 and interval_hi(Vds) <= interval_lo(Vgse)):  # linear everywhere in the hyperrectangle
			#print ("lp_grind: if3")
			vertList = [np.array([Vgse[0], Vds[0]]), \
						np.array([Vgse[1], Vds[0]]), \
						np.array([Vgse[1], Vds[1]]), \
						np.array([Vgse[0], Vds[1]])]
			return self.quad_lin_constraints(linA, vertList, Vt).concat(hyperLp)

		else:
			return hyperLp
			#print ("lp_grind: if4: hyperLp")
			#print (hyperLp)
			# take the union of the separate region constraints
			# When taking the union, it is important to bound each variable
			# because of the way the union code works involves solving
			# linear programs and we cannot have degenerate columns in this case
			lp = LP()
			lp.concat(hyperLp)
			newVgse, newVds = Vgse, Vds
			if Vgse[0] < 0.0 and Vgse[1] > 0.0:
				newVgse = np.array([Vgse[0], 0.0])
				regionLp = self.lp_grind(newVgse, newVds, Vt, ks, hyperLp)
				#print ("regionLp")
				#print (regionLp)
				lp = lp.union(regionLp, (regionLp.num_constraints() - hyperLp.num_constraints(), regionLp.num_constraints()))
				#print ("lp after regionLp")
				#print (lp)
				newVgse = np.array([0.0, Vgse[1]])

			# Find intersection between Vgse = Vds line and hyperrectangle
			vertList = [np.array([newVgse[0], newVds[0]]), \
						np.array([newVgse[1], newVds[0]]), \
						np.array([newVgse[1], newVds[1]]), \
						np.array([newVgse[0], newVds[1]])]
			intersectionPoints = self.intersection([-1, 1, 0.0], vertList)
			#print ("intersectionPoints")
			#print (intersectionPoints)
			if len(intersectionPoints) == 0:
				regionLp = self.lp_grind(newVgse, newVds, Vt, ks, hyperLp)
				#print ("regionLp")
				#print (regionLp)
				lp = lp.union(regionLp, (regionLp.num_constraints() - hyperLp.num_constraints(), regionLp.num_constraints()))
				#print ("lp after regionLp")
				#print (lp)
			else:
				#print ("saturation + linear")
				# find the two polygons caused by the line Vgse = Vds line
				# intersecting newVgse
				vertList1, vertList2 = [], []
				leftI, rightI = intersectionPoints[0], intersectionPoints[1]

				if leftI[1] > newVds[0]:
					vertList1 += [leftI, np.array([newVgse[0], newVds[0]])]
					vertList2 += [np.array([leftI[0]])]
				else:
					vertList1 += [leftI]
					vertList2 += [np.array([newVgse[0]]), np.array([leftI[0]])]

				vertList1.append(np.array([newVgse[1], newVds[0]]))

				if rightI[1] < newVds[1]:
					vertList1 += [rightI]
					vertList2 += [np.array([rightI[0]]), np.array([newVgse[1]])]
				else:
					vertList1 += [np.array([newVgse[1], newVds[1]]),rightI]
					vertList2 += [np.array([rightI[0]])]

				vertList2.append(np.array([newVgse[0]]))
				
				# linear region
				regionLp = self.quad_lin_constraints(linA, vertList1, Vt).concat(hyperLp)
				lp = lp.union(regionLp, (regionLp.num_constraints() - hyperLp.num_constraints(), regionLp.num_constraints()))

				# saturation region
				regionLp = self.quad_lin_constraints(satA, vertList2, Vt).concat(hyperLp)
				lp = lp.union(regionLp, (regionLp.num_constraints() - hyperLp.num_constraints(), regionLp.num_constraints()))

			return lp

	def lp_grind_pre(self, Vgs, Vds, numDivisions, mainDict):
		minVal, maxVal = -1.8, 1.8
		unitDiff = (maxVal - minVal)/numDivisions

		dictVgsEntry = np.array([math.floor((Vgs[0] - minVal)/unitDiff), math.ceil((Vgs[1] - minVal)/unitDiff)])
		dictVdsEntry = np.array([math.floor((Vds[0] - minVal)/unitDiff), math.ceil((Vds[1] - minVal)/unitDiff)])
		if dictVgsEntry[0] == dictVgsEntry[1]:
			if dictVgsEntry[0] != 0:
				dictVgsEntry[0] -= 1
			elif dictVgsEntry[1] != numDivisions:
				dictVgsEntry[1] += 1
		if dictVdsEntry[0] == dictVdsEntry[1]:
			if dictVdsEntry[0] != 0:
				dictVdsEntry[0] -= 1
			elif dictVdsEntry[1] != numDivisions:
				dictVdsEntry[1] += 1
		'''print ("Vgs", Vgs, "Vds", Vds)
		print ("dictVgsEntry", dictVgsEntry)
		print ("dictVdsEntry", dictVdsEntry)'''
		lp = mainDict[(int(dictVgsEntry[0]), int(dictVdsEntry[0]))][(int(dictVgsEntry[1]), int(dictVdsEntry[1]))]
		newLp = LP()
		for i in range(len(lp.A)):
			newLp.ineq_constraint([-lp.A[i][0]-lp.A[i][1], lp.A[i][0], lp.A[i][1], lp.A[i][2]], lp.b[i])
		return newLp


	
	def lp_ids_help_pre(self, Vs, Vg, Vd, channelType, Vt, ks, numDivisions, mainDict):
		Vgs = interval_sub(Vg, Vs)
		Vgse = Vgs - Vt
		Vds = interval_sub(Vd, Vs)
		#print ("Vgse", Vgse, "Vds", Vds)
		if(not(interval_p(Vs) or interval_p(Vg) or interval_p(Vd))):
			#print ("if1")
			# Vs, Vg, and Vd are non-intervals -- they define a point
			# Add an inequality that our Ids is the value for this point
			return(LP(None, None, None, [[0,0,0,1.0]], self.ids_help(Vs, Vg, Vd, channelType, Vt, ks)))
		elif(channelType == 'pfet'):
			#print ("if2")
			LPnfet = self.lp_ids_help_pre(interval_neg(Vs), interval_neg(Vg), interval_neg(Vd), 'nfet', -Vt, -ks, numDivisions, mainDict)
			return LPnfet.neg_A()
		elif((interval_lo(Vs) <= interval_hi(Vd)) and (interval_hi(Vs) >= interval_lo(Vd))):
			#print ("if3")
			# If the Vs interval overlaps the Vd interval, then Vds can change sign.
			# That means we've got a saddle.  We won't try to generated LP constraints
			# for the saddle.  So, we just return an empty LP.
			return(LP())
		elif(interval_lo(Vs) > interval_hi(Vd)):
			#print ("if4")
			LPswap = self.lp_ids_help_pre(Vd, Vg, Vs, channelType, Vt, ks, numDivisions, mainDict)
			A = []
			for i in range(len(LPswap.A)):
				row = LPswap.A[i]
				A.append([row[2], row[1], row[0], -row[3]])
			Aeq = []
			for i in range(len(LPswap.Aeq)):
				row = LPswap.Aeq[i]
				Aeq.append([row[2], row[1], row[0], -row[3]])
			return LP(LPswap.c, A, LPswap.b, Aeq, LPswap.beq)

		else:
			#print ("if5")
			#return LP()
			Ids = self.ids_help(Vs, Vg, Vd, channelType, Vt, ks)
			# form the LP where the constraints for the Vgse and Vds
			# bounds are added. This is important to make sure we don't get
			# unboundedness from our linear program
			hyperLp = LP()
			if interval_p(Vs):
				hyperLp.ineq_constraint([-1.0, 0.0, 0.0, 0.0], -Vs[0])
				hyperLp.ineq_constraint([1.0, 0.0, 0.0, 0.0], Vs[1])
			else:
				hyperLp.ineq_constraint([-1.0, 0.0, 0.0, 0.0], -(Vs - 1e-5))
				hyperLp.ineq_constraint([1.0, 0.0, 0.0, 0.0], Vs + 1e-5)
			if interval_p(Vg):
				hyperLp.ineq_constraint([0.0, -1.0, 0.0, 0.0], -Vg[0])
				hyperLp.ineq_constraint([0.0, 1.0, 0.0, 0.0], Vg[1])
			else:
				hyperLp.ineq_constraint([0.0, -1.0, 0.0, 0.0], -(Vg - 1e-5))
				hyperLp.ineq_constraint([0.0, 1.0, 0.0, 0.0], Vg + 1e-5)
			if interval_p(Vd):
				hyperLp.ineq_constraint([0.0, 0.0, -1.0, 0.0], -Vd[0])
				hyperLp.ineq_constraint([0.0, 0.0, 1.0, 0.0], Vd[1])
			else:
				hyperLp.ineq_constraint([0.0, 0.0, -1.0, 0.0], -(Vd - 1e-5))
				hyperLp.ineq_constraint([0.0, 0.0, 1.0, 0.0], Vd + 1e-5)
			if interval_p(Ids):
				hyperLp.ineq_constraint([0.0, 0.0, 0.0, -1.0], -(Ids[0] - 1e-5))
				hyperLp.ineq_constraint([0.0, 0.0, 0.0, 1.0], (Ids[1] + 1e-5))
			else:
				hyperLp.ineq_constraint([0.0, 0.0, 0.0, -1.0], -(Ids - 1e-5))
				hyperLp.ineq_constraint([0.0, 0.0, 0.0, 1.0], (Ids + 1e-5))
			return self.lp_grind_pre(Vgs, Vds, numDivisions, mainDict).concat(hyperLp)


	def lp_ids_help(self, Vs, Vg, Vd, channelType, Vt, ks):
		#print ("Vs", Vs, "Vg", Vg, "Vd", Vd, "channelType", channelType)
		#print ("interval_r(Vs)", interval_r(Vs), "interval_r(Vd)", interval_r(Vd))
		Vgs = interval_sub(Vg, Vs)
		Vgse = Vgs - Vt
		Vds = interval_sub(Vd, Vs)

		Ids = self.ids_help(Vs, Vg, Vd, channelType, Vt, ks)
		# form the LP where the constraints for the Vgse and Vds
		# bounds are added. This is important to make sure we don't get
		# unboundedness from our linear program
		hyperLp = LP()
		if interval_p(Vs):
			hyperLp.ineq_constraint([-1.0, 0.0, 0.0, 0.0], -Vs[0])
			hyperLp.ineq_constraint([1.0, 0.0, 0.0, 0.0], Vs[1])
		else:
			hyperLp.ineq_constraint([-1.0, 0.0, 0.0, 0.0], -(Vs))
			hyperLp.ineq_constraint([1.0, 0.0, 0.0, 0.0], Vs)
		if interval_p(Vg):
			hyperLp.ineq_constraint([0.0, -1.0, 0.0, 0.0], -Vg[0])
			hyperLp.ineq_constraint([0.0, 1.0, 0.0, 0.0], Vg[1])
		else:
			hyperLp.ineq_constraint([0.0, -1.0, 0.0, 0.0], -(Vg))
			hyperLp.ineq_constraint([0.0, 1.0, 0.0, 0.0], Vg)
		if interval_p(Vd):
			hyperLp.ineq_constraint([0.0, 0.0, -1.0, 0.0], -Vd[0])
			hyperLp.ineq_constraint([0.0, 0.0, 1.0, 0.0], Vd[1])
		else:
			hyperLp.ineq_constraint([0.0, 0.0, -1.0, 0.0], -(Vd))
			hyperLp.ineq_constraint([0.0, 0.0, 1.0, 0.0], Vd)
		if interval_p(Ids):
			hyperLp.ineq_constraint([0.0, 0.0, 0.0, -1.0], -(Ids[0]))
			hyperLp.ineq_constraint([0.0, 0.0, 0.0, 1.0], (Ids[1]))
		else:
			hyperLp.ineq_constraint([0.0, 0.0, 0.0, -1.0], -(Ids))
			hyperLp.ineq_constraint([0.0, 0.0, 0.0, 1.0], (Ids))

		if interval_r(Vs) < 1e-14 and interval_r(Vd) < 1e-14:
			return hyperLp
		
		#print ("Vgse", Vgse, "Vds", Vds)
		if(not(interval_p(Vs) or interval_p(Vg) or interval_p(Vd))):
			#print ("if1")
			# Vs, Vg, and Vd are non-intervals -- they define a point
			# Add an inequality that our Ids is the value for this point
			return(LP(None, None, None, [[0,0,0,1.0]], self.ids_help(Vs, Vg, Vd, channelType, Vt, ks)))
		elif(channelType == 'pfet'):
			#print ("if2")
			LPnfet = self.lp_ids_help(interval_neg(Vs), interval_neg(Vg), interval_neg(Vd), 'nfet', -Vt, -ks)
			return LPnfet.neg_A()
		elif((interval_lo(Vs) < interval_hi(Vd)) and (interval_hi(Vs) > interval_lo(Vd))):
			#print ("if3")
			# If the Vs interval overlaps the Vd interval, then Vds can change sign.
			# That means we've got a saddle.  We won't try to generated LP constraints
			# for the saddle.  So, we just return an empty LP.
			#return hyperLp
			#print ("Vs", Vs, "Vd", Vd)
			'''intersect = interval_intersect(Vs, Vd)
			intersectMidPoint = interval_mid(intersect)
			Vs1stHalf = np.array(Vs[0], intersectMidPoint)
			Vs2ndHalf = np.array(intersectMidPoint, Vs[1])
			Vd1stHalf = np.array(Vd[0], intersectMidPoint)
			Vd2ndHalf = np.array([intersectMidPoint, Vd[1]])
			lp1 = self.lp_ids_help(Vs1stHalf, Vg, Vd2ndHalf, channelType, Vt, ks)
			lp2 = self.lp_ids_help(Vs2ndHalf, Vg, Vd1stHalf, channelType, Vt, ks)
			return lp1.union(lp2, (lp2.num_constraints() - hyperLp.num_constraints(), lp2.num_constraints()))'''

			return hyperLp
		elif(interval_lo(Vs) >= interval_hi(Vd)):
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

		else:
			#print ("if5")
			return self.lp_grind(Vgse, Vds, Vt, ks, hyperLp)

	# Find the intersection points between the line
	# represented by [a, b, c] where a, b, and c represent
	# the line ax + by = c and polygon represented by vertList
	# a list of arrays where each array represents the vertex
	def intersection(self, line, vertList):
		intersectionPoints = []
		for vi in range(len(vertList)):
			vert1 = vertList[vi]
			vert2 = vertList[(vi + 1)%len(vertList)]
			minVert = np.minimum(vert1, vert2)
			maxVert = np.maximum(vert1, vert2)
			if vert2[0] - vert1[0] == 0:
				x0 = vert2[0]
				x1 = (line[2] - line[0]*x0)/line[1]
				if x1 >= minVert[1] and x1 <= maxVert[1]:
					intersectionPoints.append(np.array([x0, x1]))
			else:
				m = (vert2[1] - vert1[1])/(vert2[0] - vert1[0])
				c = vert1[1] - m*vert1[0]
				#print ("m", m, "c", c)
				x0 = (line[2] - line[1]*c)/(line[0] + line[1]*m)
				x1 = m*x0 + c
				if x0 >= minVert[0] and x0 <= maxVert[0] and x1 >= minVert[1] and x1 <= maxVert[1]:
					intersectionPoints.append(np.array([x0, x1]))

		return intersectionPoints

	def lp_ids(self, V, numDivisions = None, mainDict = None):
		model = self.model
		idsLp = self.lp_ids_help(V[self.s], V[self.g], V[self.d], model.channelType, model.Vt, model.k*self.shape)
		#idsLp = self.lp_ids_help_pre(V[self.s], V[self.g], V[self.d], model.channelType, model.Vt, model.k*self.shape, numDivisions, mainDict)
	
		#print ("idsLp")
		#print (idsLp)
		return idsLp


class Circuit:
	def __init__(self, tr):
		self.tr = tr
		'''filename = "precompConvexHull18.pkl"
		theFile = open(filename, "rb")
		self.numDivisions, self.mainDict = pickle.load(theFile)
		theFile.close()
		#print ("loaded")'''

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
		#print ("n_tr", n_tr)

		eqCoeffs = np.zeros((n_nodes, nvars))

		# keep an array of nvars with boolean values
		# if they haven't been dealt with yet, add the hyper constraint
		# for that specific variable
		hyperAdded = [False]*nvars
		for i in range(n_tr):
			#print ("transistor number", i)
			tr = self.tr[i]
			#lptr = tr.lp_ids(V, self.numDivisions, self.mainDict)
			lptr = tr.lp_ids(V)

			# remove the hyper constraints that we might have added 
			# (hyper constraints might have been added in order to make the
			# union code work)
			modLptr = LP()
			for ai in range(6,len(lptr.A)):
				modLptr.ineq_constraint(lptr.A[ai], lptr.b[ai])
			#print ("lptr")
			#print (lptr)
			#print ("modLptr")
			#print (modLptr)
			
			lp.concat(modLptr.varmap(nvars, [tr.s, tr.g, tr.d, n_nodes+i]))
			
			# Add the hyper constraints for tr.s,tr.g and tr.d if they have not been
			# added yet
			for ind in [tr.s, tr.g, tr.d]:
				if not(hyperAdded[ind]):
					#print ("ind", ind)
					hyperAdded[ind] = True
					greatConstr = np.zeros((nvars))
					greatConstr[ind] = -1.0
					lessConstr = np.zeros((nvars))
					lessConstr[ind] = 1.0
					if interval_p(V[ind]):
						lp.ineq_constraint(greatConstr, -V[ind][0])
						lp.ineq_constraint(lessConstr, V[ind][1])
					else:
						lp.ineq_constraint(greatConstr, -V[ind])
						lp.ineq_constraint(lessConstr, V[ind])

			eqCoeffs[tr.s, n_nodes + i] += 1.0
			eqCoeffs[tr.d, n_nodes + i] += -1.0


		#print ("eqCoeffs")
		#print (eqCoeffs)


		# need to add equality constraints that the sum of the currents into each node is zero
		for i in range(n_nodes):
			if all([i != gpi for gpi in grndPowerIndex]):
				#if i < 2:
				lp.ineq_constraint(list(-eqCoeffs[i]), 0.0)
				lp.ineq_constraint(list(eqCoeffs[i]), 0.0)
				#lp.eq_constraint(list(eqCoeffs[i]), 0.0)

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
		numSuccessLp, numUnsuccessLp, numTotalLp = 0, 0, 0
		for i in range(n_nodes):
			if interval_p(tighterHyper[i]):
				#print ("nvars", nvars)
				#print ("numConstraints", lp.num_constraints())
				numTotalLp += 2
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
					print ("nvars", nvars)
					print ("numConstraints", lp.num_constraints())
					print ("hyperrectangle", V)
					print ("minSol is None?", minSol, "maxSol is None?", maxSol)
					print (lp)
					numUnsuccessLp += 2
					continue

				#print ("minSol status", minSol["status"])
				#print ("maxSol status", maxSol["status"])


				if minSol["status"] == "primal infeasible" and maxSol["status"] == "primal infeasible":
					numSuccessLp += 2
					feasible = False
					break

				if minSol["status"] == "optimal":
					#print ("min lp optimal", minSol["status"])
					#print (lp)
					tighterHyper[i][0] = minSol['x'][i]
					numSuccessLp += 1
				else:
					numUnsuccessLp += 1
					#print ("min lp not optimal", minSol["status"])
					#print (lp)
				if maxSol["status"] == "optimal":
					tighterHyper[i][1] = maxSol['x'][i]
					numSuccessLp += 1
				else:
					numUnsuccessLp += 1


		#print ("numTotalLp", numTotalLp, "numSuccessLp", numSuccessLp, "numUnsuccessLp", numUnsuccessLp)
		return [feasible, tighterHyper, numTotalLp, numSuccessLp, numUnsuccessLp]


class TransistorCircuit(Circuit):
	def __init__(self, tr):
		Circuit.__init__(self, tr)
		filename = "precompConvexHullInv9_20.pkl"
		theFile = open(filename, "rb")
		self.numDivisions, self.mainDict = pickle.load(theFile)
		theFile.close()
		#print ("loaded")

	def lp(self, V, grndPowerIndex):


		lp = LP()
		n_nodes = len(V)
		n_tr = len(self.tr)
		nvars = len(V) + n_tr

		#print ("nvars", nvars)
		#print ("n_tr", n_tr)

		eqCoeffs = np.zeros((n_nodes, nvars))
		for i in range(n_tr):
			#print ("transistor number", i)
			tr = self.tr[i]
			#print ("tr.s", tr.s, "V[tr.s]", V[tr.s])
			VinInterval = V[tr.g]
			VoutInterval = V[tr.d]

			minVal, maxVal = 0.0, 1.8
			unitDiff = (maxVal - minVal)/self.numDivisions

			dictVinEntry = np.array([math.floor((VinInterval[0] - minVal)/unitDiff), math.ceil((VinInterval[1] - minVal)/unitDiff)])
			dictVoutEntry = np.array([math.floor((VoutInterval[0] - minVal)/unitDiff), math.ceil((VoutInterval[1] - minVal)/unitDiff)])
			if dictVinEntry[0] == dictVinEntry[1]:
				if dictVinEntry[0] != 0:
					dictVinEntry[0] -= 1
				elif dictVinEntry[1] != self.numDivisions:
					dictVinEntry[1] += 1
			if dictVoutEntry[0] == dictVoutEntry[1]:
				if dictVoutEntry[0] != 0:
					dictVoutEntry[0] -= 1
				elif dictVoutEntry[1] != numDivisions:
					dictVoutEntry[1] += 1
			'''print ("VinInterval", VinInterval, "VoutInterval", VoutInterval)
			print ("dictVinEntry", dictVinEntry)
			print ("dictVoutEntry", dictVoutEntry)'''
			#localLp = self.mainDict[(int(dictVinEntry[0]), int(dictVoutEntry[0]))][(int(dictVinEntry[1]), int(dictVoutEntry[1]))]
			localLp = LP()
			localLp.ineq_constraint([-1.0, 0.0, 0.0], -VinInterval[0])
			localLp.ineq_constraint([1.0, 0.0, 0.0], VinInterval[1])
			localLp.ineq_constraint([0.0, -1.0, 0.0], -VoutInterval[0])
			localLp.ineq_constraint([0.0, 1.0, 0.0], VoutInterval[1])
			localLp.ineq_constraint([0.0, 0.0, -1.0], 100.0)
			localLp.ineq_constraint([0.0, 0.0, 1.0], 100.0)
			#print ("localLp")
			#print (localLp)
			lp.concat(localLp.varmap(nvars, [tr.g, tr.d, n_nodes+i]))
			newIneqConstrant1 = np.zeros((nvars))
			newIneqConstrant1[tr.s] = -1.0
			newIneqConstrant2 = np.zeros((nvars))
			newIneqConstrant2[tr.s] = 1.0
			if interval_p(V[tr.s]):
				lp.ineq_constraint(list(newIneqConstrant1), -V[tr.s][0])
				lp.ineq_constraint(list(newIneqConstrant2), V[tr.s][1])
			else:
				lp.ineq_constraint(list(newIneqConstrant1), -(V[tr.s] - 1e-3))
				lp.ineq_constraint(list(newIneqConstrant2), (V[tr.s] + 1e-3))
			eqCoeffs[tr.s, n_nodes + i] += 1.0
			eqCoeffs[tr.d, n_nodes + i] += -1.0

		#print ("eqCoeffs")
		#print (eqCoeffs)

		#print ("lp")
		#print (lp)


		# need to add equality constraints that the sum of the currents into each node is zero
		for i in range(n_nodes):
			if all([i != gpi for gpi in grndPowerIndex]):
				#if i < 2:
				lp.ineq_constraint(list(-eqCoeffs[i]), 1e-3)
				lp.ineq_constraint(list(eqCoeffs[i]), 1e-3)
				#lp.eq_constraint(list(eqCoeffs[i]), 0.0)

		return lp




