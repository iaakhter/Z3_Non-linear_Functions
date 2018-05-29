import numpy as np
import lpUtils
from cvxopt import matrix,solvers
from scipy.spatial import ConvexHull


def my_reduce_last_dim_help(op, src, dst):
	if(src.ndim == 2): 
		for i in range(len(src)):
			dst[i] = reduce(op, src[i])
	else:
		for i in range(len(src)):
			my_reduce_last_dim_help(op, src[i], dst[i])

def my_reduce_last_dim(op, x):
	if(not hasattr(x, 'ndim')):
		return x
	if(x.ndim == 1):
		return(np.array(reduce(op, x)))
	dims = [];
	xx = x;
	for i in range(x.ndim - 1):
		dims.append(len(xx))
		xx = xx[0]
	result = np.zeros(dims)
	my_reduce_last_dim_help(op, x, result)
	return result

def my_min(x):
	return my_reduce_last_dim(lambda x, y: min(x,y), x)

def my_max(x):
	return my_reduce_last_dim(lambda x, y: max(x,y), x)

def interval_p(x):
	return hasattr(x, 'ndim') and (x.ndim == 1) and (len(x) == 2)

def tiny_p(x):
	if(interval_p(x)):
	  return(tiny_p(x[0]) and tiny_p(x[1]))
	else:
	  return(abs(x) < 1.0e-14)

def interval_fix(x):
	return (x if interval_p(x) else np.array([x, x]))

def interval_lo(x):
	if(interval_p(x)): return x[0]
	else: return x

def interval_hi(x):
	if(interval_p(x)): return x[1]
	else: return x

def interval_add(x, y):
	if(interval_p(x) and interval_p(y)):
		return(np.array([x[0]+y[0], x[1]+y[1]]))
	elif(interval_p(x)):
		return(np.array([x[0]+y, x[1]+y]))
	elif(interval_p(y)):
		return(np.array([x+y[0], x+y[1]]))
	else: return(x+y)

def interval_neg(x):
	if(interval_p(x)):
		return(np.array([-x[1], -x[0]]))
	else: return(-x)

def interval_sub(x, y):
	return(interval_add(x, interval_neg(y)))

def interval_mult(x, y):
	if(interval_p(x) and interval_p(y)):
		p = [xx*yy for xx in x for yy in y]
		return np.array([min(p), max(p)])
	elif(interval_p(x)):
		if(y >= 0): return np.array([y*x[0], y*x[1]])
		else: return(np.array[y*x[1], y*x[0]])
	elif(interval_p(y)):
		return interval_mult(y,x)
	else: return(x*y)

def interval_div(x, y):
	if((interval_p(y) and y[0]*y[1] <= 0) or tiny_p(y)):
		return np.array([float('-inf'), float('+inf')])
	elif(interval_p(y)):
		q = [xx/yy for xx in interval_fix(x) for yy in y]
		return np.array([min(q), max(q)])
	elif(interval_p(x)):
		if(y >= 0): return np.array([x[0]/y, x[1]/y])
		else: return(np.array[x[1]/y, x[0]/y])
	else: return((x+0.0)/y)
		
def interval_union(x, y):
	if(x is None): return y
	elif(y is None): return x
	else: return np.array([min(x[0], y[0]), max(x[1], y[1])])


class LP:
	def __init__(self, nvar, c=None, A=None, b=None, Aeq=None, beq=None):
		self.nvar = nvar
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

	def concat(self, LP2):
		if(L2.nvar != self.nvar):
			raise Exception('Incompatible LPs: self.nvar = ' + str(self.nvar) + ', LP2.nvar = ' + str(LP2.nvar))
		else:
			self.A = [ self.A[i] + LP2.A[i] for i in range(nvar) ]
			self.b = self.b + LP2.b

	def neg_A(self):
		nA = [[-e for e in row] for row in self.A]
		nAeq = [[-e for e in row] for row in self.Aeq]
		return LP(self.nvar, self.c, nA, self.b, nAeq, self.beq)

	def eq_constraint(self, aeq, beq):
		self.Aeq.append(aeq)
		self.beq.append(beq)

	def ineq_constraint(self, a, b):
		self.A.append(a)
		self.b.append(b)

	def varmap(self, nvar, varmap):
		n_ineq = len(self.A)
		A = []
		for i in range(n_ineq):
			r = [0 for j in range(nvar)]
			for k in range(len(self.nvar)):
				r[varmap[k]] = self.A[i,k]
			A.append(row)
		n_eq = len(self.Aeq)
		Aeq = []
		for i in range(n_eq):
			r = [0 for j in range(nvar)]
			for k in range(len(self.nvar)):
				r[varmap[k]] = self.Aeq[i,k]
			Aeq.append(row)
		return LP(nvar, self.c, A, self.b, Aeq, self.beq)


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
		dg = interval_mult(ks, np.array([min(Vgse[0], Vds[0]), Vgse[1]]))
		dd = interval_add(interval_mult(ks, Vx), self.model.gds)
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
		lp = LP(4)

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
					ids[i_s, i_g, i_d] = self.ids_help(vvs, vvg, vvd, 'nfet', Vt, ks)
					g = self.grad_ids_help(vvs, vvg, vvd, 'nfet', Vt, ks)
					d = np.dot(v,g) - ids[i_s, i_g, i_d]
					if(cvx_flag):
						lp.ineq_constraint([g[0], g[1], g[2], -1], d)
					else:
						lp.ineq_constraint([-g[0], -g[1], -g[2], 1], -d)

		# To compute the secant contraint, we estimate (d Ids)/(d V) for V in [Vs, Vg, Vd]
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
			dv[2] = (sum+0.0)/(n*(Vg[1] - Vg[0]))

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
			lp.ineq_constraint([dv[0], dv[1], dv[2], -1], b)
		else:
			lp.ineq_constraint([-dv[0], -dv[1], -dv[2], 1], -b)

		return lp
	# end lp_grind


	def lp_ids_help(self, Vs, Vg, Vd, channelType, Vt, ks):
		if(not(interval_p(Vs) or interval_p(Vg) or interval_p(Vd))):
			# Vs, Vg, and Vd are non-intervals -- they define a point
			# Add an inequality that our Ids is the value for this point
			return(LP(4, None, None, None, [0,0,0,1], self.ids_help(Vs, Vg, Vd, channelType, Vt, ks)))
		elif(channelType == 'pfet'):
			LPnfet = self.lp_ids_help(self, interval_neg(Vs), interval_neg(Vg), interval_neg(Vd), 'nfet', interval_neg(Vt), interval_neg(k), s)
			return LPnfet.neg_A()
		elif((interval_lo(Vs) <= interval_hi(Vd)) and (interval_hi(Vs) >= interval_lo(Vd))):
			return(LP(4))
		elif(interval_lo(Vs) > interval_hi(Vd)):
			LPswap = self.lp_ids_help(Vd, Vg, Vs, model.channelType, model.Vt, model.k*self.shape)
			A = []
			for i in range(len(LPswap.A)):
				row = LPswap.A[i]
				A.append([row[2], row[1], row[0], -row[3]])
			Aeq = []
			for i in range(len(LPswap.Aeq)):
				row = LPswap.Aeq[i]
				Aeq.append([row[2], row[1], row[0], -row[3]])
			return LP(LPswap.nvar, LPswap.c, A, LPswap.b, Aeq, LPswap.beq)
		elif(interval_hi(Vg) - Vt <= interval_lo(Vd)):  # cut-off and/or saturation
			if(not(interval_p(Vs) or interval_p(Vg))):
				return(LP(4, None, None, None, [0,0,0,1], self.ids_help(Vs, Vg, Vd, channelType, Vt, ks)))
			else: return self.lp_grind(Vs, Vg, interval_hi(Vd), Vt, ks, True)
		elif(interval_lo(Vg) - Vt >= interval_hi(Vd)):  # cut-off and/or saturation
			return self.lp_grind(Vs, Vg, Vd, Vt, ks, False)
		else: return(LP(4))

	def lp_ids(self, V):
		model = self.model
		return(self.lp_ids_help(V[self.s], V[self.g], V[self.d], model.channelType, model.Vt, model.k*self.shape))

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
			# print "Circuit.f: i = " + str(i) + ", tr.s = " + str(tr.s) + "(" + str(V[tr.s]) + "), tr.g = " + str(tr.g) + "(" + str(V[tr.g]) + "), tr.d = " + str(tr.d) + "(" + str(V[tr.d]) + "), ids = " + str(Ids)
			I_node[tr.s] = interval_add(I_node[tr.s], Ids)
			I_node[tr.d] = interval_sub(I_node[tr.d], Ids)
		return I_node

	# Because the Rambus oscillator was our first example, other parts of
	# other parts of the code expect an 'oscNum' function.  I think this
	# if what I'm supposed to provide.
	def oscNum(self, V):
		return [None, None, self.f(V)]

	def jacobian(self, V):
		intervalVal = any([interval_p(x) for x in V])
		
		
		if intervalVal:
			J = np.zeros([len(V), len(V), 2])
		else:
			J = np.zeros([len(V), len(V)])
		for i in range(len(self.tr)):
			tr = self.tr[i]
			g = tr.grad_ids(V)
			sgd = [tr.s, tr.g, tr.d]
			for i in range(len(sgd)):
				J[tr.s, sgd[i]] = interval_add(J[tr.s, sgd[i]], g[i])
				J[tr.d, sgd[i]] = interval_sub(J[tr.d, sgd[i]], g[i])
		return J
