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


class MosfetModel:
	def __init__(self, channelType, Vt, k, s):
		self.channelType = channelType   # 'pfet' or 'nfet'
		self.Vt = Vt                     # threshold voltage
		self.k = k                       # carrier mobility
		self.s = s                       # shape factor

	def __str__(self):
		return "MosfetModel(" + str(self.channelType) + ", " + str(self.Vt) + ", " + str(self.k) + ", " + str(self.s) + ")"

class Mosfet:
	def __init__(self, s, g, d, model):
		self.s = s
		self.g = g
		self.d = d
		self.model = model

	# ids_help: right now, I'm not including any leakage term.  I want to see
	#   if we can find the equilibrium points without it.  That will make the
	#   written explanations simpler.
	def ids_help(self, Vs, Vg, Vd, channelType, Vt, k, s):
		if((Vs.ndim > 0) or (Vg.ndim > 0) or (Vd.ndim > 0)):
			# at least one of Vs, Vg, or Vd is an interval, we should return an interval
			return np.array([
				self.ids_help(my_max(Vs), my_min(Vg), my_min(Vd), channelType, Vt, k, s),
				self.ids_help(my_min(Vs), my_max(Vg), my_max(Vd), channelType, Vt, k, s)])
		elif(channelType == 'pfet'):
			return -self.ids_help(-Vs, -Vg, -Vd, 'nfet', -Vt, -k, s)
		elif(Vd < Vs):
			return -self.ids_help(Vd, Vg, Vs, channelType, Vt, k, s)
		Vgse = (Vg - Vs) - Vt
		Vds = Vd - Vs
		if(Vgse < 0):  # cut-off
			return 0
		elif(Vgse < Vds): # saturation
			return (k*s/2.0)*Vgse*Vgse
		else: # linear
			return k*s*(Vgse - Vds/2.0)*Vds

	def ids(self, V):
		model = self.model
		return(self.ids_help(V[self.s], V[self.g], V[self.d], model.channelType, model.Vt, model.k, model.s))


	# grad_ids: compute the partials of ids wrt. Vs, Vg, and Vd
	#   This function is rather dense.  I would be happier if I could think of
	#    a way to make it more obvious.
	def dg_fun(self, Vs, Vg, Vd, Vt, ks):
		if(Vs[0] > Vd[1]): return None
		Vgse = interval_sub(interval_sub(Vg, np.array([Vs[0], min(Vs[1], Vd[1])])), Vt)
		Vgse[0] = max(Vgse[0], 0)
		Vds = interval_sub(Vd, Vs)
		Vds[0] = max(Vds[0], 0)
		Vx = np.array([Vg[0] - Vt[1] - Vd[1], Vg[1] - Vt[0] - max(Vs[0], Vd[0])])
		Vx[0] = max(Vx[0], 0)
		Vx[1] = max(Vx[1], 0)
		dg = interval_mult(ks, np.array([min(Vgse[0], Vds[0]), Vgse[1]]))
		dd = interval_mult(ks, Vx)
		return np.array([interval_neg(interval_add(dg, dd)), dg, dd])

	def grad_ids_help(self, Vs, Vg, Vd, channelType, Vt, k, s):
		if(channelType == 'pfet'):
			# self.grad_ids_help(-Vs, -Vg, -Vd, 'nfet', -Vt, -k, s)
			# returns the partials of -Ids wrt. -Vs, -Vg, and -Vd,
			# e.g. (d -Ids)/(d -Vs).  The negations cancel out; so
			# we can just return that gradient.
			return self.grad_ids_help(interval_neg(Vs), interval_neg(Vg), interval_neg(Vd), 'nfet', interval_neg(Vt), interval_neg(k), s)
		elif((Vs.ndim > 0) or (Vg.ndim > 0) or (Vd.ndim > 0)):
			Vs = interval_fix(Vs)
			Vg = interval_fix(Vg)
			Vd = interval_fix(Vd)
			Vt = interval_fix(Vt)
			ks = interval_mult(interval_fix(k), interval_fix(s))
			g0 = self.dg_fun(Vs, Vg, Vd, Vt, ks)
			g1x = self.dg_fun(Vd, Vg, Vs, Vt, ks)
			if(g1x is None): g1 = None
			else: g1 = np.array([interval_neg(g1x[2]), interval_neg(g1x[1]), interval_neg(g1x[0])])
			if g0 is None: return g1
			elif g1 is None: return g0
			else: return np.array([interval_union(g0[i], g1[i]) for i in range(len(g0))])
		elif(Vd < Vs):
			gx = self.grad_ids_help(Vd, Vg, Vs, channelType, Vt, k, s)
			return np.array([-gx[2], -gx[1], -gx[0]])
		Vgse = (Vg - Vs) - Vt
		Vds = Vd - Vs
		if(Vgse < 0):  # cut-off: Ids = 0
			return np.array([0,0,0])
		elif(Vgse < Vds): # saturation: Ids = (k*s/2.0)*Vgse*Vgse
			return np.array([-k*s*Vgse, k*s*Vgse, 0])
		else: # linear: k*s*(Vgse - Vds/2.0)*Vds
			dg = k*s*Vds
			dd = k*s*(Vgse - Vds)
			return np.array([-(dg + dd), dg, dd])

	def grad_ids(self, V):
		model = self.model
		return(self.grad_ids_help(V[self.s], V[self.g], V[self.d], model.channelType, model.Vt, model.k, model.s))
			

	def lp_ids(self, V):
		model = self.model
		return(self.lp_ids_help(V[self.s], V[self.g], V[self.d], model.channelType, model.Vt, model.k, model.s))

class Circuit:
	def __init__(self, tr):
		self.tr = tr

	def f(self, V):
		I_node = np.zeros(len(V))
		for i in range(len(self.tr)):
			tr = self.tr[i]
			Ids = tr.ids(V)
			I_node[tr.s] = interval_add(I_node[tr.s], Ids)
			I_node[tr.d] = interval_sub(I_node[tr.d], Ids)
		return I_node

	# Because the Rambus oscillator was our first example, other parts of
	# other parts of the code expect an 'oscNum' function.  I think this
	# if what I'm supposed to provide.
	def oscNum(self, V):
		return [None, None, self.f(V)]

	def jacobian(self, V):
		J = np.zeros([len(V), len(V)])
		for i in range(len(self.tr)):
			tr = self.tr[i]
			g = tr.grad_ids(V)
			sgd = [tr.s, tr.g, tr.d]
			for i in range(len(sgd)):
				J[tr.s, sgd[i]] = interval_add(J[tr.s, sgd[i]], g[i])
				J[tr.d, sgd[i]] = interval_sub(J[tr.s, sgd[i]], g[i])
		return J
