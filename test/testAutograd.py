import autograd.numpy as anp
from autograd import grad
from autograd import elementwise_grad as egrad
from autograd import jacobian as ajac
from autograd.extend import primitive, defvjp
import numpy as np


# x should be of the type [xlow, xhigh]
@primitive
def tanhFunInterval(x):
	tanhVal1 = anp.tanh(x[0])
	tanhVal2 = anp.tanh(x[1])
	tanhValInterval = anp.zeros((x.shape))
	tanhValInterval[0] = anp.minimum(tanhVal1, tanhVal2)
	tanhValInterval[1] = anp.maximum(tanhVal1, tanhVal2)
	return tanhValInterval

# x should be of the type [[xlow, xhigh]]å
def tanhGradInterval(ans, x):
	def gradient_product(g):
		#print ("len(x)", len(x))
		#print ("g",g)
		#print ("x[:,0]", x[:,0])
		#print ("ans", ans)
		val1 = g[0]*anp.divide(1,anp.cosh(x[0]))*anp.divide(1,anp.cosh(x[0]))
		val2 = g[1]*anp.divide(1,anp.cosh(x[0]))*anp.divide(1,anp.cosh(x[0]))
		val3 = g[0]*anp.divide(1,anp.cosh(x[1]))*anp.divide(1,anp.cosh(x[1]))
		val4 = g[1]*anp.divide(1,anp.cosh(x[1]))*anp.divide(1,anp.cosh(x[1]))
		gradResult =  anp.zeros(x.shape)
		#print ("val1", val1)
		#print ("val2", val2)
		#print ("val3", val3)
		#print ("val4", val4)
		#print ("minVal", anp.minimum(anp.minimum(anp.minimum(val1, val2), val3), val4))
		gradResult[0] = anp.minimum(anp.minimum(anp.minimum(val1, val2), val3), val4)
		gradResult[1] = anp.maximum(anp.maximum(anp.maximum(val1, val2), val3), val4)
		#print ("gradResult")
		#print (gradResult)
		
		if x[0]*x[1] < 0:
			val0_1 = g[0]*anp.divide(1,anp.cosh(anp.zeros(x[0].shape)))*anp.divide(1,anp.cosh(anp.zeros(x[0].shape)))
			val0_2 = g[1]*anp.divide(1,anp.cosh(anp.zeros(x[0].shape)))*anp.divide(1,anp.cosh(anp.zeros(x[0].shape)))
			minVal0 = anp.minimum(val0_1, val0_2)
			maxVal0 = anp.maximum(val0_1, val0_2)
			gradResult[0] = anp.minimum(gradResult[0], minVal0)
			gradResult[1] = anp.maximum(gradResult[1], maxVal0)

		return gradResult
	return gradient_product

def tanhFunConstInterval(x, modelParam):
	x = x*modelParam
	if modelParam < 0:
		#print ("x", x)
		newX = anp.array([x[1], x[0]])
	else:
		newX = x
	return tanhFunInterval(newX)

def rambusTanhInterval(V):
	modelParam = -5.0
	g_fwd = 1.0
	g_cc = 0.5
	lenV = len(V)
	
	fVal = []
	for i in range(lenV):
		fwdInd = (i-1)%lenV
		ccInd = (i + lenV//2) % lenV
		tanhFwd = tanhFunConstInterval(anp.array(V[fwdInd]), modelParam)
		tanhCc = tanhFunConstInterval(anp.array(V[ccInd]), modelParam)
		fwdTerm = anp.array([tanhFwd[0] - V[i,1], tanhFwd[1] - V[i,0]])
		ccTerm = anp.array([tanhCc[0] - V[i,1], tanhCc[1] - V[i,0]])
		fVal.append(g_fwd*fwdTerm + g_cc*ccTerm)

	return anp.array(fVal)


def tanhFun(x, const):
	tanhVal = anp.tanh(const*x)
	return tanhVal


def rambusTanh(V):
	modelParam = -5.0
	g_fwd = 1.0
	g_cc = 0.5
	lenV = len(V)
	
	fVal = []
	for i in range(lenV):
		fwdInd = (i-1)%lenV
		ccInd = (i + lenV//2) % lenV
		tanhFwd = tanhFun(V[fwdInd], modelParam)
		tanhCc = tanhFun(V[ccInd], modelParam)
		fwdTerm = tanhFwd - V[i]
		ccTerm = tanhCc - V[i]
		fVal.append(g_fwd*fwdTerm + g_cc*ccTerm)

	return anp.array(fVal)


defvjp(tanhFunInterval, tanhGradInterval)
#gradOfTanhInterval = egrad(tanhFunConstInterval)
#print ("gradient", gradOfTanhInterval(anp.array([-1.0,1.0]), -5))

jac_rambusInterval = ajac(rambusTanhInterval)
print ("autograd calculated jacobian")
print (jac_rambusInterval(anp.array([[-0.3, 0.0], [0.0, 0.3], [-0.3, 0.0], [0.0, 0.3]])))

'''grad_rambus = egrad(rambusTanh)
jac_rambus = ajac(rambusTanh)
print ("example voltage", np.array([-0.3, 0.3, -0.3, 0.3]))
print ("autograd calculated element wise grad")
print (grad_rambus(anp.array([-0.3, 0.3, -0.3, 0.3])))
print ("autograd calculated jacobian")
print (jac_rambus(anp.array([-0.3, 0.3, -0.3, 0.3])))'''

