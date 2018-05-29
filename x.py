# x.py -- one of those stupid python things
#   I'm putting this together to debug intervalUtils.checkExistenceOfSolutionGS.
#   Some bozo decided that indentation should be a syntax issue in python.
#   As best I can tell, the only purpose this serves is to make it impossible
#   to cut-and-paste code from a .py file into the interpretter when you want
#   to check it.  Instead, I cut-and-paste it into an editor session so I ca
#   reformat it to satisfy pythons silly indentation rules.

import numpy as np
import copy
import random
import schmittMosfet
from intervalUtils import *
import mosfetModel

# stuff from prototype2.schmittTriger
modelParam = [-0.4, 0.4, 1.8, 270*1e-6, -90*1e-6]
#model = mosfetModel.MosfetModel(modelParam = modelParam, g_cc = g_cc, g_fwd = 1.0, numStages = numStages)
model = mosfetModel.RambusMosfetMark(modelParam = modelParam, g_cc = 0.5, g_fwd = 1.0, numStages = 2)

# stuff from intervalUtils.checkExistenceOfSolutions

startBounds = np.array([[ 1.4031211 ,  1.4031212],
       [ 0.2507801,  0.2507804],
       [ 1.4031211 ,  1.4031212],
       [ 0.2507801,  0.2507804]])
numVolts = len(startBounds)

if hasattr(model, 'f'):
	funVal = model.f(startBounds)
	print ("funVal", funVal)

	zeroExists = True
	for fi in range(numVolts):
		if funVal[fi,0]*funVal[fi,1] > 0:
			zeroExists = False
			break
	if not(zeroExists):
		print ("invalid bound")

midPoint = (startBounds[:,0] + startBounds[:,1])/2.0
_,_,IMidPoint = np.array(model.oscNum(midPoint))
jacMidPoint = model.jacobian(midPoint)
print ("jacMidPoint", jacMidPoint)
print ("det of jacMidPoint", np.linalg.det(jacMidPoint))
try:
  C = np.linalg.inv(jacMidPoint)
except Exception:
  print "Oops, that didn't go as well as I had hoped.  :("
jacInterval = model.jacobianInterval(startBounds)
# At this point I tried printing jacInterval.  The elements of jacInterval should
#   be all be negative because for any transistor (nfet or pfet), (d Ids)/(d V) < 0
#   for any node voltage V.  I tried this, and found that most of the elements of
#   jacInteraval were intervals spanning 0.  I'll send this to Itrat as a snapshort
#   of getting started on the debugging, and then I'll dig into the mosfet models.

print ("C")
print (C)
#print "condition number of C", np.linalg.cond(C)

#Jacobian interval matrix for startBounds
I = np.identity(numVolts)

jacInterval = model.jacobianInterval(startBounds)
#print "jacInterval"
#print jacInterval
#print "IMidPoint"
#print IMidPoint
C_IMidPoint = np.dot(C,IMidPoint)
#print "C_IMidPoint", C_IMidPoint

C_jacInterval = multiplyRegularMatWithIntervalMat(C,jacInterval)
#print "C_jacInterval"
#print C_jacInterval
I_minus_C_jacInterval = subtractIntervalMatFromRegularMat(I,C_jacInterval)
#print "I_minus_C_jacInterval"
#print I_minus_C_jacInterval
xi_minus_midPoint = np.zeros((numVolts,2))
for i in range(numVolts):
	xi_minus_midPoint[i][0] = min(startBounds[i][0] - midPoint[i], startBounds[i][1] - midPoint[i])
	xi_minus_midPoint[i][1] = max(startBounds[i][0] - midPoint[i], startBounds[i][1] - midPoint[i])
#print "xi_minus_midPoint", xi_minus_midPoint
lastTerm = multiplyIntervalMatWithIntervalVec(I_minus_C_jacInterval, xi_minus_midPoint)
#print "lastTerm "
#print lastTerm

kInterval1 = midPoint - C_IMidPoint + lastTerm[:,0]
kInterval2 = midPoint - C_IMidPoint + lastTerm[:,1]
kInterval = np.zeros((numVolts,2))