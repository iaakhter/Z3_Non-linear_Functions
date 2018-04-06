import numpy as np
import matplotlib.pyplot as plt
import mosfetModel

numStages = 2
modelParam = [-0.4, 0.4, 1.8, 1.5, 8/3.0]
model = mosfetModel.MosfetModel(modelParam = modelParam, g_cc = 0.5, g_fwd = 1.0, numStages = numStages)

Vs = np.arange(0,2.01,0.01)
Is = np.zeros((len(Vs)))
for i in range(len(Vs)):
	voltages = [Vs[i], 1.8 - Vs[i], Vs[i], 1.8-Vs[i]]
	_,_,currents = model.oscNum(voltages)
	Is[i] = currents[0]
	if Is[i] < 0:
		print ("Vs[i] at zero", Vs[i])

_,_,testCurrents = model.oscNum([0.84948974,  0.84948974,  0.84948974,  0.84948974])
print ("testCurrents", testCurrents)

plt.plot(Vs,Is,Vs,np.zeros((len(Vs))))
plt.show()
