# @author Itrat Ahmed Akhter
# Script to run any of the dReal implementations of finding DC
# equilibrium points of the circuit

from dRealLcMosfet import *
from dRealScMosfet import *
from dRealTanh import *

# rambus oscillator with tanh
#allSolutions = rambusOscillatorTanh(numStages = 4, g_cc = 0.5)
# rambus oscillator with long channel mosfet model
#allSolutions = rambusOscillatorLcMosfet(numStages = 2, g_cc = 0.5)
# rambus oscillator with short channel mosfet model
#allSolutions = rambusOscillatorScMosfet(numStages = 2, g_cc = 0.5)

# Schmitt trigger with long channel mosfet model
#allSolutions = schmittTriggerLcMosfet(inputVoltage = 0.9)
# Schmitt trigger with short channel mosfet model
#allSolutions = schmittTriggerScMosfet(inputVoltage = 0.0)

# Inverter with long channel mosfet model
#allSolutions = inverterLcMosfet(inputVoltage = 1.8)
# Inverter with short channel mosfet model
#allSolutions = inverterScMosfet(inputVoltage = 1.0)

print ("allSolutions")
print (allSolutions)