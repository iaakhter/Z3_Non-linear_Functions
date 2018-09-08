# @author Itrat Ahmed Akhter
# Script to run any of the z3 implementations of finding DC
# equilibrium points of circuits

from z3LcMosfet import *
from z3Tanh import *

# rambus oscillator with tanh
#allSolutions = rambusOscillatorTanh(numStages = 2, g_cc = 0.5)
# rambus oscillator with long channel mosfet model
#allSolutions = rambusOscillatorLcMosfet(numStages = 2, g_cc = 0.5)

# Schmitt trigger with long channel mosfet model
#allSolutions = schmittTriggerLcMosfet(inputVoltage = 1.8)

# Inverter with long channel mosfet model
allSolutions = inverterLcMosfet(inputVoltage = 1.8)
