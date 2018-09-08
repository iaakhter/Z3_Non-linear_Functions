The goal of this project is to find DC equilibrium points of circuits using interval verification algorithms like the Krawczyk operator. Our main example is Rambus ring oscillator challenge problem - http://www.cs.um.edu.mt/gordon.pace/Workshops/DCC2008/Presentations/10.pdf.
We also find DC equilibrium points for the Schmitt Trigger.

All of the implementations of our programs have been done in python2.7
Our main main method is implemented in prototype.py. It gives a list of hyperrectangles each of which contains a unique DC equlibrium point.
To run prototype.py - install numpy and cvxopt and run "python prototype.py" in the terminal.

Under the 'if __name__ == "__main__"' section towards the end of protoype.py, comment and uncomment to run a specific specific example with a specific model.
For example to run the rambus oscillator example uncomment the following line

#allHypers = rambusOscillator(modelType="tanh", numStages=2, g_cc=0.5, statVars=statVars, numSolutions="all")

The modelType indicates the way each inverter in the oscillator has been modeled. 
When modelType == "tanh", each inverter is represented by a tanh model with a negative gain. 
When modelType == "lcMosfet", each inverter is represented by 2 CMOS transistors with a long channel MOSFET model.
When modelType == "scMosfet", each inverter is represented by 2 CMOS transistors with a short channel MOSFET model.
numStages is the number of stages in the oscillator (Should be even)
g_cc represents the strength of the cross-coupled nodes in the oscillator in relation to the forward nodes. For example if g_cc = 0.5, the cross coupled nodes are 0.5 times stronger than the forward nodes.

To run the schmitt trigger example for a specific input voltage, uncomment the line:

#allHypers = schmittTrigger(modelType="lcMosfet", inputVoltage = 0.5, statVars=statVars, numSolutions = "all")

Again, like the rambus oscillator, modelType == "lcMosfet" indicates long channel model and modelType == "scMosfet" indicates short channel model.  

##################################################################################

We also compare our results with dReal and Z3. 

To install dReal with python bindings - follow the instructions in:
https://github.com/dreal/dreal4. 
To run the dReal versions of our problem, cd into dRealCode directory and run "python dRealAll.py" in the terminal. Comment and uncomment the relevant lines to run specific examples.

To install z3 with python bindings - follow the instructions in:
https://github.com/Z3Prover/z3.
To run the z3 versions of our problem, cd into z3Code directory and run "python z3All.py" in the terminal. Comment and uncomment the relevant lines to run the specific examples.