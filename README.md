The goal of this project is to find DC equilibrium points of circuits using interval verification algorithms like the Krawczyk operator. Our main example is Rambus ring oscillator challenge problem - http://www.cs.um.edu.mt/gordon.pace/Workshops/DCC2008/Presentations/10.pdf.
We also find DC equilibrium points for the Schmitt Trigger.

All of the implementations of our programs have been done in python2.7 and require numpy and cvxopt packages.

# Overall Structure

Our main main method is implemented in **prototype.py**. It gives a list of hyperrectangles each of which contains a unique DC equlibrium point.

Note that this method will only work properly with well-conditioned systems where the jacobians are not singular or badly conditioned at the solutions.

Our main algorithm is implemented in the function `solverLoopNoLp` in **prototype.py**. The two main parameters for this function is uniqueHypers (which should be an empty list when the function is called) and a model object. uniqueHypers will hold the list of hyperrectangles containing unique solution to a function f. The model class should have a method called f which gives point and interval evaluation of the function we are trying to solve. f should take a numpy array of variable values. If any entry in the array is an interval, f returns an interval value. Otherwise it returns a point. Similarly, the model class should also have a method called jacobian that returns jacobian and jacobian interval depending on the arguments. The model class should also have a field called bounds that specifies bounds for each variable. Look at `RambusTanh` class in **circuitModels.py** for more details. After having defined the model class, in prototype.py it is enough to call `solverLoop(uniqueHypers, model)` to find all the hyperrectangles containing unique solutions. uniqueHypers will contain them.

# Defining Custom Circuit

It is also possible to define a custom circuit with long channel and short channel MOSFET models and use our method to find DC equilibrium points. 

The `Circuit` class in **circuit.py** takes a list of transistors. Each transistor can be a short channel transistor or long channel transistor. The transistor object will take in indices for source, drain and gate from an array of voltages. We can specify these requirements in a model class and extract the required function and jacobian information from the `Circuit` object. An example of creating and using a circuit with long channel or short channel MOSFET in a model object can be seen in `RambusMosfet` class in **circuitModels.py**.

# Running Our Examples:

Our rambus oscillator and schmitt trigger examples can be run from **examples.py**. Comment and uncomment to run a specific specific example with a specific model.
For example to run the rambus oscillator example uncomment the following line

`#allHypers = rambusOscillator(modelType="tanh", numStages=2, g_cc=0.5, statVars=statVars, numSolutions="all")`

The modelType indicates the way each inverter in the oscillator has been modeled. 
When `modelType == "tanh"`, each inverter is represented by a tanh model with a negative gain. 
When `modelType == "lcMosfet"`, each inverter is represented by 2 CMOS transistors with a long channel MOSFET model.
When `modelType == "scMosfet"`, each inverter is represented by 2 CMOS transistors with a short channel MOSFET model.
numStages is the number of stages in the oscillator (Should be even)
g_cc represents the strength of the cross-coupled nodes in the oscillator in relation to the forward nodes. For example if g_cc = 0.5, the cross coupled nodes are 0.5 times stronger than the forward nodes.

To run the schmitt trigger example for a specific input voltage, uncomment the line:

`#allHypers = schmittTrigger(modelType="lcMosfet", inputVoltage = 0.5, statVars=statVars, numSolutions = "all")`

Again, like the rambus oscillator, `modelType == "lcMosfet"` indicates long channel model and `modelType == "scMosfet"` indicates short channel model.  

# Comparisons with dReal and Z3

We also compare our results with dReal and Z3. 

To install dReal with python bindings - follow the instructions in:
https://github.com/dreal/dreal4. 
To run the dReal versions of our problem, cd into dRealCode directory and run `python dRealAll.py` in the terminal. Comment and uncomment the relevant lines to run specific examples.

To install z3 with python bindings - follow the instructions in:
https://github.com/Z3Prover/z3.
To run the z3 versions of our problem, cd into z3Code directory and run `python z3All.py` in the terminal. Comment and uncomment the relevant lines to run the specific examples.