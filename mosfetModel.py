import numpy as np
import copy
from sympy import Plane, Point3d, Line3d, Polygon, Point
from scipy.spatial import ConvexHull


def currentFun(Vin, Vout, Vtp=-0.25, Vtn=0.25, Vdd = 1, Kn = 1, Sn = 1):
	Kp = -Kn/2.0
	Sp = Sn*2.0
	In = 0.0
	firDerInn, firDerOutn = 0.0, 0.0
	secDerInn, secDerOutn = 0.0, 0.0
	if Vin <= Vtn:
		In = 0.0
	    firDerInn = 0.0
	    firDerOutn = 0.0
	    secDerInn = 0.0
	    secDerOutn = 0.0
	elif Vtn <= Vin and Vin <=Vout + Vtn:
		In = Sn*(Kn/2.0)*(Vin - Vtn)*(Vin - Vtn)
		firDerInn = Sn*Kn*(Vin - Vtn)
    	firDerOutn = 0.0
    	secDerInn = Sn*Kn
    	secDerOutn = 0.0
    elif  Vin >= Vout + Vtn:
    	In = Sn*(Kn)*(Vin - Vtn - Vout/2.0)*Vout;
    	firDerInn = Sn*Kn*Vout
    	firDerOutn = -Sn*Kn*Vout
    	secDerInn = 0.0
    	secDerOutn = -Sn*Kn

    Ip = 0.0
    firDerInp, firDerOutp = 0.0, 0.0
    secDerInp, secDerOutp = 0.0, 0.0
    if Vin - Vtp >= Vdd:
    	Ip = 0.0
    	firDerInp = 0.0
    	firDerOutp = 0.0
    	secDerInp = 0.0
    	secDerOutp = 0.0
    elif Vout <= Vin - Vtp and Vin - Vtop <= Vdd:
    	Ip = Sp*(Kp/2.0)*(Vin - Vtp - Vdd)*(Vin - Vtp - Vdd)
    	firDerInp = Sp*Kp*(Vin - Vtp - Vdd)
    	firDerOutp = 0.0
    	secDerInp = Sp*Kp
    	secDerOutp = 0.0
    elif Vin - Vtop <= Vout:
    	Ip = Sp*Kp*((Vin - Vtp - Vdd) - (Vout - Vdd)/2.0)*(Vout - Vdd)
    	firDerInp = Sp*Kp*(Vout - Vdd)
    	firDerOutp = -Sp*Kp*(Vout - Vdd)
    	secDerInp = 0.0
    	secDerOutp = -Sp*Kp

    I = -(In + Ip)
    firDerIn = -(firDerInn + firDerInp)
    firDerOut = -(firDerOutn + firDerOutp)

    secDerIn = -(secDerInn + secDerInp)
    secDerOut = -(secDerOutn + secDerOutp)
    return [I, firDerIn, firDerOut, secDerIn, secDerOut]

# patch is a polygon
def ICrossRegConstraint(I, Vin, Vout, patch, Vtp=-0.25, Vtn=0.25, Vdd = 1, Kn = 1, Sn = 1):
	I = [None]*4
	firDerIn = [None]*4
	firDerOut = [None]*4
	secDerIn = [None]*4
	secDerOut = [None]*4
	# for point1
	[I[0], firDerIn[0], firDerOut[0], secDerIn[0], secDerOut[0]] = currentFun(patch[0][0], patch[0][1], Vtp, Vtn, Vdd, Kn, Sn)
	# for point2
	[I[1], firDerIn[1], firDerOut[1], secDerIn[1], secDerOut[1]] = currentFun(patch[1][0], patch[1][1], Vtp, Vtn, Vdd, Kn, Sn)
	# for point3
	[I[2], firDerIn[2], firDerOut[2], secDerIn[2], secDerOut[2]] = currentFun(patch[2][0], patch[2][1], Vtp, Vtn, Vdd, Kn, Sn)
	# for point4
	[I[3], firDerIn[3], firDerOut[3], secDerIn[3], secDerOut[3]] = currentFun(patch[3][0], patch[3][1], Vtp, Vtn, Vdd, Kn, Sn)

	if (secDerIn[1] >= 0 and secDerOut[1] >= 0 and secDerIn[2] >=0 and secDerOut[2] >= 0 and secDerIn[3] >= 0 and secDerOut[3] >= 0 \
		and secDerIn[0] >= 0 and secDerOut[0] >= 0) or (\
		secDerIn[1] <= 0 and secDerOut[1] <= 0 and secDerIn[2] <=0 and secDerOut[2] <= 0 and secDerIn[3] <= 0 and secDerOut[3] <= 0 \
		and secDerIn[0] <= 0 and secDerOut[0] <= 0):

		return IRegConstraint(I, Vin, Vout, patch, Vtp, Vtn, Vdd, Kn, Sn)[0]

	polygonRegs = [None]*7
	polygonRegs[0] = Polygon(Point(0.0,0.0), Point(Vtn, 0.0), Point(0.0,-Vtp), Point(Vtn, Vtn - Vtp))
	polygonRegs[1] = Polygon(Point(0.0,-Vtp), Point(Vtn, Vtn - Vtp), Point(0.0,1.0), Point(Vtn, 1.0))
	polygonRegs[2] = Polygon(Point(Vtn,0.0), Point(1 + Vtp, 0.0), Point(1 + Vtp, 1 + Vtp - Vtn))
	polygonRegs[3] = Polygon(Point(Vtn,0.0), Point(1 + Vtp, 1 + Vtp - Vtn), Point(Vtn,Vtn -Vtp), Point(1 + Vtp, 1.0))
	polygonRegs[4] = Polygon(Point(Vtn,Vtn -Vtp), Point(Vtn, 1.0), Point(1 + Vtp, 1.0))
	polygonRegs[5] = Polygon(Point(1 + Vtp,0.0), Point(1.0, 0.0), Point(1 + Vtp, 1 + Vtp - Vtn), Point(1.0, 1 - Vtn))
	polygonRegs[6] = Polygon(Point(1 + Vtp,1 + Vtp - Vtn), Point(1.0, 1 - Vtn), Point(1 + Vtp, 1), Point(1.0, 1.0))

	feasiblePlanes = []
	possiblePoints = []
	for polygon in polygonRegs:
		intersect = patch.intersect(polygon)
		if len(intersect >1):
			_,regPoints,regPlanes = IRegConstraint(I, Vin, Vout, intersect, Vtp, Vtn, Vdd, Kn, Sn)
			possiblePoints += regPoints
			feasiblePlanes += regPlanes

	feasiblePoints = []
	for point in possiblePoints:
		pointFeasible = True
		for planeSign in feasiblePlanes:
			plane = planeSign[0]
			sign = planeSign[1]
			normal = plane.normal_vector
			planePoint = plane.p1
			IAtPt = (-normal[0]*(point[0] - planePoint[0])\
				-normal[1]*(point[1] - planePoint[1]))/normal[2] + planePoint[2]

			if sign == " <= ":
				if point[2] > IAtPt:
					pointFeasible = False
					break

			if sign == " >= ":
				if point[2] < IAtPt:
					pointFeasible = False
					break
		if pointFeasible:
			feasiblePoints.append([point[0], point[1], point[2]])

	# Now construct convex hull with feasible points and add the constraint
	feasiblePointsNp = np.array(feasiblePoints)
	hull = ConvexHull(feasiblePointsNp)


# point2 point3
# point0 point1
def IRegConstraint(I, Vin, Vout, patch, Vtp=-0.25, Vtn=0.25, Vdd = 1, Kn = 1, Sn = 1):
	
	# for point1
	[I0, firDerIn0, firDerOut0, secDerIn0, secDerOut0] = currentFun(patch[0][0], patch[0][1], Vtp, Vtn, Vdd, Kn, Sn)
	# for point2
	[I1, firDerIn1, firDerOut1, secDerIn1, secDerOut1] = currentFun(patch[1][0], patch[1][1], Vtp, Vtn, Vdd, Kn, Sn)
	# for point3
	[I2, firDerIn2, firDerOut2, secDerIn2, secDerOut2] = currentFun(patch[2][0], patch[2][1], Vtp, Vtn, Vdd, Kn, Sn)
	# for point4
	[I3, firDerIn3, firDerOut3, secDerIn3, secDerOut3] = currentFun(patch[3][0], patch[3][1], Vtp, Vtn, Vdd, Kn, Sn)


	if not((secDerIn1 >= 0 and secDerOut1 >= 0 and secDerIn2 >=0 and secDerOut2 >= 0 and secDerIn3 >= 0 and secDerOut3 >= 0 \
		and secDerIn0 >= 0 and secDerOut0 >= 0) or (\
		secDerIn1 <= 0 and secDerOut1 <= 0 and secDerIn2 <=0 and secDerOut2 <= 0 and secDerIn3 <= 0 and secDerOut3 <= 0 \
		and secDerIn0 <= 0 and secDerOut0 <= 0)):
		return None

	overallConstraint = ""

	point0 = Point3d(patch[0][0], patch[0][1], I1)
	point1 = Point3d(patch[1][0], patch[1][1], I2)
	point2 = Point3d(patch[2][0], patch[2][1], I3)
	point3 = Point3d(patch[3][0], patch[3][1], I4)

	possibleSecantPlanes = []
	excludedPoints = []
	possibleSecantPlanes.append(Plane(point0, point1, point2))
	excludedPoints.append(point3)

	possibleSecantPlanes.append(Plane(point0, point1, point3))
	excludedPoints.append(point2)

	possibleSecantPlanes.append(Plane(point0, point2, point3))
	excludedPoints.append(point1)

	possibleSecantPlanes.append(Plane(point1, point2, point3))
	excludedPoints.append(point0)

	tangentSign = " >= "
	secantSign = " <= "

	# concave downward constraint - function should be less than equal to tangent plane and
	# greater than equal to secant planes
	if secDerIn0 < 0:
		tangentSign = " <= "
		secantSign = " >= "

	# a list of point and normal
	feasiblePlanes = []
	# tangent constraints
	overallConatraint += "1 " + I + " + " + str(-firDerIn0) + " " + Vin +\
	" + " + str(-firDerOut0) + " " + Vout + tangentSign + str(-firDerIn0*patch[0][0] - firDerOut0*patch[0][1] + I0)
	feasiblePlanes.append([Plane(Point3d(patch[0][0], patch[0][1], I0),normal_vector =(firDerIn0, firDerOut0, -1)), tangentSign])
	
	overallConatraint += "1 " + I + " + " + str(-firDerIn1) + " " + Vin +\
	" + " + str(-firDerOut1) + " " + Vout + tangentSign + str(-firDerIn1*patch[1][0] - firDerOut1*Vouthigh + I1)
	feasiblePlanes.append([Plane(Point3d(patch[1][0], patch[1][1], I1),normal_vector = (firDerIn1, firDerOut1, -1)), tangentSign])
	
	overallConatraint += "1 " + I + " + " + str(-firDerIn2) + " " + Vin +\
	" + " + str(-firDerOut2) + " " + Vout + tangentSign + str(-firDerIn2*patch[2][0] - firDerOut2*patch[2][1] + I2)
	feasiblePlanes.append([Plane(Point3d(patch[2][0], patch[2][1], I2),normal_vector = (firDerIn2, firDerOut2, -1)), tangentSign])
	
	overallConatraint += "1 " + I + " + " + str(-firDerIn3) + " " + Vin +\
	" + " + str(-firDerOut3) + " " + Vout + tangentSign + str(-firDerIn3*patch[3][0] - firDerOut3*patch[3][1] + I3)
	feasiblePlanes.append([Plane(Point3d(patch[3][0], patch[3][1], I3),normal_vector = (firDerIn3, firDerOut3, -1)), tangentSign])

	# secant constraints
	numSecantConstraints = 0
	for plane in possibleSecantPlanes:
		if numSecantConstraints >= 2:
			break
		normal = plane.normal_vector
		includedPt = plane.p1
		excludedPt = excludedPoints[i]
		# check if excluded point feasible with plane as a secant
		IAtExcludedPt = (-normal[0]*(excludedPt[0] - includedPt[0])\
			-normal[1]*(excludedPt[1] - includedPt[1]))/normal[2] + includedPt[2]

		feasible = includedPt[2] <= IAtExcludedPt
		if secantSign == " >= ":
			feasible = includedPt[2] >= IAtExcludedPt
		if feasible:
			overallConatraint += "1 " + I + " + " + str(normal[0]/normal[2]) + " " + Vin +\
			" + " + str(normal[1]/normal[2]) + " " + Vout + secantSign + str((normal[0]/normal[2])*includedPt[0] +\
				(normal[1]/normal[2])*includedPt[1] + includedPt[2])
			numSecantConstraints += 1
			feasiblePlanes.append([plane, secantSign])

	feasiblePoints = []
	# find intersection of all possible combinations of feasible planes
	# store points that are satisfied by all constraints
	for i in range(len(feasiblePlanes)):
		for j in range(i+1, len(feasiblePlanes)):
			for k in range(j+1, len(feasiblePlanes)):
				intersectingLine = feasiblePlanes[i].intersection(feasiblePlanes[j])
				intersectingPoint = feasiblePlanes[k].intersection(intersectingLine)
				feasiblePoints.append(intersectingPoint)


	return overallConstraint, feasiblePoints, feasiblePlanes



def oscNum(V,g_cc,g_fwd = 1,Vtp=-0.25, Vtn=0.25, Vdd = 1, Kn = 1, Sn = 1):
	lenV = len(V)
	Vin = [V[i % lenV] for i in range(-1,lenV-1)]
	Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
	IFwd = [currentFun(Vin[i], V[i], Vtp, Vtn, Vdd, Kn, Sn)[0] for i in range(lenV)]
	ICc = [currentFun(Vcc[i], V[i], Vtp, Vtn, Vdd, Kn, Sn)[0] for i in range(lenV)]
	return [Ifwd[i]*g_fwd + Icc[i]*g_cc) for i in range(lenV)]

def getJacobian(V,g_cc,g_fwd = 1,Vtp=-0.25, Vtn=0.25, Vdd = 1, Kn = 1, Sn = 1):
	lenV = len(V)
	Vin = [V[i % lenV] for i in range(-1,lenV-1)]
	Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
	jacobian = np.zeros((lenV, lenV))
	for i in range(lenV):
		[Ifwd, firDerInfwd, firDerOutfwd, secDerInfwd, secDerOutfwd] = currentFun(Vin[i], V[i], Vtp, Vtn, Vdd, Kn, Sn)
		[Icc, firDerIncc, firDerOutcc, secDerIncc, secDerOutcc] = currentFun(Vcc[i], V[i], Vtp, Vtn, Vdd, Kn, Sn)
		jacobian[i, (i-1)%lenV] = g_fwd*firDerInfwd
		jacobian[i, (i + lenV/2) % lenV] = g_cc*firDerIncc
		jacobian[i, i] = g_fwd*firDerOutfwd + g_cc*firDerOutcc
	return jacobian

# numerical approximation where i sample jacobian in the patch by bounds
# might need a more analytical way of solving this later
def getJacobianInterval(bounds,g_cc,g_fwd=1,Vtp=-0.25, Vtn=0.25, Vdd = 1, Kn = 1, Sn = 1):
	lowerBound = bounds[:,0]
	upperBound = bounds[:,1]
	lenV = len(lowerBound)
	jacobian = np.zeros((lenV, lenV,2))
	jacobian[:,:,0] = jacobian[:,:,0] 
	jacobian[:,:,1] = jacobian[:,:,1]
	zerofwd =  g_fwd*tanhFunder(a,0)
	zerocc = g_cc*tanhFunder(a,0)
	for i in range(lenV):
		lowOut = lowerBound[i]
		highOut = upperBound[i]
		rangeOut = highOut - lowOut

		minFirDerInFwd, maxFirDerInFwd = float("inf"), -float("inf")
		minFirDerOutFwd, maxFirDerOutFwd = float("inf"), -float("inf")
		
		lowFwd = lowerBound[(i-1)%lenV]
		highFwd = upperBound[(i-1)%lenV]
		rangeFwd = highFwd - lowFwd
		for vin in range(lowFwd, highFwd, rangeFwd/100.0):
			for vout in range(lowOut, highOut, rangeOut/100.0):
				[Ifwd, firDerInfwd, firDerOutfwd, secDerInfwd, secDerOutfwd] = currentFun(vin, vout, Vtp, Vtn, Vdd, Kn, Sn)
				minFirDerInFwd = min(minFirDerInFwd, firDerInfwd)
				maxFirDerInFwd = max(maxFirDerInFwd, firDerInfwd)
				minFirDerOutFwd = min(minFirDerOutFwd, firDerOutfwd)
				maxFirDerOutFwd = max(maxFirDerOutFwd, firDerOutfwd)

		minFirDerInCc, maxFirDerInCc = float("inf"), -float("inf")
		minFirDerOutCc, maxFirDerOutCc = float("inf"), -float("inf")
		
		lowCc = lowerBound[(i-1)%lenV]
		highCc = upperBound[(i-1)%lenV]
		rangeCc = highCc - lowCc
		for vin in range(lowCc, highCc, rangeCc/100.0):
			for vout in range(lowOut, highOut, rangeOut/100.0):
				[Icc, firDerIncc, firDerOutcc, secDerIncc, secDerOutcc] = currentFun(vin, vout, Vtp, Vtn, Vdd, Kn, Sn)
				minFirDerInCc = min(minFirDerInCc, firDerIncc)
				maxFirDerInCc = max(maxFirDerInCc, firDerIncc)
				minFirDerOutCc = min(minFirDerOutCc, firDerOutcc)
				maxFirDerOutCc = max(maxFirDerOutCc, firDerOutcc)

		jacobian[i, (i-1)%lenV, 0] = g_fwd*minFirDerInFwd
		jacobian[i, (i-1)%lenV, 1] = g_fwd*maxFirDerInFwd
		jacobian[i, (i + lenV/2) % lenV, 0] = g_cc*minFirDerIncc
		jacobian[i, (i + lenV/2) % lenV, 1] = g_cc*maxFirDerIncc
		jacobian[i, i, 0] = g_fwd*minFirDerOutfwd + g_cc*minFirDerOutcc
		jacobian[i, i, 1] = g_fwd*maxFirDerOutfwd + g_cc*maxFirDerOutcc

	return jacobian


