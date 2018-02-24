import numpy as np
import lpUtils
from cvxopt import matrix,solvers
from sympy import Plane, Point3D, Line3D, Line
from sympy import Polygon, Point
from scipy.spatial import ConvexHull

class MosfetModel:
	# modelParam = [Vtp, Vtn, Vdd, Kn, Sn]
	def __init__(self, modelParam, g_cc, g_fwd, numStages):
		# gradient of tanh -- y = tanh(modelParam*x)
		self.Vtp = modelParam[0]
		self.Vtn = modelParam[1]
		self.Vdd = modelParam[2]
		self.Kn = modelParam[3]
		self.Sn = modelParam[4]
		self.Kp = -self.Kn/2.0
		self.Sp = self.Sn*2.0
		self.g_cc = g_cc
		self.g_fwd = g_fwd
		self.numStages = numStages
		self.xs = []
		self.IsFwd = []
		self.IsCc = []
		for i in range(numStages*2):
			self.xs.append("x" + str(i))
			self.IsFwd.append("ifwd" + str(i))
			self.IsCc.append("icc" + str(i))

	def currentFun(self, Vin, Vout):
		In = 0.0
		firDerInn, firDerOutn = 0.0, 0.0
		secDerInn, secDerOutn = 0.0, 0.0
		if Vin <= self.Vtn:
			In = 0.0
			firDerInn = 0.0
			firDerOutn = 0.0
			secDerInn = 0.0
			secDerOutn = 0.0
		elif self.Vtn <= Vin and Vin <=Vout + self.Vtn:
			In = self.Sn*(self.Kn/2.0)*(Vin - self.Vtn)*(Vin - self.Vtn)
			firDerInn = self.Sn*self.Kn*(Vin - self.Vtn)
			firDerOutn = 0.0
			secDerInn = self.Sn*self.Kn
			secDerOutn = 0.0
		elif  Vin >= Vout + self.Vtn:
			In = self.Sn*(self.Kn)*(Vin - self.Vtn - Vout/2.0)*Vout;
			firDerInn = self.Sn*self.Kn*Vout
			firDerOutn = -self.Sn*self.Kn*Vout
			secDerInn = 0.0
			secDerOutn = -self.Sn*self.Kn

		Ip = 0.0
		firDerInp, firDerOutp = 0.0, 0.0
		secDerInp, secDerOutp = 0.0, 0.0
		if Vin - self.Vtp >= self.Vdd:
			Ip = 0.0
			firDerInp = 0.0
			firDerOutp = 0.0
			secDerInp = 0.0
			secDerOutp = 0.0
		elif Vout <= Vin - self.Vtp and Vin - self.Vtp <= self.Vdd:
			Ip = self.Sp*(self.Kp/2.0)*(Vin - self.Vtp - self.Vdd)*(Vin - self.Vtp - self.Vdd)
			firDerInp = self.Sp*self.Kp*(Vin - self.Vtp - self.Vdd)
			firDerOutp = 0.0
			secDerInp = self.Sp*self.Kp
			secDerOutp = 0.0
		elif Vin - self.Vtp <= Vout:
			Ip = self.Sp*self.Kp*((Vin - self.Vtp - self.Vdd) - (Vout - self.Vdd)/2.0)*(Vout - self.Vdd)
			firDerInp = self.Sp*self.Kp*(Vout - self.Vdd)
			firDerOutp = -self.Sp*self.Kp*(Vout - self.Vdd)
			secDerInp = 0.0
			secDerOutp = -self.Sp*self.Kp

		I = -(In + Ip)
		firDerIn = -(firDerInn + firDerInp)
		firDerOut = -(firDerOutn + firDerOutp)

		secDerIn = -(secDerInn + secDerInp)
		secDerOut = -(secDerOutn + secDerOutp)
		return [I, firDerIn, firDerOut, secDerIn, secDerOut]

	# patch is a polygon
	def ICrossRegConstraint(self,I, Vin, Vout, patch):
		INum = [None]*4
		firDerIn = [None]*4
		firDerOut = [None]*4
		secDerIn = [None]*4
		secDerOut = [None]*4

		for i in range(len(INum)):
			[INum[i], firDerIn[i], firDerOut[i], secDerIn[i], secDerOut[i]] = self.currentFun(patch.vertices[i][0], patch.vertices[i][1])

		if (secDerIn[1] >= 0 and secDerOut[1] >= 0 and secDerIn[2] >=0 and secDerOut[2] >= 0 and secDerIn[3] >= 0 and secDerOut[3] >= 0 \
			and secDerIn[0] >= 0 and secDerOut[0] >= 0) or (\
			secDerIn[1] <= 0 and secDerOut[1] <= 0 and secDerIn[2] <=0 and secDerOut[2] <= 0 and secDerIn[3] <= 0 and secDerOut[3] <= 0 \
			and secDerIn[0] <= 0 and secDerOut[0] <= 0):

			return self.IRegConstraint(I, Vin, Vout, patch)[0]

		# the different regions with respect to the sign of the second derivative
		# of I
		polygonRegs = [None]*7
		polygonRegs[0] = Polygon(Point(0.0,0.0), Point(self.Vtn, 0.0), Point(self.Vtn, self.Vtn - self.Vtp), Point(0.0,-self.Vtp))
		polygonRegs[1] = Polygon(Point(0.0,-self.Vtp), Point(self.Vtn, self.Vtn - self.Vtp), Point(self.Vtn, 1.0), Point(0.0,1.0))
		polygonRegs[2] = Polygon(Point(self.Vtn,0.0), Point(1 + self.Vtp, 0.0), Point(1 + self.Vtp, 1 + self.Vtp - self.Vtn))
		polygonRegs[3] = Polygon(Point(self.Vtn,0.0), Point(1 + self.Vtp, 1 + self.Vtp - self.Vtn), Point(1 + self.Vtp, 1.0), Point(self.Vtn,self.Vtn -self.Vtp))
		polygonRegs[4] = Polygon(Point(self.Vtn,self.Vtn -self.Vtp), Point(1 + self.Vtp, 1.0), Point(self.Vtn, 1.0))
		polygonRegs[5] = Polygon(Point(1 + self.Vtp,0.0), Point(1.0, 0.0), Point(1.0, 1 - self.Vtn), Point(1 + self.Vtp, 1 + self.Vtp - self.Vtn))
		polygonRegs[6] = Polygon(Point(1 + self.Vtp,1 + self.Vtp - self.Vtn), Point(1.0, 1 - self.Vtn), Point(1.0, 1.0), Point(1 + self.Vtp, 1))

		feasiblePoints = []
		# If one of the regions intersects with the hyperrectangle
		# Construct constraints for that intersection
		for polygon in polygonRegs:
			print "polygon ", polygon
			print "patch ", patch
			intersect = None
			polyVertsInsidePatch = True
			polyVertsOutsidePatch = True
			for vert in polygon.vertices:
				if not(patch.encloses(vert)):
					onLine = False
					for side in patch.sides:
						if side.contains(vert):
							onLine = True
							break
					if not(onLine):
						polyVertsInsidePatch = False
				if patch.encloses(vert):
					polyVertsOutsidePatch = False
				else:
					onLine = False
					for side in patch.sides:
						if side.contains(vert):
							onLine = True
							break
					if onLine:
						polyVertsOutsidePatch = False

			patchVertsOutsidePoly = True
			for vert in patch.vertices:
				if polygon.encloses(vert):
					patchVertsOutsidePoly = False
				else:
					onLine = False
					for side in polygon.sides:
						if side.contains(vert):
							onLine = True
							break
					if onLine:
						patchVertsOutsidePoly = False

			#print ("polyVertsOutsidePatch", polyVertsOutsidePatch)
			#print ("polyVertsInsidePatch", polyVertsInsidePatch)
			#print ("patchVertsOutsidePoly", patchVertsOutsidePoly)
			if polyVertsOutsidePatch and patchVertsOutsidePoly:
				continue

			if polyVertsInsidePatch:
				intersect = polygon

			else:
				intersectionPoints = []
				for vi in range(len(patch.vertices)):
					if polygon.encloses_point(patch.vertices[vi]):
						intersectionPoints.append(patch.vertices[vi])
					else:
						pointOnLine = False
						for side in polygon.sides:
							if side.contains(patch.vertices[vi]):
								pointOnLine = True
								break
						if pointOnLine:
							intersectionPoints.append(patch.vertices[vi])

						else:
							if patch.vertices[vi][0] >= polygon.bounds[0] and patch.vertices[vi][0] <= polygon.bounds[2]:
								polygonLowLine, polygonHighLine = None, None
								polygonLowLine = Line(polygon.vertices[0], polygon.vertices[1])
								if len(polygon.vertices) == 4:									
									polygonHighLine = Line(polygon.vertices[2], polygon.vertices[3])
								if len(polygon.vertices) == 3:
									if polygon.vertices[0][1] == 0:
										polygonHighLine = Line(polygon.vertices[2], polygon.vertices[0])
									else:
										polygonHighLine = Line(polygon.vertices[1], polygon.vertices[2])
							
								print ("polygonLowLine", polygonLowLine)
								print ("polygonHighLine", polygonHighLine)
								vertLineAtPatchPt = Line(Point(patch.vertices[vi][0],0), Point(patch.vertices[vi][0],1))
								#print ("patch.vertices[vi]", patch.vertices[vi])
								#print ("")
								lowIntersection = polygonLowLine.intersection(vertLineAtPatchPt)
								highIntersection = polygonHighLine.intersection(vertLineAtPatchPt)
								print ("patch.vertices[vi]", patch.vertices[vi])
								print ("vertLineAtPatchPt", vertLineAtPatchPt)
								print ("lowIntersection", lowIntersection)
								print ("highIntersection", highIntersection)
								if len(lowIntersection) >= 1:
									lowIntersection = lowIntersection[0]
									if patch.vertices[vi][1] <= lowIntersection[1]:
										intersectionPoints.append(Point(patch.vertices[vi][0], lowIntersection[1]))
								if len(highIntersection) >= 1:
									highIntersection = highIntersection[0]
									if patch.vertices[vi][1] >= highIntersection[1]:
										intersectionPoints.append(Point(patch.vertices[vi][0], highIntersection[1]))
							
							elif patch.vertices[vi][1] >= polygon.bounds[1] and patch.vertices[vi][1] <= polygon.bounds[3]:
								polygonLeftLine = Line(polygon.vertices[0], polygon.vertices[-1])
								polygonRightLine = Line(polygon.vertices[1], polygon.vertices[2])
								if len(polygon.vertices) == 3:
									if polygon.vertices[0][1] == 0:
										polygonRightLine = Line(polygon.vertices[1], polygon.vertices[2])
									else:
										polygonRightLine = Line(polygon.vertices[0], polygon.vertices[1])
								lowerLeftY = min(polygonLeftLine.p1[1], polygonLeftLine.p2[1])
								upperLeftY = max(polygonLeftLine.p1[1], polygonLeftLine.p2[1])

								lowerRightY = min(polygonRightLine.p1[1], polygonRightLine.p2[1])
								upperRightY = max(polygonRightLine.p1[1], polygonRightLine.p2[1])
								
								print ("polygonLeftLine", polygonLeftLine)
								print ("polygonRightLine", polygonRightLine)
								horLineAtPatchPt = Line(Point(0, patch.vertices[vi][1]), Point(1, patch.vertices[vi][1]))
								print ("patch.vertices[vi]", patch.vertices[vi])
								print ("horLineAtPatchPt", horLineAtPatchPt)
								leftIntersection = polygonLeftLine.intersection(horLineAtPatchPt)
								rightIntersection = polygonRightLine.intersection(horLineAtPatchPt)
								print ("leftIntersection", leftIntersection)
								print ("rightIntersection", rightIntersection)
								if len(leftIntersection) >= 1:
									leftIntersection = leftIntersection[0]
									if patch.vertices[vi][0] <= leftIntersection[0] and patch.vertices[vi][1] >= lowerLeftY and patch.vertices[vi][1] <= upperLeftY:
										intersectionPoints.append(Point(leftIntersection[0], patch.vertices[vi][1]))
									elif patch.vertices[vi][0] <= leftIntersection[0] and patch.vertices[vi][1] <= lowerLeftY:
										intersectionPoints.append(Point(leftIntersection[0], lowerLeftY))
									elif patch.vertices[vi][0] <= leftIntersection[0] and patch.vertices[vi][1] >= upperLeftY:
										intersectionPoints.append(Point(leftIntersection[0], upperLeftY))
								
								if len(rightIntersection) >= 1:
									rightIntersection = rightIntersection[0]
									if patch.vertices[vi][0] >= rightIntersection[0] and patch.vertices[vi][1] >= lowerRightY and patch.vertices[vi][1] <= upperRightY:
										intersectionPoints.append(Point(rightIntersection[0], patch.vertices[vi][1]))
									elif patch.vertices[vi][0] >= rightIntersection[0] and patch.vertices[vi][1] <= lowerRightY:
										intersectionPoints.append(Point(rightIntersection[0], lowerRightY))
									elif patch.vertices[vi][0] >= rightIntersection[0] and patch.vertices[vi][1] >= upperRightY:
										intersectionPoints.append(Point(rightIntersection[0], upperRightY))

				#print ("intersectionPoints", intersectionPoints)
				if len(intersectionPoints) == 4:
					intersect = Polygon(intersectionPoints[0], intersectionPoints[1], intersectionPoints[2], intersectionPoints[3])
				if len(intersectionPoints) == 3:
					intersect = Polygon(intersectionPoints[0], intersectionPoints[1], intersectionPoints[2])



			print "intersect", intersect
			if intersect is not None and type(intersect) == Polygon:
				_,regPoints = self.IRegConstraint(I, Vin, Vout, intersect)
				feasiblePoints += regPoints
		# Now construct convex hull with feasible points and add the constraint
		feasiblePointsNp = np.zeros((len(feasiblePoints),3))
		for fi in range(len(feasiblePointsNp)):
			for ii in range(3):
				feasiblePointsNp[fi][ii] = float(feasiblePoints[fi][ii])

		feasiblePointsNp = np.unique(feasiblePointsNp, axis=0)
		print ("feasiblePointsNp")
		print (feasiblePointsNp)
		print ("")
		hull = ConvexHull(feasiblePointsNp)
		convexHullMiddle = np.zeros((3))
		numPoints = 0
		for simplex in hull.simplices:
			for index in simplex:
				convexHullMiddle += feasiblePointsNp[index]
				numPoints += 1
		convexHullMiddle = convexHullMiddle/(numPoints*1.0)
		upVector = np.array([0,0,1])
		overallConstraint = ""
		for simplex in hull.simplices:
			pointsFromSimplex = [None]*3
			for ii in range(3):
				pointsFromSimplex[ii] = Point3D(feasiblePointsNp[simplex[ii]][0], feasiblePointsNp[simplex[ii]][1], feasiblePointsNp[simplex[ii]][2])
			#print ("pointsFromSimplex", pointsFromSimplex)
			planeFromSimplex = Plane(pointsFromSimplex[0], pointsFromSimplex[1],pointsFromSimplex[2])
			normal = np.array(list(planeFromSimplex.normal_vector))
			pointInPlane = np.array([planeFromSimplex.p1[0], planeFromSimplex.p1[1], planeFromSimplex.p1[2]])
			
			# Determine if the middle of the convex hull is above or below
			# the plane and add the constraint related to the plane accordingly
			dotSign = np.dot(upVector,(convexHullMiddle - pointInPlane))
			sign = " <= "
			if dotSign > 0:
				sign = " >= "

			overallConstraint += "1 " + I + " + " + str(normal[0]/normal[2]) + " " + Vin +\
				" + " + str(normal[1]/normal[2]) + " " + Vout + sign + str((normal[0]/normal[2])*pointInPlane[0] +\
					(normal[1]/normal[2])*pointInPlane[1] + pointInPlane[2]) + "\n"

		return overallConstraint


	# patch is a Polygon object indicating the hyperrectangle
	def IRegConstraint(self, I, Vin, Vout, patch):
		INum = [None]*len(patch.vertices)
		firDerIn = [None]*len(patch.vertices)
		firDerOut = [None]*len(patch.vertices)
		secDerIn = [None]*len(patch.vertices)
		secDerOut = [None]*len(patch.vertices)
		points = [None]*len(patch.vertices)

		for i in range(len(patch.vertices)):
			[INum[i], firDerIn[i], firDerOut[i], secDerIn[i], secDerOut[i]] = self.currentFun(patch.vertices[i][0], patch.vertices[i][1])
			points[i] = Point3D(patch.vertices[i][0], patch.vertices[i][1], INum[i])


		#print "secDerIn ", secDerIn
		#print "secDerOut ", secDerOut
		'''if not((secDerIn[1] >= 1 and secDerOut[1] >= 0 and secDerIn[2] >=0 and secDerOut[2] >= 0 and secDerIn[3] >= 0 and secDerOut[3] >= 0 \
			and secDerIn[0] >= 0 and secDerOut[0] >= 0) or (\
			secDerIn[1] <= 0 and secDerOut[1] <= 0 and secDerIn[2] <=0 and secDerOut[2] <= 0 and secDerIn[3] <= 0 and secDerOut[3] <= 0 \
			and secDerIn[0] <= 0 and secDerOut[0] <= 0)):
			return None'''

		overallConstraint = ""

		tangentSign = " >= "
		secantSign = " <= "

		# concave downward constraint - function should be less than equal to tangent plane and
		# greater than equal to secant planes
		if secDerIn[0] < 0:
			tangentSign = " <= "
			secantSign = " >= "

		# a list of point and normal
		feasiblePlanes = []
		# tangent constraints
		overallConstraint += "1 " + I + " + " + str(-firDerIn[0]) + " " + Vin +\
		" + " + str(-firDerOut[0]) + " " + Vout + tangentSign + str(-firDerIn[0]*patch.vertices[0][0] - firDerOut[0]*patch.vertices[0][1] + INum[0]) + "\n"
		feasiblePlanes.append([Plane(Point3D(patch.vertices[0][0], patch.vertices[0][1], INum[0]),normal_vector =(firDerIn[0], firDerOut[0], -1)), tangentSign])
		
		zeroRegion = Polygon(Point(self.Vtn,0.0), Point(1 + self.Vtp, 1 + self.Vtp - self.Vtn), Point(1 + self.Vtp, 1.0), Point(self.Vtn,self.Vtn -self.Vtp))
		patchVertsInsideZeroReg = True
		for vert in patch.vertices:
			if not(zeroRegion.encloses(vert)):
				onLine = False
				for side in zeroRegion.sides:
					if side.contains(vert):
						onLine = True
						break
				if not(onLine):
					patchVertsInsideZeroReg = False

		if patchVertsInsideZeroReg:
			print ("patch inside zero region")
			feasiblePoints = [points[0], points[1], points[2], points[3]]
			return overallConstraint, feasiblePoints


		for i in range(1,len(INum)):
			overallConstraint += "1 " + I + " + " + str(-firDerIn[i]) + " " + Vin +\
			" + " + str(-firDerOut[i]) + " " + Vout + tangentSign + str(-firDerIn[i]*patch.vertices[i][0] - firDerOut[i]*patch.vertices[i][1] + INum[i]) + "\n"
			feasiblePlanes.append([Plane(Point3D(patch.vertices[i][0], patch.vertices[i][1], INum[i]),normal_vector = (firDerIn[i], firDerOut[i], -1)), tangentSign])

		# secant constraints

		if len(points) == 3:
			feasiblePlanes.append([Plane(points[0], points[1], points[2]),secantSign])

		else:
			possibleSecantPlanes = []
			excludedPoints = []
			possibleSecantPlanes.append(Plane(points[0], points[1], points[2]))
			excludedPoints.append(points[3])

			possibleSecantPlanes.append(Plane(points[0], points[1], points[3]))
			excludedPoints.append(points[2])

			possibleSecantPlanes.append(Plane(points[0], points[2], points[3]))
			excludedPoints.append(points[1])

			possibleSecantPlanes.append(Plane(points[1], points[2], points[3]))
			excludedPoints.append(points[0])

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
					overallConstraint += "1 " + I + " + " + str(normal[0]/normal[2]) + " " + Vin +\
					" + " + str(normal[1]/normal[2]) + " " + Vout + secantSign + str((normal[0]/normal[2])*includedPt[0] +\
						(normal[1]/normal[2])*includedPt[1] + includedPt[2]) + "\n"
					numSecantConstraints += 1
					feasiblePlanes.append([plane, secantSign])

		'''print "numFeasiblePlanes", len(feasiblePlanes)
		for plane in feasiblePlanes:
			print "plane", plane
			print ""'''
		
		intersectionPoints = []
		# find intersection of all possible combinations of feasible planes
		# store points that are satisfied by all constraints
		for i in range(len(feasiblePlanes)):
			for j in range(i+1, len(feasiblePlanes)):
				for k in range(j+1, len(feasiblePlanes)):
					#print "onePlane", feasiblePlanes[i][0]
					#print "otherPlane", feasiblePlanes[j][0]
					#print "otherPlane[2]", feasiblePlanes[k][0]
					ps = [None]*3
					ps[0] = feasiblePlanes[i][0].p1
					ps[1] = feasiblePlanes[j][0].p1
					ps[2] = feasiblePlanes[k][0].p1
					norms = [None]*3
					norms[0] = feasiblePlanes[i][0].normal_vector
					norms[1] = feasiblePlanes[j][0].normal_vector
					norms[2] = feasiblePlanes[k][0].normal_vector
					AMat = np.zeros((3,3))
					BMat = np.zeros((3))
					for ii in range(3):
						d = 0.0
						for jj in range(3):
							AMat[ii][jj] = norms[ii][jj]
							d += norms[ii][jj]*ps[ii][jj]
						BMat[ii] = d

					if norms[0][0] == 0 and norms[1][0] == 0 and norms[2][0] == 0:
						AMat = np.zeros((2,2))
						BMat = np.zeros((2))
						for ii in range(2):
							AMat[ii][0] = norms[ii][1]
							AMat[ii][1] = norms[ii][2]
							BMat[ii] = norms[ii][1]*ps[ii][1] + norms[ii][2]*ps[ii][2]

					elif norms[0][1] == 0 and norms[1][1] == 0 and norms[2][1] == 0:
						AMat = np.zeros((2,2))
						BMat = np.zeros((2))
						for ii in range(2):
							AMat[ii][0] = norms[ii][0]
							AMat[ii][1] = norms[ii][2]
							BMat[ii] = norms[ii][0]*ps[ii][0] + norms[ii][2]*ps[ii][2]
		

					#print "AMat"
					#print AMat
					#print "BMat"
					#print BMat

					try:
						sol = np.linalg.solve(AMat,BMat)
						if len(sol) == 3:
							intersectingPoint = Point3D(sol[0],sol[1],sol[2])
							intersectionPoints.append(intersectingPoint)
						elif len(sol) == 2:
							if norms[0][0] == 0:
								intersectionPoints.append(Point3D(patch.bounds[0], sol[0], sol[1]))
								intersectionPoints.append(Point3D(patch.bounds[2], sol[0], sol[1]))
							elif norms[0][1] == 0:
								intersectionPoints.append(Point3D(sol[0], patch.bounds[1], sol[1]))
								intersectionPoints.append(Point3D(sol[0], patch.bounds[3], sol[1]))

						#print ("feasiblePoints", feasiblePoints)

						#intersectingLine = feasiblePlanes[i][0].intersection(feasiblePlanes[j][0])
						#print "intersectingLine", intersectingLine
						#intersectingPoint = feasiblePlanes[k][0].intersection(intersectingLine)
						#print "intersectingPoint", intersectingPoint
					except np.linalg.linalg.LinAlgError:
						pass
					#print ""

		#print ("intersectionPoints", intersectionPoints)
		print ("len(intersectionPoints)", len(intersectionPoints))
		feasiblePoints = []
		for point in intersectionPoints:
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

		#print ("feasiblePoints")
		#print (feasiblePoints)
		print ("len(feasiblePoints)", len(feasiblePoints))

		return overallConstraint, feasiblePoints



	def oscNum(self,V):
		lenV = len(V)
		Vin = [V[i % lenV] for i in range(-1,lenV-1)]
		Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
		IFwd = [self.currentFun(Vin[i], V[i])[0] for i in range(lenV)]
		ICc = [self.currentFun(Vcc[i], V[i])[0] for i in range(lenV)]
		return [IFwd, ICc, [(IFwd[i]*self.g_fwd + ICc[i]*self.g_cc) for i in range(lenV)]]

	def jacobian(self,V):
		lenV = len(V)
		Vin = [V[i % lenV] for i in range(-1,lenV-1)]
		Vcc = [V[(i + lenV/2) % lenV] for i in range(lenV)]
		jac = np.zeros((lenV, lenV))
		for i in range(lenV):
			[Ifwd, firDerInfwd, firDerOutfwd, secDerInfwd, secDerOutfwd] = self.currentFun(Vin[i], V[i])
			[Icc, firDerIncc, firDerOutcc, secDerIncc, secDerOutcc] = self.currentFun(Vcc[i], V[i])
			jac[i, (i-1)%lenV] = self.g_fwd*firDerInfwd
			jac[i, (i + lenV/2) % lenV] = self.g_cc*firDerIncc
			jac[i, i] = self.g_fwd*firDerOutfwd + self.g_cc*firDerOutcc
		return jac

	# numerical approximation where i sample jacobian in the patch by bounds
	# might need a more analytical way of solving this later
	def jacobianInterval(self,bounds):
		lowerBound = bounds[:,0]
		upperBound = bounds[:,1]
		lenV = len(lowerBound)
		jac = np.zeros((lenV, lenV,2))
		for i in range(lenV):
			lowOut = lowerBound[i]
			highOut = upperBound[i]
			rangeOut = highOut - lowOut
			outArray = np.arange(lowOut, highOut, rangeOut/100.0)

			minFirDerInFwd, maxFirDerInFwd = float("inf"), -float("inf")
			minFirDerOutFwd, maxFirDerOutFwd = float("inf"), -float("inf")
			
			lowFwd = lowerBound[(i-1)%lenV]
			highFwd = upperBound[(i-1)%lenV]
			rangeFwd = highFwd - lowFwd
			fwdArray = np.arange(lowFwd, highFwd, rangeFwd/100.0)
			for fwdIn in range(len(fwdArray)):
				for outIn in range(len(outArray)):
					[Ifwd, firDerInfwd, firDerOutfwd, secDerInfwd, secDerOutfwd] = self.currentFun(fwdArray[fwdIn], outArray[outIn])
					minFirDerInFwd = min(minFirDerInFwd, firDerInfwd)
					maxFirDerInFwd = max(maxFirDerInFwd, firDerInfwd)
					minFirDerOutFwd = min(minFirDerOutFwd, firDerOutfwd)
					maxFirDerOutFwd = max(maxFirDerOutFwd, firDerOutfwd)

			minFirDerInCc, maxFirDerInCc = float("inf"), -float("inf")
			minFirDerOutCc, maxFirDerOutCc = float("inf"), -float("inf")
			
			lowCc = lowerBound[(i-1)%lenV]
			highCc = upperBound[(i-1)%lenV]
			rangeCc = highCc - lowCc
			ccArray = np.arange(lowCc, highCc, rangeCc/100.0)
			for ccIn in range(len(ccArray)):
				for outIn in range(len(outArray)):
					[Icc, firDerIncc, firDerOutcc, secDerIncc, secDerOutcc] = self.currentFun(ccArray[ccIn], outArray[outIn])
					minFirDerInCc = min(minFirDerInCc, firDerIncc)
					maxFirDerInCc = max(maxFirDerInCc, firDerIncc)
					minFirDerOutCc = min(minFirDerOutCc, firDerOutcc)
					maxFirDerOutCc = max(maxFirDerOutCc, firDerOutcc)

			jac[i, (i-1)%lenV, 0] = self.g_fwd*minFirDerInFwd
			jac[i, (i-1)%lenV, 1] = self.g_fwd*maxFirDerInFwd
			jac[i, (i + lenV/2) % lenV, 0] = self.g_cc*minFirDerInCc
			jac[i, (i + lenV/2) % lenV, 1] = self.g_cc*maxFirDerInCc
			jac[i, i, 0] = self.g_fwd*minFirDerOutFwd + self.g_cc*minFirDerOutCc
			jac[i, i, 1] = self.g_fwd*maxFirDerOutFwd + self.g_cc*maxFirDerOutCc

		return jac

	def linearConstraints(self, hyperRectangle):
		solvers.options["show_progress"] = False
		allConstraints = ""
		lenV = self.numStages*2
		allConstraints = ""
		for i in range(lenV):
			fwdInd = (i-1)%lenV
			ccInd = (i+lenV/2)%lenV
			
			fwdP0, fwdP1, fwdP2, fwdP3 = map(Point, [(hyperRectangle[fwdInd][0], hyperRectangle[i][0]),\
			(hyperRectangle[fwdInd][1], hyperRectangle[i][0]), (hyperRectangle[fwdInd][1], hyperRectangle[i][1]),\
			(hyperRectangle[fwdInd][0], hyperRectangle[i][1])])
			fwdPatch = Polygon(fwdP0, fwdP1, fwdP2, fwdP3)

			ccP0, ccP1, ccP2, ccP3 = map(Point, [(hyperRectangle[ccInd][0], hyperRectangle[i][0]),\
			(hyperRectangle[ccInd][1], hyperRectangle[i][0]), (hyperRectangle[ccInd][1], hyperRectangle[i][1]),\
			(hyperRectangle[ccInd][0], hyperRectangle[i][1])])
			ccPatch = Polygon(ccP0, ccP1, ccP2, ccP3)
			
			fwdConstraints = self.ICrossRegConstraint(self.IsFwd[i], self.xs[fwdInd], self.xs[i], fwdPatch)
			ccConstraints = self.ICrossRegConstraint(self.IsCc[i], self.xs[ccInd], self.xs[i], ccPatch)
			allConstraints = fwdConstraints + ccConstraints
			allConstraints += str(self.g_fwd) + " " + self.IsFwd[i] + " + " + str(self.g_cc) + " " + self.IsCc[i] + " >= 0.0\n"
			allConstraints += str(self.g_fwd) + " " + self.IsFwd[i] + " + " + str(self.g_cc) + " " + self.IsCc[i] + " <= 0.0\n"

		'''allConstraintList = allConstraints.splitlines()
		allConstraints = ""
		for i in range(len(allConstraintList)):
			allConstraints += allConstraintList[i] + "\n"
		print "numConstraints ", len(allConstraintList)
		print "allConstraints"
		print allConstraints'''
		variableDict, A, B = lpUtils.constructCoeffMatrices(allConstraints)
		newHyperRectangle = np.copy(hyperRectangle)
		feasible = True
		for i in range(lenV):
			#print "min max ", i
			minObjConstraint = "min 1 " + self.xs[i]
			maxObjConstraint = "max 1 " + self.xs[i]
			Cmin = lpUtils.constructObjMatrix(minObjConstraint,variableDict)
			Cmax = lpUtils.constructObjMatrix(maxObjConstraint,variableDict)
			minSol = solvers.lp(Cmin,A,B)
			maxSol = solvers.lp(Cmax,A,B)
			if (minSol["status"] == "primal infeasible" or minSol["status"] == "dual infeasible")  and (maxSol["status"] == "primal infeasible" or maxSol["status"] == "dual infeasible"):
				feasible = False
				break
			else:
				if minSol["status"] == "optimal":
					newHyperRectangle[i,0] = minSol['x'][variableDict[self.xs[i]]] - 1e-6
				if maxSol["status"] == "optimal":
					newHyperRectangle[i,1] = maxSol['x'][variableDict[self.xs[i]]] + 1e-6

		return [feasible, newHyperRectangle]






