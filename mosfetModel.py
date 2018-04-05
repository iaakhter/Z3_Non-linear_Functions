import numpy as np
import lpUtils
from cvxopt import matrix,solvers
from scipy.spatial import ConvexHull
from osgeo import ogr

class MosfetModel:
	# modelParam = [Vtp, Vtn, Vdd, Kn, Sn]
	def __init__(self, modelParam, g_cc, g_fwd, numStages):
		# gradient of tanh -- y = tanh(modelParam*x)
		self.Vtp = modelParam[0]
		self.Vtn = modelParam[1]
		self.Vdd = modelParam[2]
		self.Kn = modelParam[3]
		self.Sn = modelParam[4]
		#self.Kp = -self.Kn/2.0
		self.Kp = -self.Kn/3.0
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

		self.constructPolygonRegions()

	def constructPolygonRegions(self):
		self.secDerSigns = [None]*7
		rings = [None]*7
		self.polygonRegs = [None]*7
		Is = [None]*7
		secDerIns = [None]*7
		secDerOuts = [None]*7
		secDerInOuts = [None]*7
		regPts = [[(0.0, 0.0),(self.Vtn, 0.0), (self.Vtn, self.Vtn - self.Vtp), (0.0, -self.Vtp)],
					[(0.0, -self.Vtp), (self.Vtn, self.Vtn - self.Vtp), (self.Vtn, self.Vdd), (0.0,self.Vdd)],
					[(self.Vtn,0.0), (self.Vdd + self.Vtp, 0.0), (self.Vdd + self.Vtp, self.Vdd + self.Vtp - self.Vtn)],
					[(self.Vtn,0.0), (self.Vdd + self.Vtp, self.Vdd + self.Vtp - self.Vtn), (self.Vdd + self.Vtp, self.Vdd), (self.Vtn,self.Vtn -self.Vtp)],
					[(self.Vtn,self.Vtn -self.Vtp), (self.Vdd + self.Vtp, self.Vdd), (self.Vtn, self.Vdd)],
					[(self.Vdd + self.Vtp,0.0), (self.Vdd, 0.0), (self.Vdd, self.Vdd - self.Vtn), (self.Vdd + self.Vtp, self.Vdd + self.Vtp - self.Vtn)],
					[(self.Vdd + self.Vtp,self.Vdd + self.Vtp - self.Vtn), (self.Vdd, self.Vdd - self.Vtn), (self.Vdd, self.Vdd), (self.Vdd + self.Vtp, self.Vdd)]]

		for pi in range(len(regPts)):
			#print ("region number", pi)
			pts = regPts[pi]
			rings[pi] = ogr.Geometry(ogr.wkbLinearRing)
			for pp in range(len(pts)+1):
				x = pts[pp%len(pts)][0]
				y = pts[pp%len(pts)][1]
				#print ("point", x, y)
				rings[pi].AddPoint(x,y)
			self.polygonRegs[pi] = ogr.Geometry(ogr.wkbPolygon)
			self.polygonRegs[pi].AddGeometry(rings[pi])

			[Is[pi], firDerIn, firDerOut, secDerIns[pi], secDerOuts[pi], secDerInOuts[pi]] = self.currentFun(pts[0][0], pts[0][1], pi)
			if secDerIns[pi] == 0 and secDerOuts[pi] == 0:
				self.secDerSigns[pi] = "zer"

			elif secDerInOuts[pi] == 0 and secDerIns[pi] >= 0 and secDerOuts[pi] >= 0:
				self.secDerSigns[pi] = "pos"

			elif secDerInOuts[pi] == 0 and secDerIns[pi] <= 0 and secDerOuts[pi] <= 0:
				self.secDerSigns[pi] = "neg"

			else:
				self.secDerSigns[pi] = "sad"
			#print ("sign ", self.secDerSigns[pi])

	def currentFun(self, Vin, Vout, regionNumber = None):
		In = 0.0
		firDerInn, firDerOutn = 0.0, 0.0
		secDerInn, secDerOutn = 0.0, 0.0
		secDerInOutn = 0.0
		eqnN, eqnP = None, None
		if regionNumber == 0:
			eqnN, eqnP  = 1, 5
		if regionNumber == 1:
			eqnN, eqnP = 1, 6
		if regionNumber == 2:
			eqnN, eqnP = 3, 5
		if regionNumber == 3:
			eqnN, eqnP = 2, 5
		if regionNumber == 4:
			eqnN, eqnP = 2, 6
		if regionNumber == 5:
			eqnN, eqnP = 3, 4
		if regionNumber == 6:
			eqnN, eqnP = 2, 4
		
		if Vin <= self.Vtn and (eqnN == None or eqnN == 1):
			In = 0.0
			firDerInn = 0.0
			firDerOutn = 0.0
			secDerInn = 0.0
			secDerOutn = 0.0
			secDerInOutn = 0.0
		elif self.Vtn <= Vin and Vin <=Vout + self.Vtn and (eqnN == None or eqnN == 2):
			In = self.Sn*(self.Kn/2.0)*(Vin - self.Vtn)*(Vin - self.Vtn)
			firDerInn = self.Sn*self.Kn*(Vin - self.Vtn)
			firDerOutn = 0.0
			secDerInn = self.Sn*self.Kn
			secDerOutn = 0.0
			secDerInOutn = 0.0
		elif  Vin >= Vout + self.Vtn and (eqnN == None or eqnN == 3):
			In = self.Sn*(self.Kn)*(Vin - self.Vtn - Vout/2.0)*Vout;
			firDerInn = self.Sn*self.Kn*Vout
			firDerOutn = self.Sn*self.Kn*(Vin - self.Vtn - Vout)
			secDerInn = 0.0
			secDerOutn = -self.Sn*self.Kn
			secDerInOutn = self.Sn*self.Kn

		Ip = 0.0
		firDerInp, firDerOutp = 0.0, 0.0
		secDerInp, secDerOutp = 0.0, 0.0
		secDerInOutp = 0.0
		if Vin - self.Vtp >= self.Vdd and (eqnP == None or eqnP == 4):
			Ip = 0.0
			firDerInp = 0.0
			firDerOutp = 0.0
			secDerInp = 0.0
			secDerOutp = 0.0
			secDerInOutp = 0.0
		elif Vout <= Vin - self.Vtp and Vin - self.Vtp <= self.Vdd and (eqnP == None or eqnP == 5):
			Ip = self.Sp*(self.Kp/2.0)*(Vin - self.Vtp - self.Vdd)*(Vin - self.Vtp - self.Vdd)
			firDerInp = self.Sp*self.Kp*(Vin - self.Vtp - self.Vdd)
			firDerOutp = 0.0
			secDerInp = self.Sp*self.Kp
			secDerOutp = 0.0
			secDerInOutp = 0.0
		elif Vin - self.Vtp <= Vout and (eqnP == None or eqnP == 6):
			Ip = self.Sp*self.Kp*((Vin - self.Vtp - self.Vdd) - (Vout - self.Vdd)/2.0)*(Vout - self.Vdd)
			firDerInp = self.Sp*self.Kp*(Vout - self.Vdd)
			firDerOutp = self.Sp*self.Kp*((Vin - self.Vtp - self.Vdd) - (Vout - self.Vdd))
			secDerInp = 0.0
			secDerOutp = -self.Sp*self.Kp
			secDerInOutp = self.Sp*self.Kp

		I = -(In + Ip)
		firDerIn = -(firDerInn + firDerInp)
		firDerOut = -(firDerOutn + firDerOutp)

		secDerIn = -(secDerInn + secDerInp)
		secDerOut = -(secDerOutn + secDerOutp)
		secDerInOut = -(secDerInOutn + secDerInOutp)
		return [I, firDerIn, firDerOut, secDerIn, secDerOut, secDerInOut]

	def intersectSurfPlaneFunDer(self, Vin, Vout, plane, regionNumber):
		if regionNumber == 0 or regionNumber == 3 or regionNumber == 6:
			print ("invalid region number")
			return
		planePt = plane[0,:]
		planeNorm = plane[1,:]
		m, d = None, None
		if planeNorm[1] != 0:
			m = -planeNorm[0]
			d = planeNorm[0]*planePt[0] + planeNorm[1]*planePt[1] + planeNorm[2]*planePt[2]
		else:
			d = planePt[0]

		I = 0.0
		firDers = np.zeros((2))
		derTypes = [False, False]
		if regionNumber == 1:
			if m is None:
				I += (-self.Sp*self.Kp*(d - self.Vtp - self.Vdd)*(Vout - self.Vdd) + 
					0.5*self.Sp*self.Kp*(Vout - self.Vdd)*(Vout - self.Vdd))

				firDers[1] += (-self.Sp*self.Kp*(d - self.Vtp - Vout))
				derTypes[1] = True
			else:
				I += (-self.Sp*self.Kp*(Vin - self.Vtp - self.Vdd)*(m*Vin + d - self.Vdd) + 
					0.5*self.Sp*self.Kp*(m*Vin + d - self.Vdd)*(m*Vin + d - self.Vdd))

				firDers[0] += (-self.Sp*self.Kp*(m*(Vin - self.Vtp - self.Vdd) + (1-m)*(m*Vin + d - self.Vdd)))
				derTypes[0] = True

		elif regionNumber == 2:
			if m is None:
				I += (-self.Sn*self.Kn*d*Vout + self.Sn*self.Kn*self.Vtn*Vout + 
					0.5*self.Sn*self.Kn*Vout*Vout - 0.5*self.Sp*self.Kp*(d - self.Vtp - self.Vdd)*(d - self.Vtp - self.Vdd))

				firDers[1] += (-self.Sn*self.Kn*(d - self.Vtn - Vout))
				derTypes[1] = True
			else:
				I += (-self.Sn*self.Kn*Vin*(m*Vin + d) + self.Sn*self.Kn*self.Vtn*(m*Vin + d) + 
					0.5*self.Sn*self.Kn*(m*Vin + d)*(m*Vin + d) - 0.5*self.Sp*self.Kp*(Vin - self.Vtp - self.Vdd)*(Vin - self.Vtp - self.Vdd))

				firDers[0] += (-self.Sn*self.Kn*(2*m*Vin + d - m - m*m*Vin - m*d) - 
					self.Sp*self.Kp*(Vin - self.Vtp - self.Vdd))
				derTypes[0] = True

		elif regionNumber == 4:
			if m is None:
				I += (-0.5*self.Sn*self.Kn*(d - self.Vtn)*(d - self.Vtn) - 
					self.Sp*self.Kp*(d - self.Vtp - self.Vdd)*(Vout - self.Vdd) + 
					0.5*self.Sp*self.Kp*(Vout - self.Vdd)*(Vout - self.Vdd))

				firDers[1] += (-self.Sp*self.Kp*(d - self.Vtp - Vout))
				derTypes[1] = True
			else:
				I += (-0.5*self.Sn*self.Kn*(Vin - self.Vtn)*(Vin - self.Vtn) - 
					self.Sp*self.Kp*(Vin - self.Vtp - self.Vdd)*(m*Vin + d - self.Vdd) +
					0.5*self.Sp*self.Kp*(m*Vin + d - self.Vdd)*(m*Vin + d - self.Vdd))

				firDers[0] += (-self.Sn*self.Kn*(Vin - self.Vtn) - 
					self.Sp*self.Kp*(m*Vin + d - self.Vdd))
				derTypes[0] = True

		elif regionNumber == 5:
			if m is None:
				I += (-self.Sn*self.Kn*d*Vout + self.Sn*self.Kn*self.Vtn*Vout + 0.5*self.Sn*self.Kn*Vout*Vout)

				firDers[1] += (-self.Sn*self.Kn*(d - self.Vtn - Vout))
				derTypes[1] = True
			else:
				I += (-self.Sn*self.Kn*Vin*(m*Vin + d) + self.Sn*self.Kn*self.Vtn*(m*Vin + d) + 
					0.5*self.Sn*self.Kn*(m*Vin + d)*(m*Vin + d))

				firDers[0] += (-self.Sn*self.Kn*(2*m - self.Vtn*m - m*(m*Vin + d)))
				derTypes[0] = True

		return [I, firDers, derTypes]

	def convexHullConstraints(self, feasiblePoints, I, Vin, Vout):
		hull = ConvexHull(feasiblePoints, qhull_options='QbB')
		convexHullMiddle = np.zeros((3))
		numPoints = 0
		for simplex in hull.simplices:
			for index in simplex:
				convexHullMiddle += feasiblePoints[index,:]
				numPoints += 1
		convexHullMiddle = convexHullMiddle/(numPoints*1.0)

		overallConstraint = ""
		for si in range(len(hull.simplices)):
			simplex = hull.simplices[si]
			#print ("simplex", simplex)
			pointsFromSimplex = np.zeros((3,3))
			for ii in range(3):
				pointsFromSimplex[ii] = feasiblePoints[simplex[ii]]
			
			#print ("pointsFromSimplex", pointsFromSimplex)
			normal = np.cross(pointsFromSimplex[1] - pointsFromSimplex[0], pointsFromSimplex[2] - pointsFromSimplex[0])
			'''if normal[2] < 0:
				normal = -normal'''
			pointInPlane = pointsFromSimplex[0]
			#print ("pointsFromSimplex", pointsFromSimplex)
			d = normal[0]*pointInPlane[0] + normal[1]*pointInPlane[1] + normal[2]*pointInPlane[2]
			middleD = normal[0]*convexHullMiddle[0] + normal[1]*convexHullMiddle[1] + normal[2]*convexHullMiddle[2]
			# Determine if the middle of the convex hull is above or below
			# the plane and add the constraint related to the plane accordingly
			sign = " <= "

			if middleD > d:
				sign = " >= "

			'''print ("normal", normal)
			print ("pointInPlane", pointInPlane)
			print ("sign", sign)
			print ("")'''
			
			#if np.greater_equal(np.absolute(normal),np.ones(normal.shape)*1e-5).any():
			overallConstraint += str(normal[2])+" " + I + " + " + str(normal[0]) + " " + Vin +\
				" + " + str(normal[1]) + " " + Vout + sign + str(d) + "\n"
		return overallConstraint


	# patch is a polygon
	def ICrossRegConstraint(self,I, Vin, Vout, patch):
		#print ("crossRegionPatch", patch)
		patchRing = ogr.Geometry(ogr.wkbLinearRing)
		for i in range(patch.shape[0] + 1):
			patchRing.AddPoint(patch[i%patch.shape[0],0], patch[i%patch.shape[0],1])
		patchPolygon = ogr.Geometry(ogr.wkbPolygon)
		patchPolygon.AddGeometry(patchRing)
		
		feasiblePoints = None
		numIntersections = 0
		regConstraints = None
		for i in range(len(self.polygonRegs)):
			'''if i!=5:
				continue'''
			polygon = self.polygonRegs[i]
			intersectPoly = polygon.Intersection(patchPolygon)
			intersectPolyRing = intersectPoly.GetGeometryRef(0)
			if intersectPolyRing is not None:
				intersectingPoints = []
				for pi in range(intersectPolyRing.GetPointCount()-1):
					intersectingPoints.append((intersectPolyRing.GetPoint(pi)[0], intersectPolyRing.GetPoint(pi)[1]))
				intersect = np.array(intersectingPoints)
				regConstraints,regPoints = self.IRegConstraint(I, Vin, Vout, intersect,i)
				'''if i == 1:
					print ("regConstraints", regConstraints)
					print ("regPoints", regPoints)'''
				if feasiblePoints is None:
					if len(regPoints) >= 1:
						feasiblePoints = regPoints
				else:
					if len(regPoints) >= 1:
						#print ("feasiblePoints", feasiblePoints)
						feasiblePoints = np.vstack([feasiblePoints,regPoints])
				numIntersections += 1
		if numIntersections == 1:
			return regConstraints
		# Now construct convex hull with feasible points and add the constraint
		#feasiblePoints = np.array(feasiblePoints)
		#print ("feasiblePoints non-unique before")
		#print (feasiblePoints)
		if feasiblePoints is None or len(feasiblePoints) == 0:
			return ""
		feasiblePoints = np.unique(feasiblePoints, axis=0)
		#print ("crossRegion Feasible Points")
		#print (feasiblePoints)
		#print ("")		
		overallConstraint = self.convexHullConstraints(feasiblePoints, I, Vin, Vout)
		#print ("overallConstraint")
		#print (overallConstraint)
		return overallConstraint

	def saddleConvexHull(self, boundaryPlanes, boundaryPts, polygonNumber):
		feasiblePoints = []
		for pi in range(len(boundaryPlanes)):
			plane = boundaryPlanes[pi][0]
			point1 = boundaryPts[pi][0]
			point2 = boundaryPts[pi][1]
			funValue1, firDers1, derTypes1 = self.intersectSurfPlaneFunDer(point1[0], point1[1], plane, polygonNumber)
			funValue2, firDers2, derTypes2 = self.intersectSurfPlaneFunDer(point2[0], point2[1], plane, polygonNumber)
			planeNormal = plane[1,:]
			planePt = plane[0,:]
			d = planePt[0]*planeNormal[0] + planePt[1]*planeNormal[1] + planePt[2]*planeNormal[2]
			feasiblePoints.append(point1)
			feasiblePoints.append(point2)
			#print ("point1", point1, "point2", point2)
			#print ("planeNormal", planeNormal)
			if not(derTypes1[0]) and not(derTypes2[0]):
				m1, m2 = firDers1[1], firDers2[1]
				c1 = point1[2] - m1*point1[1]
				c2 = point2[2] - m2*point2[1]
				intersectingPt = None
				if abs(m1 - m2) < 1e-14:
					xPt = (point1[1] + point2[1])/2.0
					yPt = (point1[2] + point2[2])/2.0
					intersectingPt = np.array([xPt, yPt])
				else:
					xPt = (c2 - c1)/(m1 - m2)
					yPt = m1*xPt + c1
					intersectingPt = np.array([xPt, yPt])
				# TODO: check if this makes sense
				missingCoord = None
				if planeNormal[1] == 0:
					missingCoord = point1[0]
				else:
					missingCoord = (intersectingPt[0] - d)/planeNormal[0]
				feasiblePoints.append([missingCoord,intersectingPt[0],intersectingPt[1]])
				#print ("feasiblePt added if", feasiblePoints[-1])

			elif not(derTypes1[1]) and not(derTypes2[1]):
				m1, m2 = firDers1[0], firDers2[0]
				c1 = point1[2] - m1*point1[0]
				c2 = point2[2] - m2*point2[0]
				intersectingPt = None
				if abs(m1 - m2) < 1e-14:
					xPt = (point1[0] + point2[0])/2.0
					yPt = (point1[2] + point2[2])/2.0
					intersectingPt = np.array([xPt, yPt])
				else:
					xPt = (c2 - c1)/(m1 - m2)
					yPt = m1*xPt + c1
					intersectingPt = np.array([xPt, yPt])
				# TODO: check if this makes sense
				missingCoord = planeNormal[0]*intersectingPt[0] + d
				feasiblePoints.append([intersectingPt[0],missingCoord,intersectingPt[1]])
				#print ("feasiblePt added else", feasiblePoints[-1])
		return feasiblePoints


	# patch is a polygon with a list of vertices
	def IRegConstraint(self, I, Vin, Vout, patch, polygonNumber):
		#print ("regionPatch", patch, patch.shape)
		minBounds = np.amin(patch,0)
		maxBounds = np.amax(patch,0)
		#print ("minBounds", minBounds)
		#print ("maxBounds", maxBounds)
		INum = [None]*patch.shape[0]
		firDerIn = [None]*patch.shape[0]
		firDerOut = [None]*patch.shape[0]
		secDerIn = [None]*patch.shape[0]
		secDerOut = [None]*patch.shape[0]
		points = np.zeros((patch.shape[0],3))

		for i in range(patch.shape[0]):
			[INum[i], firDerIn[i], firDerOut[i], secDerIn[i], secDerOut[i], secDerInOut] = self.currentFun(patch[i,0], patch[i,1],polygonNumber)
			points[i,:] = [patch[i,0],patch[i,1],INum[i]]

		overallConstraint = ""

		tangentSign = " >= "
		secantSign = " <= "

		#print ("secDerIn", secDerIn)
		#print ("secDerOut", secDerOut)

		patchRing = ogr.Geometry(ogr.wkbLinearRing)
		for i in range(patch.shape[0]+1):
			patchRing.AddPoint(patch[i%patch.shape[0],0], patch[i%patch.shape[0],1])
		patchPolygon = ogr.Geometry(ogr.wkbPolygon)
		patchPolygon.AddGeometry(patchRing)
		#print ("patchPolygon", patchPolygon)

		patchVertsInsideZeroReg = False
		if self.secDerSigns[polygonNumber] == "zer":
			patchVertsInsideZeroReg = True
		elif self.secDerSigns[polygonNumber] == "neg":
			tangentSign = " <= "
			secantSign = " >= "

		#print ("firDerIn", firDerIn, "firDerOut", firDerOut)
		#print ("secDerIn", secDerIn, "secDerOut", secDerOut)
		
		# a list of point and normal
		feasiblePlanes = []
		# tangent constraints
		for i in range(len(INum)):
			overallConstraint += "1 " + I + " + " + str(-firDerIn[i]) + " " + Vin +\
			" + " + str(-firDerOut[i]) + " " + Vout + tangentSign + str(-firDerIn[i]*patch[i,0] - firDerOut[i]*patch[i,1] + INum[i]) + "\n"
			feasiblePlanes.append([np.array([[points[i][0],points[i][1],points[i][2]],[-firDerIn[i],-firDerOut[i],1]]), tangentSign])
			if patchVertsInsideZeroReg:
				return overallConstraint, points

		# hyperrectangle constraints
		boundaryPlanes = []
		boundaryPts = []
		midPoint = np.sum(points, axis = 0)/(points.shape[0]*1.0)
		for i in range(points.shape[0]):
			point1 = points[i,:]
			point2 = points[(i+1)%points.shape[0],:]
			m = None, None
			norms = np.zeros((3))
			if point2[0] - point1[0] == 0:
				m = float("inf")
				norms[0] = 1
			else:
				m = (point2[1] - point1[1])/(point2[0] - point1[0])
				norms[0] = -m
				norms[1] = 1
			cSign = " <= "
			d = norms[0]*point1[0] + norms[1]*point1[1] + norms[2]*point1[2]
			dMid = norms[0]*midPoint[0] + norms[1]*midPoint[1]
			if dMid > d:
				cSign = " >= "
			feasiblePlanes.append([np.array([[point1[0],point1[1],point1[2]],[norms[0],norms[1],norms[2]]]), cSign])
			boundaryPlanes.append([np.array([[point1[0],point1[1],point1[2]],[norms[0],norms[1],norms[2]]]), cSign])
			boundaryPts.append([point1, point2])

		# secant constraints
		for i in range(points.shape[0]):
			for j in range(i+1, points.shape[0]):
				for k in range(j+1, points.shape[0]):
					normal = np.cross(points[k] - points[i], points[j] - points[i])
					if normal[2] < 0:
						normal = -normal
					includedPt = points[i]
					d = normal[0]*includedPt[0] + normal[1]*includedPt[1] + normal[2]*includedPt[2]
					planeFeasible = True
					for pi in range(points.shape[0]):
						excludedPt = points[pi]
						dExcluded = normal[0]*excludedPt[0] + normal[1]*excludedPt[1] + normal[2]*excludedPt[2]
						#print ("pi", pi, points[pi])
						#print ("dExcluded", dExcluded, "d", d, secantSign)
						feasible = dExcluded <= d
						if secantSign == " >= ":
							feasible = dExcluded >= d
						if abs(d - dExcluded) > 1e-14:
							if not(feasible):
								planeFeasible = False
								break
					if planeFeasible or points.shape[0] == 3:
						overallConstraint += str(normal[2])+ " " + I + " + " + str(normal[0]) + " " + Vin +\
							" + " + str(normal[1]) + " " + Vout + secantSign + str(d) + "\n"
						feasiblePlanes.append([np.array([[includedPt[0],includedPt[1],includedPt[2]],[normal[0],normal[1],normal[2]]]), secantSign])

		'''print ("numFeasiblePlanes", len(feasiblePlanes))
		for plane in feasiblePlanes:
			print ("plane", plane)
		print ("")'''
		feasiblePoints = []

		if self.secDerSigns[polygonNumber] != "sad":
			# find intersection of all possible combinations of feasible planes
			# store points that are satisfied by all constraints
			intersectionPoints = []
			for i in range(len(feasiblePlanes)):
				for j in range(i+1, len(feasiblePlanes)):
					for k in range(j+1, len(feasiblePlanes)):
						#print ("i ", i, "j", j, "k", k)
						ps = [None]*3
						ps[0] = feasiblePlanes[i][0][0,:]
						ps[1] = feasiblePlanes[j][0][0,:]
						ps[2] = feasiblePlanes[k][0][0,:]
						norms = [None]*3
						norms[0] = feasiblePlanes[i][0][1,:]
						norms[1] = feasiblePlanes[j][0][1,:]
						norms[2] = feasiblePlanes[k][0][1,:]
						AMat = np.zeros((3,3))
						BMat = np.zeros((3))
						for ii in range(3):
							d = 0.0
							for jj in range(3):
								AMat[ii][jj] = norms[ii][jj]
								d += norms[ii][jj]*ps[ii][jj]
							BMat[ii] = d
						#print ("AMat.shape", AMat.shape)
						#print ("condition number of AMat", np.linalg.cond(AMat))
						if np.linalg.cond(AMat) < 1e+10:
						#try:
							#print ("i ", i, "j", j, "k", k)
							#print ("conditionNumber of AMat", np.linalg.cond(AMat))
							sol = np.linalg.solve(AMat,BMat)
							#print ("sol", sol)
			
							#if sol[0] >= 0.0 and sol[0] <= 1.0 and sol[1] >= 0.0 and sol[1] <= 1.0:
							intersectionPoints.append(np.array([sol[0], sol[1], sol[2]]))
							#print ("intersectionPoint added", sol)
						

			#print ("intersectionPoints", intersectionPoints)
			#print ("len(intersectionPoints)", len(intersectionPoints))
			for point in intersectionPoints:
				#print ("intersectionPoint", point)
				pointFeasible = True
				for i in range(len(feasiblePlanes)):
					#if planeTypes[i] == "bound":
					#	continue
					planeSign = feasiblePlanes[i]
					plane = planeSign[0]
					sign = planeSign[1]
					normal = plane[1,:]
					planePoint = plane[0,:]

					d = normal[0]*planePoint[0] + normal[1]*planePoint[1] + normal[2]*planePoint[2]
					dPt = normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2]
					#IAtPt = (d - normal[0]*point[0] - normal[1]*point[1])/normal[2]
					#print ("sign", sign)
					#print ("point[2]", point[2], "IAtPt", IAtPt)
					if abs(d - dPt) > 1e-14:
						if sign == " <= ":
							if dPt > d:
								#print ("dPt", dPt, "d",d)
								pointFeasible = False
								'''print ("intersectionPointNotFeasible", point)
								print ("normal", normal)
								print ("planePoint", planePoint)
								print ("sign ", sign)'''
								break

						if sign == " >= ":
							if dPt < d:
								#print ("dPt", dPt, "d",d)
								pointFeasible = False
								'''print ("intersectionPointNotFeasible", point)
								print ("normal", normal)
								print ("planePoint", planePoint)
								print ("sign ", sign)'''
								break
				if pointFeasible:
					#print ("feasible", point)
					feasiblePoints.append([point[0], point[1], point[2]])
			feasiblePoints = np.array(feasiblePoints)
			if len(feasiblePoints) >= 1:
				feasiblePoints = np.unique(feasiblePoints, axis = 0)
		else:
			feasiblePoints = self.saddleConvexHull(boundaryPlanes, boundaryPts, polygonNumber)
			feasiblePoints = np.array(feasiblePoints)
			if len(feasiblePoints) >= 1:
				feasiblePoints = np.unique(feasiblePoints, axis = 0)

			#print ("weird feasiblePoints", feasiblePoints)
			overallConstraint = self.convexHullConstraints(feasiblePoints, I, Vin, Vout)
		
		#print ("regionFeasiblePoints")
		#print (feasiblePoints)
		#print ("len(feasiblePoints)", len(feasiblePoints))

		return overallConstraint, feasiblePoints



	def oscNum(self,V):
		lenV = len(V)
		Vin = [V[i % lenV] for i in range(-1,lenV-1)]
		Vcc = [V[(i + lenV//2) % lenV] for i in range(lenV)]
		IFwd = [self.currentFun(Vin[i], V[i])[0] for i in range(lenV)]
		ICc = [self.currentFun(Vcc[i], V[i])[0] for i in range(lenV)]
		return [IFwd, ICc, [(IFwd[i]*self.g_fwd + ICc[i]*self.g_cc) for i in range(lenV)]]

	def jacobian(self,V):
		#print ("Calculating jacobian")
		lenV = len(V)
		Vin = [V[i % lenV] for i in range(-1,lenV-1)]
		Vcc = [V[(i + lenV//2) % lenV] for i in range(lenV)]
		jac = np.zeros((lenV, lenV))
		for i in range(lenV):
			#print ("Vin[i]", Vin[i], "Vcc[i]", Vcc[i], "V[i]", V[i])
			[Ifwd, firDerInfwd, firDerOutfwd, secDerInfwd, secDerOutfwd, secDerInOutfwd] = self.currentFun(Vin[i], V[i])
			[Icc, firDerIncc, firDerOutcc, secDerIncc, secDerOutcc, secDerInOutCc] = self.currentFun(Vcc[i], V[i])
			#print ("firDerInfwd", firDerInfwd, "firDerIncc", firDerIncc)
			#print ("firDerOutfwd", firDerOutfwd, "firDerOutcc", firDerOutcc)
			jac[i, (i-1)%lenV] = self.g_fwd*firDerInfwd
			jac[i, (i + lenV//2) % lenV] = self.g_cc*firDerIncc
			jac[i, i] = self.g_fwd*firDerOutfwd + self.g_cc*firDerOutcc
		#print ("jacobian")
		#print (jac)
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

			lowFwd = lowerBound[(i-1)%lenV]
			highFwd = upperBound[(i-1)%lenV]

			firDerInFwd = np.zeros((4))
			firDerOutFwd = np.zeros((4))

			[Ifwd, firDerInFwd[0], firDerOutFwd[0], secDerInfwd, secDerOutfwd, secDerInOutfwd] = self.currentFun(lowFwd, lowOut)
			[Ifwd, firDerInFwd[1], firDerOutFwd[1], secDerInfwd, secDerOutfwd, secDerInOutfwd] = self.currentFun(lowFwd, highOut)
			[Ifwd, firDerInFwd[2], firDerOutFwd[2], secDerInfwd, secDerOutfwd, secDerInOutfwd] = self.currentFun(highFwd, lowOut)
			[Ifwd, firDerInFwd[3], firDerOutFwd[3], secDerInfwd, secDerOutfwd, secDerInOutfwd] = self.currentFun(highFwd, highOut)

			lowCc = lowerBound[(i + lenV//2) % lenV]
			highCc = upperBound[(i + lenV//2) % lenV]

			firDerInCc = np.zeros((4))
			firDerOutCc = np.zeros((4))

			[Icc, firDerInCc[0], firDerOutCc[0], secDerIncc, secDerOutcc, secDerInOutCc] = self.currentFun(lowCc, lowOut)
			[Icc, firDerInCc[1], firDerOutCc[1], secDerIncc, secDerOutcc, secDerInOutCc] = self.currentFun(lowCc, highOut)
			[Icc, firDerInCc[2], firDerOutCc[2], secDerIncc, secDerOutcc, secDerInOutCc] = self.currentFun(highCc, lowOut)
			[Icc, firDerInCc[3], firDerOutCc[3], secDerIncc, secDerOutcc, secDerInOutCc] = self.currentFun(highCc, highOut)
			
			minFirDerInFwd = np.amin(firDerInFwd)
			maxFirDerInFwd = np.amax(firDerInFwd)
			minFirDerOutFwd = np.amin(firDerOutFwd)
			maxFirDerOutFwd = np.amax(firDerOutFwd)

			minFirDerInCc = np.amin(firDerInCc)
			maxFirDerInCc = np.amax(firDerInCc)
			minFirDerOutCc = np.amin(firDerOutCc)
			maxFirDerOutCc = np.amax(firDerOutCc)

			jac[i, (i-1)%lenV, 0] = self.g_fwd*minFirDerInFwd
			jac[i, (i-1)%lenV, 1] = self.g_fwd*maxFirDerInFwd
			jac[i, (i + lenV//2) % lenV, 0] = self.g_cc*minFirDerInCc
			jac[i, (i + lenV//2) % lenV, 1] = self.g_cc*maxFirDerInCc
			jac[i, i, 0] = self.g_fwd*minFirDerOutFwd + self.g_cc*minFirDerOutCc
			jac[i, i, 1] = self.g_fwd*maxFirDerOutFwd + self.g_cc*maxFirDerOutCc

		#print ("jac")
		#print (jac)
		return jac

	def linearConstraints(self, hyperRectangle):
		#print ("linearConstraints hyperRectangle")
		#print (hyperRectangle)
		solvers.options["show_progress"] = False
		allConstraints = ""
		lenV = self.numStages*2
		allConstraints = ""
		for i in range(lenV):
			fwdInd = (i-1)%lenV
			ccInd = (i+lenV//2)%lenV
			#print ("i", i, "fwdInd", fwdInd, "ccInd", ccInd)
			
			fwdPatch = np.zeros((4,2))
			fwdPatch[0,:] = [hyperRectangle[fwdInd][0], hyperRectangle[i][0]]
			fwdPatch[1,:] = [hyperRectangle[fwdInd][1], hyperRectangle[i][0]]
			fwdPatch[2,:] = [hyperRectangle[fwdInd][1], hyperRectangle[i][1]]
			fwdPatch[3,:] = [hyperRectangle[fwdInd][0], hyperRectangle[i][1]]

			ccPatch = np.zeros((4,2))
			ccPatch[0,:] = [hyperRectangle[ccInd][0], hyperRectangle[i][0]]
			ccPatch[1,:] = [hyperRectangle[ccInd][1], hyperRectangle[i][0]]
			ccPatch[2,:] = [hyperRectangle[ccInd][1], hyperRectangle[i][1]]
			ccPatch[3,:] = [hyperRectangle[ccInd][0], hyperRectangle[i][1]]
			
			fwdHyperTooSmall = True
			for fi in range(3):
				diff = np.absolute(fwdPatch[fi,:] - fwdPatch[fi+1,:])
				#print ("fwdDiff", diff)
				if np.greater(diff, np.ones(diff.shape)*1e-5).any():
					fwdHyperTooSmall = False
					break

			ccHyperTooSmall = True
			for fi in range(3):
				diff = np.absolute(ccPatch[fi,:] - ccPatch[fi+1,:])
				#print ("ccDiff", diff)
				if np.greater(diff, np.ones(diff.shape)*1e-5).any():
					ccHyperTooSmall = False
					break

			#print ("fwdPatch", fwdPatch)
			if not(fwdHyperTooSmall):
				fwdConstraints = self.ICrossRegConstraint(self.IsFwd[i], self.xs[fwdInd], self.xs[i], fwdPatch)
				allConstraints += fwdConstraints
				#print ("fwdConstraints", fwdConstraints)
			else:
				print ("fwdHyper toosmall", fwdPatch)
			#print ("ccPatch", ccPatch)
			if not(ccHyperTooSmall):
				ccConstraints = self.ICrossRegConstraint(self.IsCc[i], self.xs[ccInd], self.xs[i], ccPatch)
				#print ("ccConstraints", ccConstraints)
				allConstraints += ccConstraints
			else:
				print ("ccHyper toosmall", ccPatch)
			allConstraints += "1 " + self.xs[i] + " >= " + str(hyperRectangle[i][0]) + "\n"
			allConstraints += "1 " + self.xs[i] + " <= " + str(hyperRectangle[i][1]) + "\n"
			allConstraints += str(self.g_fwd) + " " + self.IsFwd[i] + " + " + str(self.g_cc) + " " + self.IsCc[i] + " >= 0.0\n"
			allConstraints += str(self.g_fwd) + " " + self.IsFwd[i] + " + " + str(self.g_cc) + " " + self.IsCc[i] + " <= 0.0\n"

	
		'''allConstraintList = allConstraints.splitlines()
		allConstraints = ""
		for i in range(len(allConstraintList)):
			#if i > 33 or i < 28:
			#if i <= 29:
			allConstraints += allConstraintList[i] + "\n"

		allConstraints += "1 x0 >= 0.90732064" + "\n"
		allConstraints += "1 x0 <= 0.90732064" + "\n"
		allConstraints += "1 x1 >= 0.0" + "\n"
		allConstraints += "1 x1 <= 0.0" + "\n"
		allConstraints += "1 x2 >= 0.125" + "\n"
		allConstraints += "1 x2 <= 0.125" + "\n"
		allConstraints += "1 x3 >= 0.07694534" + "\n"
		allConstraints += "1 x3 <= 0.07694534" + "\n"
		allConstraints += "1 x4 >= 0.09267936" + "\n"
		allConstraints += "1 x4 <= 0.09267936" + "\n"
		allConstraints += "1 x5 >= 1.0" + "\n"
		allConstraints += "1 x5 <= 1.0" + "\n"
		allConstraints += "1 x6 >= 0.875" + "\n"
		allConstraints += "1 x6 <= 0.875" + "\n"
		allConstraints += "1 x7 >= 0.92305466" + "\n"
		allConstraints += "1 x7 <= 0.92305466" + "\n"

		allConstraints += "1 ifwd4 >= 0.2265012\n"
		allConstraints += "1 ifwd4 <= 0.2265013\n"
		allConstraints += "1 icc4 >= -0.0566254\n"
		allConstraints += "1 icc4 <= -0.0566253\n"

		print ("numConstraints ", len(allConstraintList))'''
		#print ("allConstraints")
		#print (allConstraints)
		variableDict, A, B = lpUtils.constructCoeffMatrices(allConstraints)
		#print ("Amat", A)
		#print ("Bmat", B)
		newHyperRectangle = np.copy(hyperRectangle)
		feasible = True
		for i in range(lenV):
			#print ("min max ", i)
			minObjConstraint = "min 1 " + self.xs[i]
			maxObjConstraint = "max 1 " + self.xs[i]
			Cmin = lpUtils.constructObjMatrix(minObjConstraint,variableDict)
			Cmax = lpUtils.constructObjMatrix(maxObjConstraint,variableDict)
			minSol, maxSol = None, None
			try:
				minSol = solvers.lp(Cmin,A,B)
			except ValueError:
				print ("weird constraints", allConstraints)
				pass
			try:
				maxSol = solvers.lp(Cmax,A,B)
			except ValueError:
				pass
			#print ("minSol Status", minSol["status"])
			#print ("maxSol Status", maxSol["status"])
			#print ("minSol", float(minSol["x"][variableDict[self.xs[0]]]))
			#print ("maxSol", float(maxSol["x"][variableDict[self.xs[0]]]))
			if minSol is not None and maxSol is not None:
				if (minSol["status"] == "primal infeasible"  and maxSol["status"] == "primal infeasible"):
					feasible = False
					break
				else:
					if minSol["status"] == "optimal":
						try:
							newHyperRectangle[i,0] = minSol['x'][variableDict[self.xs[i]]] - 1e-5
						except KeyError:
							pass
					if maxSol["status"] == "optimal":
						try:
							newHyperRectangle[i,1] = maxSol['x'][variableDict[self.xs[i]]] + 1e-5
						except KeyError:
							pass
			#print ("newVals", newHyperRectangle[i,:])
			if newHyperRectangle[i,1] < newHyperRectangle[i,0] or newHyperRectangle[i,0] < 0.0 or newHyperRectangle[i,1] > 1.0:
				#print ("old hyper", hyperRectangle[i,:])
				newHyperRectangle[i,:] = hyperRectangle[i,:]
				#print ("new hyper", newHyperRectangle[i,:])


		return [feasible, newHyperRectangle]






