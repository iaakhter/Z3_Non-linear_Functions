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

		self.constructPolygonRegions()

	def constructPolygonRegions(self):
		self.secDerSigns = [None]*7
		rings = [None]*7
		self.polygonRegs = [None]*7
		rings[0] = ogr.Geometry(ogr.wkbLinearRing)
		rings[0].AddPoint(0.0,0.0)
		rings[0].AddPoint(self.Vtn, 0.0)
		rings[0].AddPoint(self.Vtn, self.Vtn - self.Vtp)
		rings[0].AddPoint(0.0,-self.Vtp)
		rings[0].AddPoint(0.0, 0.0)
		self.polygonRegs[0] = ogr.Geometry(ogr.wkbPolygon)
		self.polygonRegs[0].AddGeometry(rings[0])
		self.secDerSigns[0] = "pos"

		rings[1] = ogr.Geometry(ogr.wkbLinearRing)
		rings[1].AddPoint(0.0,-self.Vtp)
		rings[1].AddPoint(self.Vtn, self.Vtn - self.Vtp)
		rings[1].AddPoint(self.Vtn, 1.0)
		rings[1].AddPoint(0.0,1.0)
		rings[1].AddPoint(0.0,-self.Vtp)
		self.polygonRegs[1] = ogr.Geometry(ogr.wkbPolygon)
		self.polygonRegs[1].AddGeometry(rings[1])
		self.secDerSigns[1] = "neg"

		rings[2] = ogr.Geometry(ogr.wkbLinearRing)
		rings[2].AddPoint(self.Vtn,0.0)
		rings[2].AddPoint(1 + self.Vtp, 0.0)
		rings[2].AddPoint(1 + self.Vtp, 1 + self.Vtp - self.Vtn)
		rings[2].AddPoint(self.Vtn,0.0)
		self.polygonRegs[2] = ogr.Geometry(ogr.wkbPolygon)
		self.polygonRegs[2].AddGeometry(rings[2])
		self.secDerSigns[2] = "pos"

		rings[3] = ogr.Geometry(ogr.wkbLinearRing)
		rings[3].AddPoint(self.Vtn,0.0)
		rings[3].AddPoint(1 + self.Vtp, 1 + self.Vtp - self.Vtn)
		rings[3].AddPoint(1 + self.Vtp, 1.0)
		rings[3].AddPoint(self.Vtn,self.Vtn -self.Vtp)
		rings[3].AddPoint(self.Vtn,0.0)
		self.polygonRegs[3] = ogr.Geometry(ogr.wkbPolygon)
		self.polygonRegs[3].AddGeometry(rings[3])
		self.secDerSigns[3] = "zer"

		rings[4] = ogr.Geometry(ogr.wkbLinearRing)
		rings[4].AddPoint(self.Vtn,self.Vtn -self.Vtp)
		rings[4].AddPoint(1 + self.Vtp, 1.0)
		rings[4].AddPoint(self.Vtn, 1.0)
		rings[4].AddPoint(self.Vtn,self.Vtn -self.Vtp)
		self.polygonRegs[4] = ogr.Geometry(ogr.wkbPolygon)
		self.polygonRegs[4].AddGeometry(rings[4])
		self.secDerSigns[4] = "neg"

		rings[5] = ogr.Geometry(ogr.wkbLinearRing)
		rings[5].AddPoint(1 + self.Vtp,0.0)
		rings[5].AddPoint(1.0, 0.0)
		rings[5].AddPoint(1.0, 1 - self.Vtn)
		rings[5].AddPoint(1 + self.Vtp, 1 + self.Vtp - self.Vtn)
		rings[5].AddPoint(1 + self.Vtp,0.0)
		self.polygonRegs[5] = ogr.Geometry(ogr.wkbPolygon)
		self.polygonRegs[5].AddGeometry(rings[5])
		self.secDerSigns[5] = "pos"

		rings[6] = ogr.Geometry(ogr.wkbLinearRing)
		rings[6].AddPoint(1 + self.Vtp,1 + self.Vtp - self.Vtn)
		rings[6].AddPoint(1.0, 1 - self.Vtn)
		rings[6].AddPoint(1.0, 1.0)
		rings[6].AddPoint(1 + self.Vtp, 1)
		rings[6].AddPoint(1 + self.Vtp,1 + self.Vtp - self.Vtn)
		self.polygonRegs[6] = ogr.Geometry(ogr.wkbPolygon)
		self.polygonRegs[6].AddGeometry(rings[6])
		self.secDerSigns[6] = "neg"


	def currentFun(self, Vin, Vout, regionNumber = None):
		In = 0.0
		firDerInn, firDerOutn = 0.0, 0.0
		secDerInn, secDerOutn = 0.0, 0.0
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
		elif self.Vtn <= Vin and Vin <=Vout + self.Vtn and (eqnN == None or eqnN == 2):
			In = self.Sn*(self.Kn/2.0)*(Vin - self.Vtn)*(Vin - self.Vtn)
			firDerInn = self.Sn*self.Kn*(Vin - self.Vtn)
			firDerOutn = 0.0
			secDerInn = self.Sn*self.Kn
			secDerOutn = 0.0
		elif  Vin >= Vout + self.Vtn and (eqnN == None or eqnN == 3):
			In = self.Sn*(self.Kn)*(Vin - self.Vtn - Vout/2.0)*Vout;
			firDerInn = self.Sn*self.Kn*Vout
			firDerOutn = -self.Sn*self.Kn*Vout
			secDerInn = 0.0
			secDerOutn = -self.Sn*self.Kn

		Ip = 0.0
		firDerInp, firDerOutp = 0.0, 0.0
		secDerInp, secDerOutp = 0.0, 0.0
		if Vin - self.Vtp >= self.Vdd and (eqnP == None or eqnP == 4):
			Ip = 0.0
			firDerInp = 0.0
			firDerOutp = 0.0
			secDerInp = 0.0
			secDerOutp = 0.0
		elif Vout <= Vin - self.Vtp and Vin - self.Vtp <= self.Vdd and (eqnP == None or eqnP == 5):
			Ip = self.Sp*(self.Kp/2.0)*(Vin - self.Vtp - self.Vdd)*(Vin - self.Vtp - self.Vdd)
			firDerInp = self.Sp*self.Kp*(Vin - self.Vtp - self.Vdd)
			firDerOutp = 0.0
			secDerInp = self.Sp*self.Kp
			secDerOutp = 0.0
		elif Vin - self.Vtp <= Vout and (eqnP == None or eqnP == 6):
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
		print ("crossRegionPatch", patch)
		'''INum = [None]*patch.shape[0]
		firDerIn = [None]*patch.shape[0]
		firDerOut = [None]*patch.shape[0]
		secDerIn = [None]*patch.shape[0]
		secDerOut = [None]*patch.shape[0]

		for i in range(len(INum)):
			[INum[i], firDerIn[i], firDerOut[i], secDerIn[i], secDerOut[i]] = self.currentFun(patch[i,0], patch[i,1])'''


		patchRing = ogr.Geometry(ogr.wkbLinearRing)
		for i in range(patch.shape[0] + 1):
			patchRing.AddPoint(patch[(i+1)%patch.shape[0],0], patch[(i+1)%patch.shape[0],1])
		patchPolygon = ogr.Geometry(ogr.wkbPolygon)
		patchPolygon.AddGeometry(patchRing)
		
		feasiblePoints = []
		for i in range(len(self.polygonRegs)):
			polygon = self.polygonRegs[i]
			intersectPoly = polygon.Intersection(patchPolygon)
			intersectPolyRing = intersectPoly.GetGeometryRef(0)
			if intersectPolyRing is not None:
				intersectingPoints = []
				for pi in range(intersectPolyRing.GetPointCount()-1):
					intersectingPoints.append((intersectPolyRing.GetPoint(pi)[0], intersectPolyRing.GetPoint(pi)[1]))
				intersect = np.array(intersectingPoints)
				_,regPoints = self.IRegConstraint(I, Vin, Vout, intersect,i)
				feasiblePoints += regPoints

		# Now construct convex hull with feasible points and add the constraint
		feasiblePoints = np.array(feasiblePoints)

		feasiblePoints = np.unique(feasiblePoints, axis=0)
		print ("convexHullPoints")
		print (feasiblePoints)
		print ("")
		hull = ConvexHull(feasiblePoints)
		convexHullMiddle = np.zeros((3))
		numPoints = 0
		for simplex in hull.simplices:
			for index in simplex:
				convexHullMiddle += feasiblePoints[index,:]
				numPoints += 1
		convexHullMiddle = convexHullMiddle/(numPoints*1.0)
		print ("convexHullMiddle", convexHullMiddle)
		overallConstraint = ""
		for si in range(len(hull.simplices)):
			simplex = hull.simplices[si]
			print ("simplex", simplex)
			pointsFromSimplex = np.zeros((3,3))
			for ii in range(3):
				pointsFromSimplex[ii] = feasiblePoints[simplex[ii]]
			
			print ("pointsFromSimplex", pointsFromSimplex)
			normal = np.cross(pointsFromSimplex[1] - pointsFromSimplex[0], pointsFromSimplex[2] - pointsFromSimplex[0])
			pointInPlane = pointsFromSimplex[0]
			#print ("pointsFromSimplex", pointsFromSimplex)
			d = normal[0]*pointInPlane[0] + normal[1]*pointInPlane[1] + normal[2]*pointInPlane[2]
			middleD = normal[0]*convexHullMiddle[0] + normal[1]*convexHullMiddle[1] + normal[2]*convexHullMiddle[2]
			# Determine if the middle of the convex hull is above or below
			# the plane and add the constraint related to the plane accordingly
			sign = " <= "

			if middleD > d:
				sign = " >= "
			'''if si == 6:
				sign = " >= "'''

			#print ("pointInPlane", pointInPlane)
			#print ("normal", normal)
			#print ("dotSign", dotSign)
			print ("sign", sign)
			print ("normal", normal)
			print ("pointInPlane", pointInPlane)
		
			overallConstraint += str(normal[2])+" " + I + " + " + str(normal[0]) + " " + Vin +\
				" + " + str(normal[1]) + " " + Vout + sign + str(d) + "\n"
			
		print ("overallConstraint")
		print (overallConstraint)

		return overallConstraint


	# patch is a polygon with a list of vertices
	def IRegConstraint(self, I, Vin, Vout, patch, polygonNumber):
		print ("regionPatch", patch, patch.shape)
		regionLines = []
		for i in range(patch.shape[0]):
			point1 = patch[i,:]
			point2 = patch[(i+1)%patch.shape[0],:]
			c, m = None, None
			if point2[0] - point1[0] == 0:
				m = float("inf")
				c = point2[0]
			else:
				m = (point2[1] - point1[1])/(point2[0] - point1[0])
				c = point1[1] - m*point1[0]
			regionLines.append((m,c))
		minBounds = np.amin(patch,0)
		maxBounds = np.amax(patch,0)
		print ("minBounds", minBounds)
		print ("maxBounds", maxBounds)
		INum = [None]*patch.shape[0]
		firDerIn = [None]*patch.shape[0]
		firDerOut = [None]*patch.shape[0]
		secDerIn = [None]*patch.shape[0]
		secDerOut = [None]*patch.shape[0]
		points = np.zeros((patch.shape[0],3))

		for i in range(patch.shape[0]):
			[INum[i], firDerIn[i], firDerOut[i], secDerIn[i], secDerOut[i]] = self.currentFun(patch[i,0], patch[i,1],polygonNumber)
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
		
		# a list of point and normal
		feasiblePlanes = []
		# tangent constraints
		overallConstraint += "1 " + I + " + " + str(-firDerIn[0]) + " " + Vin +\
		" + " + str(-firDerOut[0]) + " " + Vout + tangentSign + str(-firDerIn[0]*patch[0,0] - firDerOut[0]*patch[0,1] + INum[0]) + "\n"
		feasiblePlanes.append([np.array([[points[0][0],points[0][1],points[0][2]],[-firDerIn[0],-firDerOut[0],1]]), tangentSign])

		if patchVertsInsideZeroReg:
			feasiblePoints = []
			for point in points:
				feasiblePoints.append(point)
			return overallConstraint, feasiblePoints


		for i in range(1,len(INum)):
			overallConstraint += "1 " + I + " + " + str(-firDerIn[i]) + " " + Vin +\
			" + " + str(-firDerOut[i]) + " " + Vout + tangentSign + str(-firDerIn[i]*patch[i,0] - firDerOut[i]*patch[i,1] + INum[i]) + "\n"
			feasiblePlanes.append([np.array([[points[i][0],points[i][1],points[i][2]],[-firDerIn[i],-firDerOut[i],1]]), tangentSign])

		# secant constraints

		if len(points) == 3:
			normal = np.cross(points[1] - points[0], points[2] - points[0])
			feasiblePlanes.append([np.array([[points[0][0],points[0][1],points[0][2]],[normal[0],normal[1],normal[2]]]), secantSign])
		else:
			possibleSecantPlanes = []
			excludedPoints = []
			normal = np.cross(points[1] - points[0], points[2] - points[0])
			possibleSecantPlanes.append(np.array([[points[0][0],points[0][1],points[0][2]],[normal[0],normal[1],normal[2]]]))
			excludedPoints.append(points[3])

			normal = np.cross(points[1] - points[0], points[3] - points[0])
			possibleSecantPlanes.append(np.array([[points[0][0],points[0][1],points[0][2]],[normal[0],normal[1],normal[2]]]))
			excludedPoints.append(points[2])

			normal = np.cross(points[2] - points[0], points[3] - points[0])
			possibleSecantPlanes.append(np.array([[points[0][0],points[0][1],points[0][2]],[normal[0],normal[1],normal[2]]]))
			excludedPoints.append(points[1])

			normal = np.cross(points[2] - points[1], points[3] - points[1])
			possibleSecantPlanes.append(np.array([[points[1][0],points[1][1],points[1][2]],[normal[0],normal[1],normal[2]]]))
			excludedPoints.append(points[0])

			numSecantConstraints = 0
			for plane in possibleSecantPlanes:
				if numSecantConstraints >= 2:
					break
				normal = plane[1,:]
				if normal[2] < 0:
					normal = -normal
				includedPt = plane[0,:]
				excludedPt = excludedPoints[i]
				# check if excluded point feasible with plane as a secant
				d = normal[0]*includedPt[0] + normal[1]*includedPt[1] + normal[2]*includedPt[2]
				IAtExcludedPt = (d - normal[0]*excludedPt[0] - normal[1]*excludedPt[1])/normal[2]
				feasible = includedPt[2] <= IAtExcludedPt
				if secantSign == " >= ":
					feasible = includedPt[2] >= IAtExcludedPt
				if feasible:
					overallConstraint += str(normal[2])+ " " + I + " + " + str(normal[0]) + " " + Vin +\
					" + " + str(normal[1]) + " " + Vout + secantSign + str(normal[0]*includedPt[0] +\
						normal[1]*includedPt[1] + normal[2]*includedPt[2]) + "\n"
					numSecantConstraints += 1
					feasiblePlanes.append([plane, secantSign])

		print ("numFeasiblePlanes", len(feasiblePlanes))
		for plane in feasiblePlanes:
			print ("plane", plane)
		print ("")
		
		intersectionPoints = []
		intersectionLines = []
		# find intersection of all possible combinations of feasible planes
		# store points that are satisfied by all constraints
		for i in range(len(feasiblePlanes)):
			for j in range(i+1, len(feasiblePlanes)):
				for k in range(j+1, len(feasiblePlanes)):
					print ("i ", i, "j", j, "k", k)
					#print "onePlane", feasiblePlanes[i][0]
					#print "otherPlane", feasiblePlanes[j][0]
					#print "otherPlane[2]", feasiblePlanes[k][0]
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
					try:
						print ("conditionNumber of AMat", np.linalg.cond(AMat))
						sol = np.linalg.solve(AMat,BMat)
						'''if sol[0] < 0.0:
							sol[0] = 0.0
						elif sol[0] > 1.0:
							sol[0] = 1.0
						if sol[1] < 0.0:
							sol[1] = 0.0
						elif sol[1] > 1.0:
							sol[1] = 1.0'''
						print ("sol", sol)
		
						if sol[0] >= 0.0 and sol[0] <= 1.0 and sol[1] >= 0.0 and sol[1] <= 1.0:
							intersectionPoints.append(np.array([sol[0], sol[1], sol[2]]))
						print ("intersectionPoint added", sol)

					except:
						if norms[0][0] == 0 and norms[1][0] == 0 and norms[2][0] == 0:
							BMat3_1 = np.copy(BMat)
							AMat3_2 = np.delete(AMat,0,1)

							for li in range(3):
								AMat = np.delete(AMat3_2,li,0)
								BMat = np.delete(BMat3_1,li,0)
								try:
									sol = np.linalg.solve(AMat, BMat)
									minX = float("inf")
									maxX = -float("inf")
									for line in regionLines:
										m = line[0]
										c = line[1]
										xOnLine = None
										if m == float("inf"):
											xOnLine = c
										else:
											print ("else m", m, "c", c)
											xOnLine = (sol[0] - c)/m
										if m != 0:
											minX = min(minX, xOnLine)
											maxX = max(maxX, xOnLine)
									if minX < minBounds[0]:
										minX = minBounds[0]
									if maxX > maxBounds[0]:
										maxX = maxBounds[0]
									intersectionPoints.append(np.array([minX, sol[0], sol[1]]))
									print ("intersectionPoint added", intersectionPoints[-1])
									intersectionPoints.append(np.array([maxX, sol[0], sol[1]]))
									print ("intersectionPoint added", intersectionPoints[-1])
									intersectionLines.append({Vout:sol[0], I:sol[1]})
									#print ("intersectionLine added", intersectionLines[-1])
								except np.linalg.LinAlgError:
									#print ("condition number weird ", np.linalg.cond(AMat))
									#print ("AMat", AMat)
									#print ("BMat", BMat)
									pass

						elif norms[0][1] == 0 and norms[1][1] == 0 and norms[2][1] == 0:
							BMat3_1 = np.copy(BMat)
							AMat3_2 = np.delete(AMat,1,1)

							for li in range(3):
								AMat = np.delete(AMat3_2,li,0)
								BMat = np.delete(BMat3_1,li,0)
								try:
									sol = np.linalg.solve(AMat, BMat)
									minY = float("inf")
									maxY = -float("inf")
									for line in regionLines:
										m = line[0]
										c = line[1]
										yOnLine = None
										if m!= float("inf"):
											print ("m", m, "c",c)
											yOnLine = m*sol[0] + c
											print ("yOnLine", yOnLine)
											minY = min(minY, yOnLine)
											maxY = max(maxY, yOnLine)
									intersectionPoints.append(np.array([sol[0], minY, sol[1]]))
									print ("intersectionPoint added", intersectionPoints[-1])
									intersectionPoints.append(np.array([sol[0], maxY, sol[1]]))
									print ("intersectionPoint added", intersectionPoints[-1])
									intersectionLines.append({Vin:sol[0], I:sol[1]})
									#print ("intersectionLine added", intersectionLines[-1])
								except np.linalg.LinAlgError:
									#print ("condition number weird ", np.linalg.cond(AMat))
									#print ("AMat", AMat)
									#print ("BMat", BMat)
									pass

		'''for line in intersectionLines:
			#print ("intersecting line ", line)
			for plane in feasiblePlanes:
				normal = plane[0][1,:]
				pointInPlane = plane[0][0,:]
				if Vin in line:
					if normal[1] != 0:
						d = normal[0]*pointInPlane[0] + normal[1]*pointInPlane[1] + normal[2]*pointInPlane[2]
						varSol = (d - normal[2]*line[I] - normal[0]*line[Vin])/normal[1]
						intersectionPoints.append(np.array([line[Vin], varSol, line[I]]))
						#print ("intersectionPoint added", intersectionPoints[-1])

				if Vout in line:
					if normal[0] != 0:
						d = normal[0]*pointInPlane[0] + normal[1]*pointInPlane[1] + normal[2]*pointInPlane[2]
						varSol = (d - normal[2]*line[I] - normal[1]*line[Vout])/normal[0]
						intersectionPoints.append(np.array([varSol, line[Vout], line[I]]))
						#print ("intersectionPoint added", intersectionPoints[-1])'''

		#print ("intersectionPoints", intersectionPoints)
		print ("len(intersectionPoints)", len(intersectionPoints))
		feasiblePoints = []
		for point in intersectionPoints:
			#print ("intersectionPoint", point)
			pointFeasible = True
			for planeSign in feasiblePlanes:
				plane = planeSign[0]
				sign = planeSign[1]
				normal = plane[1,:]
				planePoint = plane[0,:]
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
				#print ("feasible")
				feasiblePoints.append([point[0], point[1], point[2]])

		#print ("feasiblePoints")
		#print (feasiblePoints)
		print ("len(feasiblePoints)", len(feasiblePoints))

		return overallConstraint, feasiblePoints



	def oscNum(self,V):
		lenV = len(V)
		Vin = [V[i % lenV] for i in range(-1,lenV-1)]
		Vcc = [V[(i + lenV//2) % lenV] for i in range(lenV)]
		IFwd = [self.currentFun(Vin[i], V[i])[0] for i in range(lenV)]
		ICc = [self.currentFun(Vcc[i], V[i])[0] for i in range(lenV)]
		return [IFwd, ICc, [(IFwd[i]*self.g_fwd + ICc[i]*self.g_cc) for i in range(lenV)]]

	def jacobian(self,V):
		lenV = len(V)
		Vin = [V[i % lenV] for i in range(-1,lenV-1)]
		Vcc = [V[(i + lenV//2) % lenV] for i in range(lenV)]
		jac = np.zeros((lenV, lenV))
		for i in range(lenV):
			[Ifwd, firDerInfwd, firDerOutfwd, secDerInfwd, secDerOutfwd] = self.currentFun(Vin[i], V[i])
			[Icc, firDerIncc, firDerOutcc, secDerIncc, secDerOutcc] = self.currentFun(Vcc[i], V[i])
			jac[i, (i-1)%lenV] = self.g_fwd*firDerInfwd
			jac[i, (i + lenV//2) % lenV] = self.g_cc*firDerIncc
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
			jac[i, (i + lenV//2) % lenV, 0] = self.g_cc*minFirDerInCc
			jac[i, (i + lenV//2) % lenV, 1] = self.g_cc*maxFirDerInCc
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
			
			#print ("fwdPatch", fwdPatch)
			fwdConstraints = self.ICrossRegConstraint(self.IsFwd[i], self.xs[fwdInd], self.xs[i], fwdPatch)
			#print ("fwdConstraints", fwdConstraints)
			#print ("ccPatch", ccPatch)
			ccConstraints = self.ICrossRegConstraint(self.IsCc[i], self.xs[ccInd], self.xs[i], ccPatch)
			#print ("ccConstraints", ccConstraints)
			allConstraints += fwdConstraints + ccConstraints
			allConstraints += str(self.g_fwd) + " " + self.IsFwd[i] + " + " + str(self.g_cc) + " " + self.IsCc[i] + " >= 0.0\n"
			allConstraints += str(self.g_fwd) + " " + self.IsFwd[i] + " + " + str(self.g_cc) + " " + self.IsCc[i] + " <= 0.0\n"

		#allConstraints = "2 x + 3 y >= 5\n"
		#allConstraints += "2 x + 3 y <= -5\n"
		'''allConstraintList = allConstraints.splitlines()
		allConstraints = ""
		for i in range(len(allConstraintList)):
			if i!=25:
				allConstraints += allConstraintList[i] + "\n"
		print ("numConstraints ", len(allConstraintList))'''
		print ("allConstraints")
		print (allConstraints)
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
			#print ("minSol Status", minSol["status"])
			#print ("maxSol Status", maxSol["status"])
			#print ("minSol", minSol["x"])
			#print ("maxSol", maxSol["x"])
			if (minSol["status"] == "primal infeasible"  and maxSol["status"] == "primal infeasible"):
				feasible = False
				break
			else:
				if minSol["status"] == "optimal":
					newHyperRectangle[i,0] = minSol['x'][variableDict[self.xs[i]]] - 1e-6
				if maxSol["status"] == "optimal":
					newHyperRectangle[i,1] = maxSol['x'][variableDict[self.xs[i]]] + 1e-6

		return [feasible, newHyperRectangle]






