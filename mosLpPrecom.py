from circuit import *
import scipy
from scipy.spatial import ConvexHull
import numpy as np
import pickle
from lpUtilsMark import *

# This function samples ids calculated by transistor class defined
# by mosTrans in hyper defined by Vgs (dim 1) and Vds (dim 2).
# numDivisions gives the number of divisions by which each axis is divided
def convexHullFromIdsSample(mosTrans, hyper, numDivisions):
	VgsVals = np.linspace(hyper[0,0], hyper[0,1], numDivisions)
	VdsVals = np.linspace(hyper[1,0], hyper[1,1], numDivisions)
	Ids = np.zeros((len(VgsVals), len(VdsVals)))
	points = []
	for i in range(len(VgsVals)):
		for j in range(len(VdsVals)):
			Ids[i,j] = mosTrans.ids_var2(VgsVals[i], VdsVals[j])
			points.append([VgsVals[i], VdsVals[j], Ids[i,j]])

	points = np.array(points)
	#print ("points", points)
	lp = convexHullConstraints(points)
	#print ("lp", lp.A)
	return lp, points

def convexHullConstraints(points):
	lp = LP()
	try:
		hull = ConvexHull(points)
		convexHullMiddle = np.zeros((points.shape[1]))
		numPoints = 0
		for simplex in hull.simplices:
			for index in simplex:
				convexHullMiddle += points[index,:]
				numPoints += 1
		convexHullMiddle = convexHullMiddle/(numPoints*1.0)
		#print ("hull simplices", hull.simplices)

		for si in range(len(hull.simplices)):
			simplex = hull.simplices[si]
			#print ("simplex", simplex)
			pointsFromSimplex = np.zeros((len(simplex),len(simplex)))
			for ii in range(len(simplex)):
				pointsFromSimplex[ii] = points[simplex[ii]]
			
			#print ("pointsFromSimplex", pointsFromSimplex)
			normal = np.cross(pointsFromSimplex[1] - pointsFromSimplex[0], pointsFromSimplex[2] - pointsFromSimplex[0])
			pointInPlane = pointsFromSimplex[0]
			#print ("pointsFromSimplex", pointsFromSimplex)
			d = normal[0]*pointInPlane[0] + normal[1]*pointInPlane[1] + normal[2]*pointInPlane[2]
			middleD = normal[0]*convexHullMiddle[0] + normal[1]*convexHullMiddle[1] + normal[2]*convexHullMiddle[2]	


			#print ("middleD", middleD)
			#print ("d", d)
			if middleD > d:
				lp.ineq_constraint([-normal[0], -normal[1], -normal[2]], -d)
			else:
				lp.ineq_constraint([normal[0], normal[1], normal[2]], d)

			#print ("normal", normal)
			#print ("pointInPlane", pointInPlane)
			
		return lp
	except scipy.spatial.qhull.QhullError:
		#print ("convex hull not working")
		return lp


# Take the convex hull of the sets of points passed as a list
def convexHullFrom2ConvexHulls(pointsSet):
	allPoints = []
	for points in pointsSet:
		allPoints += list(points)
	allPoints = np.array(allPoints)
	allPoints = np.unique(allPoints, axis = 0)
	lp = convexHullConstraints(allPoints)
	return lp, allPoints

def addToMainDict(mainDict, ulEntry, lrEntry, lp, points):
	if ulEntry in mainDict:
		mainDict[ulEntry][lrEntry] = (lp, points)
	else:
		mainDict[ulEntry] = {}
		mainDict[ulEntry][lrEntry] = (lp, points)	


# precompute convexhulls of each possible rectangle
# of Vgs by Vds in a rectangle defined by
# upper left corner (ulVert) and lower right corner
# (lrVert). The smallest possible rectangle inside
# the big rectangle has length (lrVert[0] - ulVert[0])/numDivisions
# and width (lrVert[1] - ulVert[1])/numDivisions
# numDivSmall decides the distribution for sampling of ids in the
# smallest rectangle using which the convex hull is calculated
def precompute(filename, mosTrans, ulVert, lrVert, numDivisions, numDivSmall):
	unitDiffx = (lrVert[0] - ulVert[0])/(numDivisions*1.0)
	unitDiffy = (lrVert[1] - ulVert[1])/(numDivisions*1.0)
	startx, starty = ulVert
	endx, endy = lrVert
	startIntx, startInty = 0,0
	endIntx, endInty = numDivisions, numDivisions


	mainDict = {}
	
	while startIntx < endIntx and startInty < endInty:
		recStartx, recStarty = startx, starty
		recEndx, recEndy = startx, starty
		recStartIntx, recStartInty = startIntx, startInty
		recEndIntx, recEndInty = startIntx, startInty
		if recEndIntx == endIntx:
			recEndx = ulVert[0]
			recEndIntx = 0
		else:
			recEndx += unitDiffx
			recEndIntx += 1
		recEndy += unitDiffy
		recEndInty += 1
		
		while recEndInty <= endInty:
			print ("ul corner", (recStartIntx, recStartInty), (recStartx, recStarty))
			print ("lr corner", (recEndIntx, recEndInty), (recEndx, recEndy))
			hyper = np.array([[recEndx - unitDiffx, recEndx], [recEndy - unitDiffy, recEndy]])
			#print ("hyper", hyper)
			lp, points = convexHullFromIdsSample(mosTrans, hyper, numDivSmall)
			ulEntry = (recEndIntx - 1, recEndInty - 1)
			lrEntry = (recEndIntx, recEndInty)
			addToMainDict(mainDict, ulEntry, lrEntry, lp, points)

			#print ("lp", str(lp))
			allPoints = [points]
			if recEndInty - recStartInty > 1:
				#print ("adding points for ", (recEndIntx - 1, recStartInty), (recEndIntx, recEndInty-1))
				_, existingPoints = mainDict[(recEndIntx - 1, recStartInty)][(recEndIntx, recEndInty-1)]
				allPoints += [existingPoints]
				lp, points = convexHullFrom2ConvexHulls(allPoints)
				ulEntry = (recEndIntx - 1, recStartInty)
				lrEntry = (recEndIntx, recEndInty)
				addToMainDict(mainDict, ulEntry, lrEntry, lp, points)

			if recEndIntx - recStartIntx > 1:
				#print ("adding points for ", (recStartIntx, recStartInty), (recEndIntx - 1, recEndInty))
				_, existingPoints = mainDict[(recStartIntx, recStartInty)][(recEndIntx - 1, recEndInty)]
				allPoints += [existingPoints]
			
			if len(allPoints) > 1:
				lp, points = convexHullFrom2ConvexHulls(allPoints)
			
			ulEntry = (recStartIntx, recStartInty)
			lrEntry = (recEndIntx, recEndInty)
			addToMainDict(mainDict, ulEntry, lrEntry, lp, points)
			
			if recEndIntx == endIntx:
				recEndx = recStartx + unitDiffx
				recEndIntx = recStartIntx + 1
				recEndy += unitDiffy
				recEndInty += 1
			else:
				recEndx += unitDiffx
				recEndIntx += 1
			#print ("")

		if startIntx == endIntx-1:
			startx = ulVert[0]
			starty += unitDiffy
			startIntx = 0
			startInty += 1
		else:
			startx += unitDiffx
			startIntx += 1

	# do not need to save the points in the file
	# only lp. So get rid of the points
	for key in mainDict:
		for val in mainDict[key]:
			mainDict[key][val] = mainDict[key][val][0]

	allInfo = [numDivisions, mainDict]
	theFile = open(filename, "wb")
	pickle.dump(allInfo, theFile)
	theFile.close()
	return mainDict

def loadMainDict(filename):
	theFile = open(filename, "rb")
	mainDict = pickle.load(theFile)
	theFile.close()
	return mainDict

if __name__ == "__main__":
	nfet = MosfetModel(channelType = 'nfet', Vt =0.4, k = 270.0e-6)
	m1 = Mosfet(0, 1, 2, nfet)
	#mainDict = precompute("precompConvexHull.pkl", m1, (0.0,0.0), (1.8, 1.8), 18, 10)
	mainDict = precompute("precompConvexHull.pkl", m1, (-1.8,-1.8), (1.8, 1.8), 18, 10)
	numDivisions, mainDict = loadMainDict("precompConvexHull.pkl")
	print ("numDivisions", numDivisions)
	'''print ("mainDict")
	for key in mainDict:
		print ("ul corner", key)
		for val in mainDict[key]:
			print ("lr corner", val)
			#print ("lp ", str(mainDict[key][val]))
		print ("")'''





