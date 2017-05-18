import numpy as np
import copy

# Applies the simplex on a standard form LP matrix
# the first row is the objective function
# the first element in the first row must be positive
def simplex(standardMat):
	numRows = standardMat.shape[0]
	numCols = standardMat.shape[1]
	count = 0
	while len(np.where(standardMat[0] < 0)[0]) > 0:
		print "mat "
		print standardMat
		print ""
		enteringVariable = np.where(standardMat[0,:] < 0)[0][0]
		ratios = np.divide(standardMat[:,numCols - 1],standardMat[:,enteringVariable])
		ratioSortedIndices = np.argsort(ratios)
		pivot = ratioSortedIndices[1]
		print "entering variable ", enteringVariable
		print "ratios ", ratios
		print "ratioIndices", ratioSortedIndices
		print "pivot ", pivot
		standardMat = gauss_jordan(standardMat,pivot,enteringVariable)
		print "mat after "
		print standardMat
		print ""
		count+=1
	sols = np.zeros((numCols - 1))
	for i in range(len(sols)):
		if np.sum(standardMat[:,i]==0) == numRows - 1:
			nonzeroIndex = np.where(standardMat[:,i] != 0)[0][0]
			if standardMat[nonzeroIndex][i] == 1:
				sols[i] = standardMat[nonzeroIndex][numCols-1]
	return sols


def gauss_jordan(mat, pivot, enteringVariable):
	mat[pivot] = mat[pivot]/mat[pivot][enteringVariable]
	for i in range(mat.shape[0]):
		if i!= pivot:
			mat[i] = mat[i] - mat[i][enteringVariable]*mat[pivot]
	return mat


if __name__=="__main__":
	mat = np.array([[1.0,-5.0,-7.0,0.0,0.0,0.0],
				[0.0,3.0,4.0,1.0,0.0,650.0],
				[0.0,2.0,3.0,0.0,1.0,500.0]])
	'''mat = np.array([[1.0,-1.0,-1.0,0.0,0.0,0.0],
				[0.0,2.0,1.0,1.0,0.0,4.0],
				[0.0,1.0,2.0,0.0,1.0,3.0]])'''
	'''reducedMat = gauss_jordan(mat,1,1)
	print "mat"
	print mat
	print "reducedMat"
	print reducedMat'''
	solutions = simplex(mat)
	print "final solutions"
	print solutions

