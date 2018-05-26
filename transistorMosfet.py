import numpy as np
import lpUtils
from cvxopt import matrix,solvers
from scipy.spatial import ConvexHull

class TransistorMosfet:
	def __init__(self, modelParam, channelType, srcVar, gateVar, drainVar, IVar, useLeakage):
		# gradient of tanh -- y = tanh(modelParam*x)
		self.Vt = modelParam[0]
		self.Vdd = modelParam[1]
		self.K = modelParam[2]
		self.S = modelParam[3]
		self.channelType = channelType
		self.useLeakage = useLeakage

		#leakage constant
		self.g = self.idsMax()/self.Vdd

		self.srcVar = srcVar
		self.gateVar = gateVar
		self.drainVar = drainVar
		self.IVar = IVar

	def constructPolygonRegions(self, srcGateDrainHyper):
		try:
			from osgeo import ogr
		except ImportError:
			return

		if self.gateVar is None:
			'''regPts = [[(0.0, 0.0), (srcGateDrainHyper[1][0] - self.Vt, 0.0), (self.srcGateDrainHyper[1][0] - self.Vt, self.srcGateDrainHyper[1][0] - self.Vt)],
						[(0.0, 0.0), (self.srcGateDrainHyper[1][0] - self.Vt, self.srcGateDrainHyper[1][0] - self.Vt), (0.0, self.srcGateDrainHyper[1][0] - self.Vt)],
						[(0.0, self.srcGateDrainHyper[1][0] - self.Vt), (self.srcGateDrainHyper[1][0]- self.Vt, self.srcGateDrainHyper[1][0] - self.Vt), (self.srcGateDrainHyper[1][0] - self.Vt, self.Vdd), (0.0, self.Vdd)],
						[(self.srcGateDrainHyper[1][0] - self.Vt, 0.0), (self.Vdd, 0.0), (self.Vdd, self.srcGateDrainHyper[1][0] - self.Vt), (self.srcGateDrainHyper[1][0] - self.Vt, self.srcGateDrainHyper[1][0] - self.Vt)],
						[(self.srcGateDrainHyper[1][0] - self.Vt, self.srcGateDrainHyper[1][0] - self.Vt), (self.Vdd, self.srcGateDrainHyper[1][0] - self.Vt), (self.Vdd, self.Vdd)],
						[(self.srcGateDrainHyper[1][0] - self.Vt, self.srcGateDrainHyper[1][0] - self.Vt), (self.Vdd, self.Vdd), (self.srcGateDrainHyper[1][0] - self.Vt, self.Vdd)]]'''

			regPts = [[(srcGateDrainHyper[1][0] - self.Vt, srcGateDrainHyper[1][0] - self.Vt), (self.Vdd, self.Vdd), (srcGateDrainHyper[1][0] - self.Vt, self.Vdd)],
						[(0.0, srcGateDrainHyper[1][0] - self.Vt), (srcGateDrainHyper[1][0]- self.Vt, srcGateDrainHyper[1][0] - self.Vt), (srcGateDrainHyper[1][0] - self.Vt, self.Vdd), (0.0, self.Vdd)],
						[(0.0, 0.0), (srcGateDrainHyper[1][0] - self.Vt, srcGateDrainHyper[1][0] - self.Vt), (0.0, srcGateDrainHyper[1][0] - self.Vt)],
						[(srcGateDrainHyper[1][0] - self.Vt, srcGateDrainHyper[1][0] - self.Vt), (self.Vdd, srcGateDrainHyper[1][0] - self.Vt), (self.Vdd, self.Vdd)],
						[(srcGateDrainHyper[1][0] - self.Vt, 0.0), (self.Vdd, 0.0), (self.Vdd, srcGateDrainHyper[1][0] - self.Vt), (srcGateDrainHyper[1][0] - self.Vt, srcGateDrainHyper[1][0] - self.Vt)],
						[(0.0, 0.0), (srcGateDrainHyper[1][0] - self.Vt, 0.0), (srcGateDrainHyper[1][0] - self.Vt, srcGateDrainHyper[1][0] - self.Vt)]]

			if self.channelType == "pFet":
				regPts = [[(0.0, 0.0), (srcGateDrainHyper[1][0] - self.Vt, 0.0), (srcGateDrainHyper[1][0] - self.Vt, srcGateDrainHyper[1][0] - self.Vt)],
						[(srcGateDrainHyper[1][0] - self.Vt, 0.0), (self.Vdd, 0.0), (self.Vdd, srcGateDrainHyper[1][0] - self.Vt), (srcGateDrainHyper[1][0] - self.Vt, srcGateDrainHyper[1][0] - self.Vt)],
						[(srcGateDrainHyper[1][0] - self.Vt, srcGateDrainHyper[1][0] - self.Vt), (self.Vdd, srcGateDrainHyper[1][0] - self.Vt), (self.Vdd, self.Vdd)],
						[(0.0, 0.0), (srcGateDrainHyper[1][0] - self.Vt, srcGateDrainHyper[1][0] - self.Vt), (0.0, srcGateDrainHyper[1][0] - self.Vt)],
						[(0.0, srcGateDrainHyper[1][0] - self.Vt), (srcGateDrainHyper[1][0]- self.Vt, srcGateDrainHyper[1][0] - self.Vt), (srcGateDrainHyper[1][0] - self.Vt, self.Vdd), (0.0, self.Vdd)],
						[(srcGateDrainHyper[1][0] - self.Vt, srcGateDrainHyper[1][0] - self.Vt), (self.Vdd, self.Vdd), (srcGateDrainHyper[1][0] - self.Vt, self.Vdd)]]


		if self.drainVar is None:
			regPts = [[(0.0, 0.0), (srcGateDrainHyper[2][0], 0.0), (srcGateDrainHyper[2][0], srcGateDrainHyper[2][0] + self.Vt), (0.0, self.Vt)],
						[(0.0, self.Vt), (srcGateDrainHyper[2][0], srcGateDrainHyper[2][0] + self.Vt), (0.0, srcGateDrainHyper[2][0] + self.Vt)],
						[(0, srcGateDrainHyper[2][0] + self.Vt), (srcGateDrainHyper[2][0], srcGateDrainHyper[2][0] + self.Vt), (srcGateDrainHyper[2][0], self.Vdd), (0, self.Vdd)],
						[(srcGateDrainHyper[2][0], 0.0), (self.Vdd, 0.0), (self.Vdd, srcGateDrainHyper[2][0] + self.Vt), (srcGateDrainHyper[2][0], srcGateDrainHyper[2][0] + self.Vt)],
						[(srcGateDrainHyper[2][0], srcGateDrainHyper[2][0] + self.Vt), (self.Vdd, srcGateDrainHyper[2][0] + self.Vt), (self.Vdd, self.Vdd), (self.Vdd - self.Vt, self.Vdd)],
						[(srcGateDrainHyper[2][0], srcGateDrainHyper[2][0] + self.Vt), (self.Vdd - self.Vt, self.Vdd), (srcGateDrainHyper[2][0], self.Vdd)]]

			if self.channelType == "pFet":
				'''regPts = [[(srcGateDrainHyper[2][0] + self.Vt, 0.0), (self.srcGateDrainHyper[2][0], 0.0), (self.srcGateDrainHyper[2][0], self.srcGateDrainHyper[2][0] + self.Vt)],
							[(0.0, 0.0), (self.srcGateDrainHyper[2][0] + self.Vt), (self.srcGateDrainHyper[2][0], self.srcGateDrainHyper[2][0] + self.Vt), (0.0, self.srcGateDrainHyper[2][0] + self.Vt)],
							[(0.0, self.srcGateDrainHyper[2][0] + self.Vt), (self.srcGateDrainHyper[2][0], self.srcGateDrainHyper[2][0] + self.Vt), (self.srcGateDrainHyper[2][0], self.Vdd), (0.0, self.Vdd)],
							[(self.srcGateDrainHyper[2][0], self.srcGateDrainHyper[2][0] + self.Vt), (self.Vdd, 0.0), (self.Vdd,self.srcGateDrainHyper[2][0] + self.Vt), (self.srcGateDrainHyper[2][0], self.srcGateDrainHyper[2][0] + self.Vt)],
							[(self.srcGateDrainHyper[2][0], self.srcGateDrainHyper[2][0] + self.Vt), (self.Vdd, self.srcGateDrainHyper[2][0] + Vt), (self.Vdd, self.Vt)],
							[(self.srcGateDrainHyper[2][0], self.srcGateDrainHyper[2][0] + self.Vt), (self.Vdd, self.Vdd + self.Vt), (self.Vdd, self.Vdd), (self.srcGateDrainHyper[2][0], self.Vdd)]]'''

				regPts = [[(srcGateDrainHyper[2][0], srcGateDrainHyper[2][0] + self.Vt), (self.Vdd, self.Vdd + self.Vt), (self.Vdd, self.Vdd), (srcGateDrainHyper[2][0], self.Vdd)],
							[(srcGateDrainHyper[2][0], srcGateDrainHyper[2][0] + self.Vt), (self.Vdd, srcGateDrainHyper[2][0] + self.Vt), (self.Vdd, self.Vt)],
							[(srcGateDrainHyper[2][0], srcGateDrainHyper[2][0] + self.Vt), (self.Vdd, 0.0), (self.Vdd,srcGateDrainHyper[2][0] + self.Vt), (srcGateDrainHyper[2][0], srcGateDrainHyper[2][0] + self.Vt)],
							[(0.0, srcGateDrainHyper[2][0] + self.Vt), (srcGateDrainHyper[2][0], srcGateDrainHyper[2][0] + self.Vt), (srcGateDrainHyper[2][0], self.Vdd), (0.0, self.Vdd)],
							[(0.0, 0.0), (srcGateDrainHyper[2][0] + self.Vt, 0.0), (srcGateDrainHyper[2][0], srcGateDrainHyper[2][0] + self.Vt), (0.0, srcGateDrainHyper[2][0] + self.Vt)],
							[(srcGateDrainHyper[2][0] + self.Vt, 0.0), (srcGateDrainHyper[2][0], 0.0), (srcGateDrainHyper[2][0], srcGateDrainHyper[2][0] + self.Vt)]]


		self.secDerSigns = [None]*len(regPts)
		self.polygonRegs = [None]*len(regPts)

		rings = [None]*len(regPts)
		for pi in range(len(regPts)):
			pts = regPts[pi]
			#print ("pi", pi, "regPtsts[pi]", regPts[pi])
			rings[pi] = ogr.Geometry(ogr.wkbLinearRing)
			for pp in range(len(pts)+1):
				x = pts[pp%len(pts)][0]
				y = pts[pp%len(pts)][1]
				#print ("point", x, y)
				rings[pi].AddPoint(x,y)
			self.polygonRegs[pi] = ogr.Geometry(ogr.wkbPolygon)
			self.polygonRegs[pi].AddGeometry(rings[pi])
			
			secDerIn, secDerOut, secDerInOut = None, None, None
			
			if self.srcVar is None:
				raise CannotHandleError('Cannot Handle the case where we dont vary srcVar')
			if self.gateVar is None:
				[I, firDerSrc, firDerGate, firDerDrain, secDerSrc, secDerGate, secDerDrain, secDerSrcGate, secDerSrcDrain, secDerGateDrain] = self.ids(pts[0][0], srcGateDrainHyper[1][0],pts[0][1], pi)
				secDerIn = secDerSrc
				secDerOut = secDerDrain
				secDerInOut = secDerSrcDrain

			if self.drainVar is None:
				[I, firDerSrc, firDerGate, firDerDrain, secDerSrc, secDerGate, secDerDrain, secDerSrcGate, secDerSrcDrain, secDerGateDrain] = self.ids(pts[0][0], pts[0][1], srcGateDrainHyper[2][0], pi)
				secDerIn = secDerSrc
				secDerOut = secDerGate
				secDerInOut = secDerSrcGate

			if secDerIn == 0 and secDerOut == 0:
				self.secDerSigns[pi] = "zer"

			elif secDerInOut == 0 and secDerIn >= 0 and secDerOut >= 0:
				self.secDerSigns[pi] = "pos"

			elif secDerInOut == 0 and secDerIn <= 0 and secDerOut <= 0:
				self.secDerSigns[pi] = "neg"

			else:
				self.secDerSigns[pi] = "sad"


			#print ("sign ", self.secDerSigns[pi])


	def idsMax(self):
		IMax = 0.0		
		src = 0.0
		gate = self.Vdd
		drain = self.Vdd
		gs = gate - src
		ds = drain - src

		firstCutOff = gs <= self.Vt
		secondCutOff = ds >= gs - self.Vt
		thirdCutOff = ds <= gs - self.Vt

		if self.channelType == "pFet":
			src = self.Vdd
			gate = 0.0
			drain = 0.0
			gs = gate - src
			ds = drain - src
			firstCutOff = gs >= self.Vt
			secondCutOff = ds <= gs - self.Vt
			thirdCutOff = ds >= gs - self.Vt

		if firstCutOff:
			IMax = 0.0;
		elif secondCutOff:
			IMax =  0.5*self.S*self.K*(gs - self.Vt)*(gs - self.Vt);
		elif thirdCutOff:
			IMax = self.S*self.K*(gs - self.Vt - ds/2.0)*ds;

		if self.channelType == "pFet":
			IMax = - IMax

		return IMax

	def leakCurrent(self, src, gate, drain):
		origSrc, origDrain = src, drain

		ILeak = 0.0
		firDerLeakSrc, firDerLeakGate, firDerLeakDrain = 0.0, 0.0, 0.0
		secDerLeakSrc, secDerSrcGate, secDerLeakDrain = 0.0, 0.0, 0.0
		secDerLeakSrcGate, secDerLeakSrcDrain, secDerLeakGateDrain = 0.0, 0.0, 0.0

		if self.channelType == "nFet":
			if origSrc > origDrain:
				src, drain = drain, src
			gs = gate - src
			ds = drain - src
			ILeak = ds*(2 + (gs - self.Vt)/self.Vdd)*(self.g*1e-4)
			firDerLeakSrc = (self.g*1e-4)*((ds)*(-1.0/self.Vdd) - (2 + (gs - self.Vt)/self.Vdd))
			firDerLeakGate = (self.g*1e-4)*(ds/self.Vdd)
			firDerLeakDrain = (self.g*1e-4)*(2 + (gs - self.Vt)/self.Vdd)
			secDerLeakSrc = (2*1e-4)*(self.g/self.Vdd)
			secDerLeakGate = 0.0
			secDerLeakDrain = 0.0
			secDerLeakSrcGate = (self.g*1e-4)*(-1/self.Vdd)
			secDerLeakSrcDrain = (self.g*1e-4)*(-1/self.Vdd)
			secDerLeakGateDrain = (self.g*1e-4)*(1/self.Vdd)
		
		elif self.channelType == "pFet":
			if origSrc < origDrain:
				src, drain = drain, src
			gs = gate - src
			ds = drain - src
			ILeak = ds*(2 - (gs - self.Vt)/self.Vdd)*(self.g*1e-4)
			firDerLeakSrc = (self.g*1e-4)*((ds)*(1.0/self.Vdd) - (2 - (gs - self.Vt)/self.Vdd))
			firDerLeakGate = (self.g*1e-4)*(-ds/self.Vdd)
			firDerLeakDrain = (self.g*1e-4)*(2 - (gs - self.Vt)/self.Vdd)
			secDerLeakSrc = (-2*1e-4)*(self.g/self.Vdd)
			secDerLeakGate = 0.0
			secDerLeakDrain = 0.0
			secDerLeakSrcGate = (self.g*1e-4)*(1/self.Vdd)
			secDerLeakSrcDrain = (self.g*1e-4)*(1/self.Vdd)
			secDerLeakGateDrain = (self.g*1e-4)*(-1/self.Vdd)

		if (self.channelType == "nFet" and origSrc > origDrain) or (self.channelType == "pFet" and origSrc < origDrain):
			ILeak = -ILeak
			firDerLeakSrc, firDerLeakGate, firDerLeakDrain = -firDerLeakSrc, -firDerLeakGate, -firDerLeakDrain
			secDerLeakSrc, secDerSrcGate, secDerLeakDrain = -secDerLeakSrc, -secDerSrcGate, -secDerLeakDrain
			secDerLeakSrcGate, secDerLeakSrcDrain, secDerLeakGateDrain = -secDerLeakSrcGate, -secDerLeakSrcDrain, -secDerLeakGateDrain

		return [ILeak, firDerLeakSrc, firDerLeakGate, firDerLeakDrain, secDerLeakSrc, secDerLeakGate, secDerLeakDrain, secDerLeakSrcGate, secDerLeakSrcDrain, secDerLeakGateDrain]


	def jacobian(self, src, gate, drain):
		[I, firDerSrc, firDerGate, firDerDrain] = self.ids(src, gate, drain)[:4]
		return [firDerSrc, firDerGate, firDerDrain]

	def ids(self, src, gate, drain, polygonNumber = None):
		I = 0.0
		firDerSrc, firDerGate, firDerDrain = 0.0, 0.0, 0.0
		secDerSrc, secDerGate, secDerDrain = 0.0, 0.0, 0.0
		secDerSrcGate, secDerSrcDrain, secDerGateDrain = 0.0, 0.0, 0.0
		origSrc, origDrain = src, drain
		
		if self.channelType == "nFet":
			gs = gate - src
			ds = drain - src

			firstCutOff = gs <= self.Vt and (polygonNumber is None or polygonNumber == 0)
			secondCutOff = ds >= gs - self.Vt and (polygonNumber is None or polygonNumber == 1)
			thirdCutOff = ds <= gs - self.Vt and (polygonNumber is None or polygonNumber == 2)

			if origSrc > origDrain:
				src, drain = drain, src
				gs = gate - src
				ds = drain - src

				firstCutOff = gs <= self.Vt and (polygonNumber is None or polygonNumber == 3)
				secondCutOff = ds >= gs - self.Vt and (polygonNumber is None or polygonNumber == 4)
				thirdCutOff = ds <= gs - self.Vt and (polygonNumber is None or polygonNumber == 5)

		elif self.channelType == "pFet":
			gs = gate - src
			ds = drain - src

			firstCutOff = gs >= self.Vt and (polygonNumber is None or polygonNumber == 0)
			secondCutOff = ds <= gs - self.Vt and (polygonNumber is None or polygonNumber == 1)
			thirdCutOff = ds >= gs - self.Vt and (polygonNumber is None or polygonNumber == 2)

			if origSrc < origDrain:
				src, drain = drain, src
				gs = gate - src
				ds = drain - src

				firstCutOff = gs >= self.Vt and (polygonNumber is None or polygonNumber == 3)
				secondCutOff = ds <= gs - self.Vt and (polygonNumber is None or polygonNumber == 4)
				thirdCutOff = ds >= gs - self.Vt and (polygonNumber is None or polygonNumber == 5)

		if firstCutOff:
			I = 0.0
			firDerSrc, firDerGate, firDerDrain = 0.0, 0.0, 0.0
			secDerSrc, secDerGate, secDerDrain = 0.0, 0.0, 0.0
			secDerSrcGate, secDerSrcDrain, secDerGateDrain = 0.0, 0.0, 0.0

		elif secondCutOff:
			I = self.S*(self.K/2.0)*(gs - self.Vt)*(gs - self.Vt)
			firDerSrc = -self.S*self.K*(gate - src - self.Vt)
			firDerGate = self.S*self.K*(gate - src - self.Vt)
			firDerDrain = 0.0
			secDerSrc = self.S*self.K
			secDerGate = self.S*self.K
			secDerDrain = 0.0
			secDerSrcGate = -self.S*self.K
			secDerSrcDrain = 0.0
			secDerGateDrain = 0.0

		elif thirdCutOff:
			I = self.S*(self.K)*(gs - self.Vt - ds/2.0)*ds
			firDerSrc = self.S*self.K*(src - gate + self.Vt)
			firDerGate = self.S*self.K*(drain - src)
			firDerDrain = self.S*self.K*(gate - self.Vt - drain)
			secDerSrc = self.S*self.K
			secDerGate = 0.0
			secDerDrain = -self.S*self.K
			secDerSrcGate = self.S*self.K
			secDerSrcDrain = 0.0
			secDerGateDrain = self.S*self.K

		if (self.channelType == "nFet" and origSrc > origDrain) or (self.channelType == "pFet" and origSrc < origDrain):
			I = -I
			firDerSrc, firDerGate, firDerDrain = -firDerSrc, -firDerGate, -firDerDrain
			secDerSrc, secDerGate, secDerDrain = -secDerSrc, -secDerGate, -secDerDrain
			secDerSrcGate, secDerSrcDrain, secDerGateDrain = -secDerSrcGate, -secDerSrcDrain, -secDerGateDrain
		
		if self.useLeakage:
			[ILeak, firDerLeakSrc, firDerLeakGate, firDerLeakDrain, secDerLeakSrc, secDerLeakGate, secDerLeakDrain, secDerLeakSrcGate, secDerLeakSrcDrain, secDerLeakGateDrain] = self.leakCurrent(origSrc, gate, origDrain)

			I += ILeak
			firDerSrc += firDerLeakSrc
			firDerGate += firDerLeakGate
			firDerDrain += firDerLeakDrain
			secDerSrc += secDerLeakSrc
			secDerGate += secDerLeakGate
			secDerDrain += secDerLeakDrain
			secDerSrcGate += secDerLeakSrcGate
			secDerSrcDrain += secDerLeakSrcDrain
			secDerGateDrain += secDerLeakGateDrain

		return [I, firDerSrc, firDerGate, firDerDrain, secDerSrc, secDerGate, secDerDrain, secDerSrcGate, secDerSrcDrain, secDerGateDrain]

	def triangleBounds(self, Vin, Vout, Vlow, Vhigh, otherVal1, otherVal2):
		INumLow, dLow, secDerLow = None, None, None
		INumHigh, dHigh, secDerHigh = None, None, None
		if self.srcVar is not None:
			[INumLow, dLow, _, _, secDerLow, _, _, _, _, _] = self.ids(Vlow, otherVal1, otherVal2)
			[INumHigh, dHigh, _, _, secDerHigh, _, _, _, _, _] = self.ids(Vhigh, otherVal1, otherVal2)
		elif self.gateVar is not None:
			[INumLow, _, dLow, _, _, secDerLow, _, _, _, _] = self.ids(otherVal1, Vlow, otherVal2)
			[INumHigh, _, dHigh, _, _, secDerHigh, _, _, _, _] = self.ids(otherVal1, Vhigh, otherVal2)
		elif self.drainVar is not None:
			[INumLow, _, _, dLow, _, _, secDerLow, _, _, _] = self.ids(otherVal1, otherVal2, Vlow)
			[INumHigh, _, _, dHigh, _, _, secDerHigh, _, _, _] = self.ids(otherVal1, otherVal2, Vhigh)
		cLow = INumLow - dLow*Vlow
		cHigh = INumHigh - dHigh*Vhigh

		diff = Vhigh - Vlow
		if(diff == 0):
			diff = 1e-10
		dThird = (INumHigh - INumLow)/diff
		cThird = INumLow - dThird*Vlow

		overallConstraint = ""
		overallConstraint += "1 " + Vin + " >= " + str(Vlow) + "\n"
		overallConstraint += "1 " + Vin + " <= " + str(Vhigh) + "\n"
		if secDerLow*secDerHigh < 0:
			return overallConstraint

		if secDerLow >= 0 and secDerHigh >= 0:
			return overallConstraint + "1 "+ Vout + " + " +str(-dThird) + " " + Vin + " <= "+str(cThird)+"\n" +\
					"1 "+Vout + " + " +str(-dLow) + " " + Vin + " >= "+str(cLow)+"\n" +\
					"1 "+Vout + " + " +str(-dHigh) + " " + Vin + " >= "+str(cHigh) + "\n"

		
		if secDerLow <= 0 and secDerHigh <= 0:
			return overallConstraint + "1 "+ Vout + " + " +str(-dThird) + " " + Vin + " >= "+str(cThird)+"\n" +\
					"1 "+Vout + " + " +str(-dLow) + " " + Vin + " <= "+str(cLow)+"\n" +\
					"1 "+Vout + " + " +str(-dHigh) + " " + Vin + " <= "+str(cHigh) + "\n"

	
	def convexHullConstraints(self, feasiblePoints, I, Vin, Vout):
		hull = ConvexHull(feasiblePoints, qhull_options='QJ')
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

			#print ("middleD", middleD)
			#print ("d", d)
			if middleD > d:
				sign = " >= "

			#print ("normal", normal)
			#print ("pointInPlane", pointInPlane)
			'''print ("sign", sign)
			print ("")'''
			
			#if np.greater_equal(np.absolute(normal),np.ones(normal.shape)*1e-5).any():
			overallConstraint += str(normal[2])+" " + I + " + " + str(normal[0]) + " " + Vin +\
				" + " + str(normal[1]) + " " + Vout + sign + str(d) + "\n"
		return overallConstraint


	# patch is a polygon
	def ICrossRegConstraint(self, I, Vin, Vout, patch, otherVal):
		try:
			from osgeo import ogr
		except ImportError:
			print ("No gdal")
			return
		#print ("crossRegionPatch", patch)
		patchRing = ogr.Geometry(ogr.wkbLinearRing)
		for i in range(patch.shape[0] + 1):
			patchRing.AddPoint(patch[i%patch.shape[0],0], patch[i%patch.shape[0],1])
		patchPolygon = ogr.Geometry(ogr.wkbPolygon)
		patchPolygon.AddGeometry(patchRing)
		
		feasiblePoints = None
		numIntersections = 0
		regConstraints = None
		regPoints = None
		for i in range(len(self.polygonRegs)):
			'''if i!=5:
				continue'''
			polygon = self.polygonRegs[i]
			intersectPolyRing = None
			intersectPoly = polygon.Intersection(patchPolygon)
			if intersectPoly.GetGeometryName() != "LINESTRING":
				#print ("Error here?", intersectPoly.GetGeometryName())
				if intersectPoly.GetGeometryName() != "POINT":
					intersectPolyRing = intersectPoly.GetGeometryRef(0)
				else:
					intersectPolyRing = None
				#print ("Or here?")
			if intersectPolyRing is not None:
				intersectingPoints = []
				for pi in range(intersectPolyRing.GetPointCount()-1):
					intersectingPoints.append((intersectPolyRing.GetPoint(pi)[0], intersectPolyRing.GetPoint(pi)[1]))
				intersect = np.array(intersectingPoints)
				regConstraints,regPoints = self.IRegConstraint(I, Vin, Vout, intersect, otherVal, i)
				#print ("regPoints")
				#print (regPoints)
				if feasiblePoints is None:
					if len(regPoints) >= 1:
						feasiblePoints = regPoints
				else:
					if len(regPoints) >= 1:
						#print ("feasiblePoints", feasiblePoints)
						feasiblePoints = np.vstack([feasiblePoints,regPoints])
				numIntersections += 1
		if numIntersections == 1:
			#print ("regConstraints")
			#print (regConstraints)
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

	def intersectSurfPlaneFunDer(self, Vin, Vout, plane, otherVal):
		#print ("intersectSurfPlaneFunDer", transistorNumber)
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
		S, K, Vt = self.S, self.K, self.Vt
		g = self.g
		if self.gateVar is None:
			firstCutoff = Vin > Vout and Vin - Vout >= otherVal - Vout - Vt
			secondCutoff = Vin > Vout and Vin - Vout <= otherVal - Vout - Vt
			thirdCutoff = Vin <= Vout and Vout - Vin >= otherVal - Vin - Vt
			forthCutoff = Vin <= Vout and Vout - Vin <= otherVal - Vin - Vt
			if self.channelType == "pFet":
				firstCutoff = Vin < Vout and Vin - Vout <= otherVal - Vout - Vt
				secondCutoff = Vin < Vout and Vin - Vout >= otherVal - Vout - Vt
				thirdCutoff = Vin >= Vout and Vout - Vin <=  otherVal - Vin - Vt
				forthCutoff = Vin >= Vout and Vout - Vin >= otherVal - Vin - Vt
			
			if firstCutoff:
				if m is None:
					I += -0.5*S*K*(otherVal - Vout - Vt)*(otherVal - Vout - Vt)
					firDers[1] += S*K*(otherVal - Vout - Vt)
					derTypes[1] = True
					
					if self.useLeakage:
						ILeak = -(g*1e-4)*(d - Vout)*(2 + (otherVal - Vt - Vout)/self.Vdd)
						derLeak = -(g*1e-4)*((d - Vout)/(-self.Vdd) - (2 + (otherVal - Vt - Vout)/self.Vdd))
						if self.channelType == "pFet":
							ILeak = -(g*1e-4)*(d - Vout)*(2 - (otherVal - Vt - Vout)/self.Vdd)
							derLeak = -(g*1e-4)*((d - Vout)/(self.Vdd) - (2 - (otherVal - Vt - Vout)/self.Vdd))
						I += ILeak
						firDers[1] += derLeak
				else:
					I += -0.5*S*K*(otherVal - m*Vin - d - Vt)*(otherVal - m*Vin - d - Vt)
					firDers[0] += m*S*K*(otherVal - m*Vin - d -Vt)
					derTypes[0] = True
					
					if self.useLeakage:
						ILeak = -(g*1e-4)*(Vin - m*Vin - d)*(2 + (otherVal - Vt - m*Vin - d)/self.Vdd)
						derLeak = -(g*1e-4)*((Vin - m*Vin - d)*(-m/self.Vdd) + (2 + (otherVal - Vt - m*Vin - d)/self.Vdd)*(1 - m))
						if self.channelType == "pFet":
							ILeak = -(g*1e-4)*(Vin - m*Vin - d)*(2 - (otherVal - Vt - m*Vin - d)/self.Vdd)
							derLeak = -(g*1e-4)*((Vin - m*Vin - d)*(m/self.Vdd) + (2 - (otherVal - Vt - m*Vin - d)/self.Vdd)*(1 - m))
						I += ILeak
						firDers[0] += derLeak
			elif secondCutoff:
				if m is None:
					I += -S*K*(otherVal - Vout - Vt - (d - Vout)/2.0)*(d - Vout)
					firDers[1] += -S*K*(-(otherVal - Vout - Vt - (d - Vout)/2.0) - 0.5*(d - Vout))
					derTypes[1] = True
					
					if self.useLeakage:
						ILeak = -(g*1e-4)*(d - Vout)*(2 + (otherVal - Vt - Vout)/self.Vdd)
						derLeak = -(g*1e-4)*((d - Vout)/(-self.Vdd) - (2 + (otherVal - Vt - Vout)/self.Vdd))
						if self.channelType == "pFet":
							ILeak = -(g*1e-4)*(d - Vout)*(2 - (otherVal - Vt - Vout)/self.Vdd)
							derLeak = -(g*1e-4)*((d - Vout)/(self.Vdd) - (2 - (otherVal - Vt - Vout)/self.Vdd))
						I += ILeak
						firDers[1] += derLeak
				else:
					I += -S*K*(otherVal - m*Vin - d - Vt - (Vin - m*Vin - d)/2.0)*(Vin - m*Vin - d)
					firDers[0] += -S*K*((1 - m)*(otherVal - m*Vin -d - Vt - (Vin - m*Vin - d)/2.0) + 
										(-m - (1-m)/2.0)*(Vin - m*Vin - d))
					derTypes[0] = True
					
					if self.useLeakage:
						ILeak = -(g*1e-4)*(Vin - m*Vin - d)*(2 + (otherVal - Vt - m*Vin - d)/self.Vdd)
						derLeak = -(g*1e-4)*((Vin - m*Vin - d)*(-m/self.Vdd) + (2 + (otherVal - Vt - m*Vin - d)/self.Vdd)*(1 - m))
						if self.channelType == "pFet":
							ILeak = -(g*1e-4)*(Vin - m*Vin - d)*(2 - (otherVal - Vt - m*Vin - d)/self.Vdd)
							derLeak = -(g*1e-4)*((Vin - m*Vin - d)*(m/self.Vdd) + (2 - (otherVal - Vt - m*Vin - d)/self.Vdd)*(1 - m))
						I += ILeak
						firDers[0] += derLeak
			elif thirdCutoff:
				if m is None:
					I += 0.5*S*K*(otherVal- d - Vt)*(otherVal - d - Vt)
					derTypes[1] = True
					
					if self.useLeakage:
						ILeak = (g*1e-4)*(Vout - d)*(2 + (otherVal- Vt - d)/self.Vdd)
						derLeak = (g*1e-4)*(2 + (otherVal - Vt - d)/self.Vdd)
						if self.channelType == "pFet":
							ILeak = (g*1e-4)*(Vout - d)*(2 - (otherVal - Vt - d)/self.Vdd)
							derLeak = (g*1e-4)*(2 - (otherVal - Vt - d)/self.Vdd)
						I += ILeak
						firDers[1] += derLeak
				else:
					I += 0.5*S*K*(otherVal - Vin - Vt)*(otherVal - Vin - Vt)
					firDers[0] += -S*K*(otherVal - Vin - Vt)
					derTypes[0] = True
					
					if self.useLeakage:
						ILeak = (g*1e-4)*(m*Vin + d - Vin)*(2 + (otherVal - Vt - Vin)/self.Vdd)
						derLeak = (g*1e-4)*((m*Vin + d - Vin)*(-1/self.Vdd) + (2 + (otherVal - Vt - Vin)/self.Vdd)*(m - 1))
						if self.channelType == "pFet":
							ILeak = (g*1e-4)*(m*Vin + d - Vin)*(2 - (otherVal - Vt - Vin)/self.Vdd)
							derLeak = (g*1e-4)*((m*Vin + d - Vin)*(1/self.Vdd) + (2 - (otherVal - Vt - Vin)/self.Vdd)*(m - 1))
						I += ILeak
						firDers[0] += derLeak
			elif forthCutoff:
				if m is None:
					I += S*K*(otherVal - d - Vt - (Vout - d)/2.0)*(Vout - d)
					firDers[1] += S*K*((otherVal - d - Vt - (Vout - d)/2.0) - 0.5*(Vout - d)/2.0)
					derTypes[1] = True
					
					if self.useLeakage:
						ILeak = (g*1e-4)*(Vout - d)*(2 + (otherVal - Vt - d)/self.Vdd)
						derLeak = (g*1e-4)*(2 + (otherVal - Vt - d)/self.Vdd)
						if self.channelType == "pFet":
							ILeak = (g*1e-4)*(Vout - d)*(2 - (otherVal - Vt - d)/self.Vdd)
							derLeak = (g*1e-4)*(2 - (otherVal - Vt - d)/self.Vdd)
						I += ILeak
						firDers[1] += derLeak
				else:
					I += S*K*(otherVal - Vin - Vt - (m*Vin + d - Vin)/2.0)*(m*Vin + d - Vin)
					firDers[0] += S*K*((otherVal- Vin - Vt - (m*Vin + d - Vin)/2.0)*(m-1) + 
										(m*Vin + d - Vin)*(-1 - 0.5*(m-1)))
					derTypes[0] = True
					
					if self.useLeakage:
						ILeak = (g*1e-4)*(m*Vin + d - Vin)*(2 + (otherVal - Vt - Vin)/self.Vdd)
						derLeak = (g*1e-4)*((m*Vin + d - Vin)*(-1/self.Vdd) + (2 + (otherVal - Vt - Vin)/self.Vdd)*(m - 1))
						if self.channelType == "pFet":
							ILeak = (g*1e-4)*(m*Vin + d - Vin)*(2 - (otherVal - Vt - Vin)/self.Vdd)
							derLeak = (g*1e-4)*((m*Vin + d - Vin)*(1/self.Vdd) + (2 - (otherVal - Vt - Vin)/self.Vdd)*(m - 1))
						I += ILeak
						firDers[0] += derLeak

		if self.drainVar is None:
			S, K, Vt = self.S, self.K, self.Vt
			g = self.g
			drain = otherVal
			firstCutoff = Vin > drain and Vin - drain >= Vout - drain - Vt
			secondCutoff = Vin > drain and Vin - drain <= Vout - drain - Vt
			thirdCutoff = drain - Vin >= Vout - Vin - Vt
			forthCutoff = drain - Vin <= Vout - Vin - Vt
			if self.channelType == "pFet":
				firstCutoff = Vin < drain and Vin - drain <= Vout - drain - Vt
				secondCutoff = Vin < drain and Vin - drain >= Vout - drain - Vt
				thirdCutoff = drain - Vin <= Vout - Vin - Vt
				forthCutoff = drain - Vin >= Vout - Vin - Vt
			
			if firstCutoff:
				if m is None:
					I += -0.5*S*K*(Vout - d - Vt)*(Vout - d - Vt)
					firDers[1] += -S*K*(Vout - d - Vt)
					derTypes[1] = True 
					
					if self.useLeakage:
						ILeak = -(g*1e-4)*(d - drain)*(2 + (Vout - Vt - drain)/self.Vdd)
						derLeak = -(g*1e-4)*(d - drain)*(1/self.Vdd)
						if self.channelType == "pFet":
							ILeak = -(g*1e-4)*(d - drain)*(2 - (Vout - Vt - drain)/self.Vdd)
							derLeak = -(g*1e-4)*(d - drain)*(-1/self.Vdd)
						I += ILeak
						firDers[1] += derLeak
				else:
					I += -0.5*S*K*(m*Vin + d - drain - Vt)*(m*Vin + d - drain - Vt)
					firDers[0] += -S*K*(m*Vin + d - self.Vdd - Vt)*(m)
					derTypes[0] = True
					
					if self.useLeakage:
						ILeak = -(g*1e-4)*(Vin - drain)*(2 + (m*Vin + d - Vt - drain)/self.Vdd)
						derLeak = -(g*1e-4)*((Vin - drain)*(m/self.Vdd) + (2 + (m*Vin + d - Vt - drain)/self.Vdd))
						if self.channelType == "pFet":
							ILeak = -(g*1e-4)*(Vin - drain)*(2 - (m*Vin + d - Vt - drain)/self.Vdd)
							derLeak = -(g*1e-4)*((Vin - drain)*(-m/self.Vdd) + (2 - (m*Vin + d - Vt - drain)/self.Vdd))
						I += ILeak
						firDers[0] += derLeak
			elif secondCutoff:
				if m is None:
					I += -S*K*(Vout - drain - Vt - (d - drain)/2.0)*(d - drain)
					firDers[1] += -S*K*(d - drain)
					derTypes[1] = True
					
					if self.useLeakage:
						ILeak = -(g*1e-4)*(d - drain)*(2 + (Vout - Vt - drain)/self.Vdd)
						derLeak = -(g*1e-4)*(d - drain)*(1/self.Vdd)
						if self.channelType == "pFet":
							ILeak = -(g*1e-4)*(d - drain)*(2 - (Vout - Vt - drain)/self.Vdd)
							derLeak = -(g*1e-4)*(d - drain)*(-1/self.Vdd)
						I += ILeak
						firDers[1] += derLeak
				else:
					I += -S*K*(m*Vin + d - drain - Vt - (Vin - drain)/2.0)*(Vin - drain)
					firDers[0] += -S*K*((m*Vin + d - drain - Vt - (Vin - drain)/2.0) + (Vin - drain)*(m - 0.5))
					derTypes[0] = True
					
					if self.useLeakage:
						ILeak = -(g*1e-4)*(Vin - drain)*(2 + (m*Vin + d - Vt - drain)/self.Vdd)
						derLeak = -(g*1e-4)*((Vin - drain)*(m/self.Vdd) + (2 + (m*Vin + d - Vt - drain)/self.Vdd))
						if self.channelType == "pFet":
							ILeak = -(g*1e-4)*(Vin - drain)*(2 - (m*Vin + d - Vt - drain)/self.Vdd)
							derLeak = -(g*1e-4)*((Vin - drain)*(-m/self.Vdd) + (2 - (m*Vin + d - Vt - drain)/self.Vdd))
						I += ILeak
						firDers[0] += derLeak

			elif thirdCutoff:
				if m is None:
					I += 0.5*S*K*(Vout - d - Vt)*(Vout - d - Vt)
					firDers[1] += S*K*(Vout - d - Vt)
					derTypes[1] = True
					
					if self.useLeakage:
						ILeak = (g*1e-4)*(drain - d)*(2 + (Vout - Vt - d)/self.Vdd)
						derLeak = (g*1e-4)*(drain - d)*(1/self.Vdd)
						if self.channelType == "pFet":
							ILeak = (g*1e-4)*(drain - d)*(2 - (Vout - Vt - d)/self.Vdd)
							derLeak = (g*1e-4)*(drain - d)*(-1/self.Vdd)
						I += ILeak
						derTypes[1] += derLeak
				else:
					I += 0.5*S*K*(m*Vin + d - Vin - Vt)*(m*Vin + d - Vin - Vt)
					firDers[0] += -S*K*(m*Vin + d - Vin - Vt)*(m-1)
					derTypes[0] = True
					
					if self.useLeakage:
						ILeak = (g*1e-4)*(drain - Vin)*(2 + (m*Vin + d - Vt - Vin)/self.Vdd)
						derLeak = (g*1e-4)*((drain - Vin)*(m-1)/self.Vdd - (2 + (m*Vin + d - Vt - Vin)/self.Vdd))
						if self.channelType == "pFet":
							ILeak = (g*1e-4)*(drain - Vin)*(2 - (m*Vin + d - Vt - Vin)/self.Vdd)
							derLeak = (g*1e-4)*((drain - Vin)*(-m+1)/self.Vdd - (2 - (m*Vin + d - Vt - Vin)/self.Vdd))
						I += ILeak
						firDers[0] += derLeak
			elif forthCutoff:
				if m is None:
					I += S*K*(Vout - d - Vt - (drain - d)/2.0)*((drain - d)/2.0)
					firDers[1] += S*K*(drain - d)
					derTypes[1] = True
					
					if self.useLeakage:
						ILeak = (g*1e-4)*(drain - d)*(2 + (Vout - Vt - d)/self.Vdd)
						derLeak = (g*1e-4)*(drain - d)*(1/self.Vdd)
						if self.channelType == "pFet":
							ILeak = (g*1e-4)*(drain - d)*(2 - (Vout - Vt - d)/self.Vdd)
							derLeak = (g*1e-4)*(drain - d)*(-1/self.Vdd)
						I += ILeak
						derTypes[1] += derLeak
				else:
					I += 0.5*S*K*(m*Vin + d - Vin - Vt - (drain - Vin)/2.0)*(drain - Vin)
					firDers[0] += S*K*(-(m*Vin + d - Vin - Vt - (drain - Vin)/2.0) + 
						(drain - Vin)*(m - 0.5))
					derTypes[0] = True
					
					if self.useLeakage:
						ILeak = (g*1e-4)*(drain - Vin)*(2 + (m*Vin + d - Vt - Vin)/self.Vdd)
						derLeak = (g*1e-4)*((drain - Vin)*(m-1)/self.Vdd - (2 + (m*Vin + d - Vt - Vin)/self.Vdd))
						if self.channelType == "pFet":
							ILeak = (g*1e-4)*(drain - Vin)*(2 - (m*Vin + d - Vt - Vin)/self.Vdd)
							derLeak = (g*1e-4)*((drain - Vin)*(-m+1)/self.Vdd - (2 - (m*Vin + d - Vt - Vin)/self.Vdd))
						I += ILeak
						firDers[0] += derLeak

		return [I, firDers, derTypes]

	def saddleConvexHull(self, boundaryPlanes, boundaryPts, otherVal):
		#print ("saddleConvexHull", transistorNumber)
		feasiblePoints = []
		for pi in range(len(boundaryPlanes)):
			plane = boundaryPlanes[pi][0]
			point1 = boundaryPts[pi][0]
			point2 = boundaryPts[pi][1]
			funValue1, firDers1, derTypes1 = self.intersectSurfPlaneFunDer(point1[0], point1[1], plane, otherVal)
			funValue2, firDers2, derTypes2 = self.intersectSurfPlaneFunDer(point2[0], point2[1], plane, otherVal)
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
				elif planeNormal[0] != 0:
					missingCoord = (intersectingPt[0] - d)/planeNormal[0]
				if missingCoord is not None:
					feasiblePoints.append([missingCoord,intersectingPt[0],intersectingPt[1]])
				#print ("feasiblePt added if", feasiblePoints[-1])

			elif not(derTypes1[1]) and not(derTypes2[1]):
				m1, m2 = firDers1[0], firDers2[0]
				c1 = point1[2] - m1*point1[0]
				c2 = point2[2] - m2*point2[0]
				#print ("m1", m1, "m2", m2)
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
	def IRegConstraint(self, I, Vin, Vout, patch, otherVal, polygonNumber):
		try:
			from osgeo import ogr
		except ImportError:
			return

		#print ("regionPatch", patch, patch.shape)
		#print ("I", I, "Vin", Vin, "Vout", Vout)
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
			if self.gateVar is None:
				[INum[i], firDerIn[i], firDerGate, firDerOut[i], secDerIn[i], secDerGate, secDerOut[i], secDerSrcGate, secDerSrcDrain, secDerGateDrain] = self.ids(patch[i,0], otherVal, patch[i,1], polygonNumber)
			elif self.drainVar is None:
				[INum[i], firDerIn[i], firDerOut[i], firDerDrain, secDerIn[i], secDerOut[i], secDerDrain, secDerSrcGate, secDerSrcDrain, secDerGateDrain] = self.ids(patch[i,0], patch[i,1], otherVal, polygonNumber)
			#print ("i ", i, "Vin", patch[i,0], "Vout", patch[i,1], "I", INum[i])
			points[i,:] = [patch[i,0],patch[i,1],INum[i]]

		overallConstraint = ""

		tangentSign = " >= "
		secantSign = " <= "

		#print ("secDerIn", secDerIn)
		#print ("secDerOut", secDerOut)
		#print ("secDersign", self.secDerSigns[transistorNumber][polygonNumber])

		patchRing = ogr.Geometry(ogr.wkbLinearRing)
		for i in range(patch.shape[0]+1):
			patchRing.AddPoint(patch[i%patch.shape[0],0], patch[i%patch.shape[0],1])
		patchPolygon = ogr.Geometry(ogr.wkbPolygon)
		patchPolygon.AddGeometry(patchRing)
		#print ("patchPolygon", patchPolygon)
		#print ("sign", self.secDerSigns[polygonNumber])
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
					#print ("point i", points[i,:])
					#print ("point j", points[j,:])
					#print ("point k", points[k,:])
					normal = np.cross(points[k,:] - points[i,:], points[j,:] - points[i,:])
					if normal[2] < 0:
						normal = -normal
					#print ("normal", normal)
					includedPt = points[i,:]
					d = normal[0]*includedPt[0] + normal[1]*includedPt[1] + normal[2]*includedPt[2]
					planeFeasible = True
					for pi in range(points.shape[0]):
						excludedPt = points[pi,:]
						dExcluded = normal[0]*excludedPt[0] + normal[1]*excludedPt[1] + normal[2]*excludedPt[2]
						#print ("pi", pi, points[pi])
						#print ("dExcluded", dExcluded, "d", d, secantSign)
						feasible = dExcluded <= d
						if secantSign == " >= ":
							feasible = dExcluded >= d
						#print ("d - dExcluded", abs(d - dExcluded))
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
			feasiblePoints = self.saddleConvexHull(boundaryPlanes, boundaryPts, otherVal)
			feasiblePoints = np.array(feasiblePoints)
			if len(feasiblePoints) >= 1:
				feasiblePoints = np.unique(feasiblePoints, axis = 0)

			#print ("feasiblePoints", feasiblePoints)
			overallConstraint = self.convexHullConstraints(feasiblePoints, I, Vin, Vout)
		
		#print ("regionFeasiblePoints")
		#print (feasiblePoints)
		#print ("len(feasiblePoints)", len(feasiblePoints))

		return overallConstraint, feasiblePoints


	def linearConstraints(self, srcGateDrainHyper):
		allConstraints = ""
		numVariableIntervals = len(srcGateDrainHyper[0]) + len(srcGateDrainHyper[1]) + len(srcGateDrainHyper[2])
		if self.srcVar is not None and self.gateVar is not None and self.drainVar is not None:
			raise CannotHandleError('Cannot handle 4d case error')

		if numVariableIntervals == 4:
			if self.srcVar is not None:
				allConstraints += self.triangleBounds(self.srcVar, self.IVar, srcGateDrainHyper[0][0], srcGateDrainHyper[0][1], srcGateDrainHyper[1][0], srcGateDrainHyper[2][0])

			elif self.gateVar is not None:
				allConstraints += self.triangleBounds(self.gateVar, self.IVar, srcGateDrainHyper[1][0], srcGateDrainHyper[1][1], srcGateDrainHyper[0][0], srcGateDrainHyper[2][0])

			elif self.drainVar is not None:
				allConstraints += self.triangleBounds(self.drainVar, self.IVar, srcGateDrainHyper[2][0], srcGateDrainHyper[2][1], srcGateDrainHyper[0][0], srcGateDrainHyper[1][0])

		elif numVariableIntervals == 5:
			self.constructPolygonRegions(srcGateDrainHyper)

			patch = np.zeros((4,2))
			inIndex, outIndex = None, None
			inVar, outVar = None, None
			otherVal = None

			if self.srcVar is None:
				raise CannotHandleError('Cannot handle the case where src is a constant yet')

			if self.gateVar is None:
				inIndex, outIndex = 0, 2
				inVar, outVar = self.srcVar, self.drainVar
				otherVal = srcGateDrainHyper[1][0]

			if self.drainVar is None:
				inIndex, outIndex = 0, 1
				inVar, outVar = self.srcVar, self.gateVar
				otherVal = srcGateDrainHyper[2][0]

			patch[0,:] = [srcGateDrainHyper[inIndex][0], srcGateDrainHyper[outIndex][0]]
			patch[1,:] = [srcGateDrainHyper[inIndex][1], srcGateDrainHyper[outIndex][0]]
			patch[2,:] = [srcGateDrainHyper[inIndex][1], srcGateDrainHyper[outIndex][1]]
			patch[3,:] = [srcGateDrainHyper[inIndex][0], srcGateDrainHyper[outIndex][1]]

			hyperTooSmall = True
			for fi in range(3):
				diff = np.absolute(patch[fi,:] - patch[fi+1,:])
				#print ("fwdDiff", diff)
				if np.greater(diff, np.ones(diff.shape)*1e-5).any():
					hyperTooSmall = False
					break
			if not(hyperTooSmall):
				constraints = self.ICrossRegConstraint(self.IVar, inVar, outVar, patch, otherVal)
				allConstraints += constraints
			else:
				#print ("hyperTooSmall", patch)
				pass

		return allConstraints



