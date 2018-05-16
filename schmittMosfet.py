import numpy as np
import lpUtils
from cvxopt import matrix,solvers
from scipy.spatial import ConvexHull

## Remember to add leakage terms so that jacobians are not singular
class SchmittMosfet:
	# modelParam = [Vtp, Vtn, Vdd, Kn, Sn]
	def __init__(self, modelParam, inputVoltage):
		# gradient of tanh -- y = tanh(modelParam*x)
		self.Vtp = modelParam[0]
		self.Vtn = modelParam[1]
		self.Vdd = modelParam[2]
		self.Kn = modelParam[3]
		self.Kp = modelParam[4]
		self.Sn = modelParam[5]
		#self.Kp = -self.Kn/2.0
		#self.Kp = -self.Kn/3.0
		self.Sp = self.Sn*2.0
		self.inputVoltage = inputVoltage
		self.xs = []
		self.tIs = []
		self.nIs = []
		for i in range(3):
			self.xs.append("x" + str(i))
			self.nIs.append("ni" + str(i))
		for i in range(6):
			self.tIs.append("ti" + str(i))
		self.boundMap = []
		for i in range(3):
			self.boundMap.append({0:[0.0,self.Vdd/2.0],1:[self.Vdd/2.0,self.Vdd]})

		# leakage constants
		self.gn = self.nFetMax()/self.Vdd
		self.gp = self.pFetMax()/self.Vdd

		self.constructPolygonRegions()

		self.solver = None

	def constructPolygonRegions(self):
		try:
			from osgeo import ogr
		except ImportError:
			return
		regPts = {1: [[(0.0, 0.0), (self.inputVoltage - self.Vtn, 0.0), (self.inputVoltage - self.Vtn, self.inputVoltage - self.Vtn)],
						[(0.0, 0.0), (self.inputVoltage - self.Vtn, self.inputVoltage - self.Vtn), (0.0, self.inputVoltage - self.Vtn)],
						[(0.0, self.inputVoltage - self.Vtn), (self.inputVoltage - self.Vtn, self.inputVoltage - self.Vtn), (self.inputVoltage - self.Vtn, self.Vdd), (0.0, self.Vdd)],
						[(self.inputVoltage - self.Vtn, 0.0), (self.Vdd, 0.0), (self.Vdd, self.inputVoltage - self.Vtn), (self.inputVoltage - self.Vtn, self.inputVoltage - self.Vtn)],
						[(self.inputVoltage - self.Vtn, self.inputVoltage - self.Vtn), (self.Vdd, self.inputVoltage - self.Vtn), (self.Vdd, self.Vdd)],
						[(self.inputVoltage - self.Vtn, self.inputVoltage - self.Vtn), (self.Vdd, self.Vdd), (self.inputVoltage - self.Vtn, self.Vdd)]],
					2: [[(0.0, 0.0), (self.Vdd, 0.0), (self.Vdd, self.Vdd), (1 - self.Vtn, self.Vdd), (0.0, self.Vtn)],
						[(0.0, self.Vtn), (1 - self.Vtn, self.Vdd), (0.0, self.Vdd)]],
					4: [[(0.0, 0.0), (-self.Vtp + self.inputVoltage, 0.0), (-self.Vtp + self.inputVoltage, -self.Vtp + self.inputVoltage)],
						[(0.0, 0.0), (-self.Vtp + self.inputVoltage, -self.Vtp + self.inputVoltage), (0.0, -self.Vtp + self.inputVoltage)],
						[(0.0, -self.Vtp + self.inputVoltage), (-self.Vtp + self.inputVoltage, -self.Vtp + self.inputVoltage), (-self.Vtp + self.inputVoltage, self.Vdd), (0.0, self.Vdd)],
						[(-self.Vtp + self.inputVoltage, 0.0), (self.Vdd, 0.0), (self.Vdd, -self.Vtp + self.inputVoltage), (-self.Vtp + self.inputVoltage, -self.Vtp + self.inputVoltage)],
						[(-self.Vtp + self.inputVoltage, -self.Vtp + self.inputVoltage), (self.Vdd, -self.Vtp + self.inputVoltage), (self.Vdd, self.Vdd)],
						[(-self.Vtp + self.inputVoltage, -self.Vtp + self.inputVoltage), (self.Vdd, self.Vdd), (-self.Vtp + self.inputVoltage, self.Vdd)]],
					5: [[(0.0, 0.0),(-self.Vtp, 0.0),(self.Vdd, self.Vdd + self.Vtp),(self.Vdd, self.Vdd),(0.0, self.Vdd)],
						[(-self.Vtp, 0.0),(self.Vdd, 0.0),(self.Vdd, self.Vdd + self.Vtp)]]}

		self.secDerSigns = {}
		self.polygonRegs = {}
		for key in regPts:
			self.secDerSigns[key] = [None]*len(regPts[key])
			self.polygonRegs[key] = [None]*len(regPts[key])

			rings = [None]*len(regPts[key])
			Is = [None]*len(regPts[key])
			secDerIns = [None]*len(regPts[key])
			secDerOuts = [None]*len(regPts[key])
			secDerInOuts = [None]*len(regPts[key])
			for pi in range(len(regPts[key])):
				pts = regPts[key][pi]
				rings[pi] = ogr.Geometry(ogr.wkbLinearRing)
				for pp in range(len(pts)+1):
					x = pts[pp%len(pts)][0]
					y = pts[pp%len(pts)][1]
					#print ("point", x, y)
					rings[pi].AddPoint(x,y)
				self.polygonRegs[key][pi] = ogr.Geometry(ogr.wkbPolygon)
				self.polygonRegs[key][pi].AddGeometry(rings[pi])
				
				[Is[pi], firDerIn, firDerOut, secDerIns[pi], secDerOuts[pi], secDerInOuts[pi]] = self.currentFun(pts[0][0], pts[0][1], key, pi)

				if secDerIns[pi] == 0 and secDerOuts[pi] == 0:
					self.secDerSigns[key][pi] = "zer"

				elif secDerInOuts[pi] == 0 and secDerIns[pi] >= 0 and secDerOuts[pi] >= 0:
					self.secDerSigns[key][pi] = "pos"

				elif secDerInOuts[pi] == 0 and secDerIns[pi] <= 0 and secDerOuts[pi] <= 0:
					self.secDerSigns[key][pi] = "neg"

				else:
					self.secDerSigns[key][pi] = "sad"


			#print ("sign ", self.secDerSigns[pi])

	def currentFun(self, VinVal, VoutVal, transistorNumber, polygonNumber):
		INum = 0.0
		firDerIn, firDerOut = 0.0, 0.0
		secDerIn, secDerOut, secDerInOut = 0.0, 0.0, 0.0
		if transistorNumber == 0:
			[INum, firDerIn, firDerGate, firDerOut, secDerIn, secDerGate, secDerOut, secDerSrcGate, secDerInOut, secDerGateDrain] = self.nFet(0.0, self.inputVoltage, VoutVal, transistorNumber)

		if transistorNumber == 1:
			[INum, firDerIn, firDerGate, firDerOut, secDerIn, secDerGate, secDerOut, secDerSrcGate, secDerInOut, secDerGateDrain] = self.nFet(VinVal, self.inputVoltage, VoutVal, transistorNumber)

		if transistorNumber == 2:
			[INum, firDerIn, firDerOut, firDerDrain, secDerIn, secDerOut, secDerDrain, secDerInOut, secDerSrcDrain, secDerGateDrain] = self.nFet(VinVal, VoutVal, self.Vdd, transistorNumber)

		if transistorNumber == 3:
			[INum, firDerIn, firDerGate, firDerOut, secDerIn, secDerGate, secDerOut, secDerSrcGate, secDerInOut, secDerGateDrain] = self.pFet(self.Vdd, self.inputVoltage, VoutVal, transistorNumber)

		if transistorNumber == 4:
			[INum, firDerIn, firDerGate, firDerOut, secDerIn, secDerGate, secDerOut, secDerSrcGate, secDerInOut, secDerGateDrain] = self.pFet(VinVal, self.inputVoltage, VoutVal, transistorNumber)
			#print ("currentFun firDerIn", firDerIn, " firDerOut", firDerOut)

		if transistorNumber == 5:
			[INum, firDerIn, firDerOut, firDerDrain, secDerIn, secDerOut, secDerDrain, secDerInOut, secDerSrcDrain, secDerGateDrain] = self.pFet(VinVal, VoutVal, 0.0, transistorNumber)

		return [INum, firDerIn, firDerOut, secDerIn, secDerOut, secDerInOut]


	def nFetMax(self):
		InMax = 0.0		
		src = 0.0
		gate = self.Vdd
		drain = self.Vdd
		gs = gate - src
		ds = drain - src
		if gs <= self.Vtn:
			InMax = 0.0;
		elif ds >= gs - self.Vtn:
			InMax =  0.5*self.Sn*self.Kn*(gs - self.Vtn)*(gs - self.Vtn);
		elif ds <= gs - self.Vtn:
			InMax = self.Sn*self.Kn*(gs - self.Vtn - ds/2.0)*ds;
		return InMax

	def pFetMax(self):
		IpMax = 0.0
		src = self.Vdd
		gate = 0.0
		drain = 0.0
		gs = gate - src
		ds = drain - src
		  
		if gs >= self.Vtp:
			IpMax = 0.0
		elif ds <= gs - self.Vtp:
			IpMax =  0.5*self.Sp*self.Kp*(gs - self.Vtp)*(gs - self.Vtp)
		elif ds >= gs - self.Vtp:
			IpMax = self.Sp*self.Kp*(gs - self.Vtp - ds/2.0)*ds

		return -IpMax

	def nFet(self, src, gate, drain, transistorNumber, polygonNumber = None):
		I = 0.0
		firDerSrc, firDerGate, firDerDrain = 0.0, 0.0, 0.0
		secDerSrc, secDerGate, secDerDrain = 0.0, 0.0, 0.0
		secDerSrcGate, secDerSrcDrain, secDerGateDrain = 0.0, 0.0, 0.0
		gs = gate - src
		ds = drain - src
		origGs = gs
		origDs = ds
		#print ("nfet transistorNumber", transistorNumber)
		if src > drain:
			src, drain = drain, src
			gs = gate - src
			ds = drain - src
			if gs <= self.Vtn and (polygonNumber == None or
									(transistorNumber == 1 and polygonNumber == 4)):
				#print ("entering zero region", 1)
				I = 0.0
				firDerSrc, firDerGate, firDerDrain = 0.0, 0.0, 0.0
				secDerSrc, secDerGate, secDerDrain = 0.0, 0.0, 0.0
				secDerSrcGate, secDerSrcDrain, secDerGateDrain = 0.0, 0.0, 0.0
			elif ds >= gs - self.Vtn and (polygonNumber == None or
											(transistorNumber == 1 and polygonNumber == 3)):
				#print ("entering non-zero region", 2)
				I = self.Sn*(self.Kn/2.0)*(gs - self.Vtn)*(gs - self.Vtn)
				firDerSrc = -self.Sn*self.Kn*(gate - src - self.Vtn)
				firDerGate = self.Sn*self.Kn*(gate - src - self.Vtn)
				firDerDrain = 0.0
				secDerSrc = self.Sn*self.Kn
				secDerGate = self.Sn*self.Kn
				secDerDrain = 0.0
				secDerSrcGate = -self.Sn*self.Kn
				secDerSrcDrain = 0.0
				secDerGateDrain = 0.0
			elif ds <= gs - self.Vtn and (polygonNumber == None or
											(transistorNumber == 1 and polygonNumber == 0)):
				#print ("entering non-zero region", 3)
				I = self.Sn*(self.Kn)*(gs - self.Vtn - ds/2.0)*ds
				firDerSrc = self.Sn*self.Kn*(src - gate + self.Vtn)
				firDerGate = self.Sn*self.Kn*(drain - src)
				firDerDrain = self.Sn*self.Kn*(gate - self.Vtn - drain)
				secDerSrc = self.Sn*self.Kn
				secDerGate = 0.0
				secDerDrain = -self.Sn*self.Kn
				secDerSrcGate = self.Sn*self.Kn
				secDerSrcDrain = 0.0
				secDerGateDrain = self.Sn*self.Kn
			I = -I
			firDerSrc, firDerGate, firDerDrain = -firDerSrc, -firDerGate, -firDerDrain
			secDerSrc, secDerGate, secDerDrain = -secDerSrc, -secDerGate, -secDerDrain
			secDerSrcGate, secDerSrcDrain, secDerGateDrain = -secDerSrcGate, -secDerSrcDrain, -secDerGateDrain
			InLeak = -ds*(2 + (gs - self.Vtn)/self.Vdd)*(self.gn*1e-4)
			firDerLeakSrc = -(self.gn*1e-4)*((ds)*(-1.0/self.Vdd) - (2 + (gs - self.Vtn)/self.Vdd))
			firDerLeakGate = -(self.gn*1e-4)*(ds/self.Vdd)
			firDerLeakDrain = -(self.gn*1e-4)*(2 + (gs - self.Vtn)/self.Vdd)
			secDerLeakSrc = -(2*1e-4)*(self.gn/self.Vdd)
			secDerLeakGate = 0.0
			secDerLeakDrain = 0.0
			secDerLeakSrcGate = -(self.gn*1e-4)*(-1/self.Vdd)
			secDerLeakSrcDrain = -(self.gn*1e-4)*(-1/self.Vdd)
			secDerLeakGateDrain = -(self.gn*1e-4)*(1/self.Vdd)
		
		else:
			if gs <= self.Vtn and (polygonNumber == None or 
									(transistorNumber == 1 and polygonNumber == 5) or
									(transistorNumber == 2 and polygonNumber == 0)):
				#print ("entering zero region", 4)
				I = 0.0
				firDerSrc, firDerGate, firDerDrain = 0.0, 0.0, 0.0
				secDerSrc, secDerGate, secDerDrain = 0.0, 0.0, 0.0
				secDerSrcGate, secDerSrcDrain, secDerGateDrain = 0.0, 0.0, 0.0
			elif ds >= gs - self.Vtn and (polygonNumber == None or 
											(transistorNumber == 1 and polygonNumber == 2) or
											(transistorNumber == 2 and polygonNumber == 1)):
				#print ("entering non-zero region", 5)
				I = self.Sn*(self.Kn/2.0)*(gs - self.Vtn)*(gs - self.Vtn)
				firDerSrc = -self.Sn*self.Kn*(gate - src - self.Vtn)
				firDerGate = self.Sn*self.Kn*(gate - src - self.Vtn)
				firDerDrain = 0.0
				secDerSrc = self.Sn*self.Kn
				secDerGate = self.Sn*self.Kn
				secDerDrain = 0.0
				secDerSrcGate = -self.Sn*self.Kn
				secDerSrcDrain = 0.0
				secDerGateDrain = 0.0
			elif ds <= gs - self.Vtn and (polygonNumber == None or 
											(transistorNumber == 1 and polygonNumber == 1)):
				#print ("entering non-zero region", 6)
				I = self.Sn*(self.Kn)*(gs - self.Vtn - ds/2.0)*ds
				firDerSrc = self.Sn*self.Kn*(src - gate + self.Vtn)
				firDerGate = self.Sn*self.Kn*(drain - src)
				firDerDrain = self.Sn*self.Kn*(gate - self.Vtn - drain)
				secDerSrc = self.Sn*self.Kn
				secDerGate = 0.0
				secDerDrain = -self.Sn*self.Kn
				secDerSrcGate = self.Sn*self.Kn
				secDerSrcDrain = 0.0
				secDerGateDrain = self.Sn*self.Kn

			#print ("transistorNumber", transistorNumber)
			#print ("ds", ds)
			#print ("gs", gs)
			InLeak = ds*(2 + (gs - self.Vtn)/self.Vdd)*(self.gn*1e-4)
			firDerLeakSrc = (self.gn*1e-4)*((ds)*(-1.0/self.Vdd) - (2 + (gs - self.Vtn)/self.Vdd))
			firDerLeakGate = (self.gn*1e-4)*(ds/self.Vdd)
			firDerLeakDrain = (self.gn*1e-4)*(2 + (gs - self.Vtn)/self.Vdd)
			secDerLeakSrc = (2*1e-4)*(self.gn/self.Vdd)
			secDerLeakGate = 0.0
			secDerLeakDrain = 0.0
			secDerLeakSrcGate = (self.gn*1e-4)*(-1/self.Vdd)
			secDerLeakSrcDrain = (self.gn*1e-4)*(-1/self.Vdd)
			secDerLeakGateDrain = (self.gn*1e-4)*(1/self.Vdd)
			#print ("self.gn", self.gn)
			#print ("InLeak", InLeak)

		#print ("I before leak", I)
		I += InLeak
		#print ("I after leak", I)
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

	def pFet(self, src, gate, drain, transistorNumber, polygonNumber = None):
		I = 0.0
		firDerSrc, firDerGate, firDerDrain = 0.0, 0.0, 0.0
		secDerSrc, secDerGate, secDerDrain = 0.0, 0.0, 0.0
		secDerSrcGate, secDerSrcDrain, secDerGateDrain = 0.0, 0.0, 0.0
		gs = gate - src
		ds = drain - src

		origGs, origDs = gs, ds
		#print ("pfet transistorNumber", transistorNumber)
		#print ("src", src, "gate", gate, "drain", drain)
		if src < drain:
			src, drain = drain, src
			#print ("flipped src", src, "gate", gate, "drain", drain)
			gs = gate - src
			ds = drain - src
			#print ("gs", gs, "ds", ds)
			if gs >= self.Vtp and (polygonNumber == None or 
									(transistorNumber == 4 and polygonNumber == 0) or 
									(transistorNumber == 5 and polygonNumber == 0)):
				#print ("entering zero region 1")
				I = 0.0
				firDerSrc, firDerGate, firDerDrain = 0.0, 0.0, 0.0
				secDerSrc, secDerGate, secDerDrain = 0.0, 0.0, 0.0
				secDerSrcGate, secDerSrcDrain, secDerGateDrain = 0.0, 0.0, 0.0
			elif ds <= gs - self.Vtp and (polygonNumber == None or 
											(transistorNumber == 4 and polygonNumber == 3) or 
											(transistorNumber == 5 and polygonNumber == 1)):
				#print ("entering non-zero region 2")
				I = self.Sp*(self.Kp/2.0)*(gs - self.Vtp)*(gs - self.Vtp)
				firDerSrc = -self.Sp*self.Kp*(gate - src - self.Vtp)
				firDerGate = self.Sp*self.Kp*(gate - src - self.Vtp)
				firDerDrain = 0.0
				secDerSrc = self.Sp*self.Kp
				secDerGate = self.Sp*self.Kp
				secDerDrain = 0.0
				secDerSrcGate = -self.Sp*self.Kp
				secDerSrcDrain = 0.0
				secDerGateDrain = 0.0
			elif ds >= gs - self.Vtp and (polygonNumber == None or 
											(transistorNumber == 4 and polygonNumber == 4)):
				#print ("entering non-zero region 3")
				I = self.Sp*(self.Kp)*(gs - self.Vtp - ds/2.0)*ds
				firDerSrc = self.Sp*self.Kp*(src - gate + self.Vtp)
				firDerGate = self.Sp*self.Kp*(drain - src)
				firDerDrain = self.Sp*self.Kp*(gate - self.Vtp - drain)
				secDerSrc = self.Sp*self.Kp
				secDerGate = 0.0
				secDerDrain = -self.Sp*self.Kp
				secDerSrcGate = self.Sp*self.Kp
				secDerSrcDrain = 0.0
				secDerGateDrain = self.Sp*self.Kp
			#print ("firDerDrain before flippling signs", firDerDrain)
			I = -I
			firDerSrc, firDerGate, firDerDrain = -firDerSrc, -firDerGate, -firDerDrain
			secDerSrc, secDerGate, secDerDrain = -secDerSrc, -secDerGate, -secDerDrain
			secDerSrcGate, secDerSrcDrain, secDerGateDrain = -secDerSrcGate, -secDerSrcDrain, -secDerGateDrain
			#print ("firDerDrain after flippling signs", firDerDrain)
			IpLeak = -ds*(2 - (gs - self.Vtp)/self.Vdd)*(self.gp*1e-4)
			firDerLeakSrc = -(self.gp*1e-4)*((ds)*(1.0/self.Vdd) - (2 - (gs - self.Vtp)/self.Vdd))
			firDerLeakGate = -(self.gp*1e-4)*(-ds/self.Vdd)
			firDerLeakDrain = -(self.gp*1e-4)*(2 - (gs - self.Vtp)/self.Vdd)
			secDerLeakSrc = -(-2*1e-4)*(self.gp/self.Vdd)
			secDerLeakGate = 0.0
			secDerLeakDrain = 0.0
			secDerLeakSrcGate = -(self.gp*1e-4)*(1/self.Vdd)
			secDerLeakSrcDrain = -(self.gp*1e-4)*(1/self.Vdd)
			secDerLeakGateDrain = -(self.gp*1e-4)*(-1/self.Vdd)

		

		else:
			if gs >= self.Vtp and (polygonNumber == None or 
									(transistorNumber == 4 and polygonNumber == 0) or 
									(transistorNumber == 5 and polygonNumber == 0)):
				#print ("entering zero region 4")
				I = 0.0
				firDerSrc, firDerGate, firDerDrain = 0.0, 0.0, 0.0
				secDerSrc, secDerGate, secDerDrain = 0.0, 0.0, 0.0
				secDerSrcGate, secDerSrcDrain, secDerGateDrain = 0.0, 0.0, 0.0
			elif ds <= gs - self.Vtp and (polygonNumber == None or 
											(transistorNumber == 4 and polygonNumber == 3) or 
											(transistorNumber == 5 and polygonNumber == 1)):
				#print ("entering non-zero region 5")
				I = self.Sp*(self.Kp/2.0)*(gs - self.Vtp)*(gs - self.Vtp)
				firDerSrc = -self.Sp*self.Kp*(gate - src - self.Vtp)
				firDerGate = self.Sp*self.Kp*(gate - src - self.Vtp)
				firDerDrain = 0.0
				secDerSrc = self.Sp*self.Kp
				secDerGate = self.Sp*self.Kp
				secDerDrain = 0.0
				secDerSrcGate = -self.Sp*self.Kp
				secDerSrcDrain = 0.0
				secDerGateDrain = 0.0
			elif ds >= gs - self.Vtp and (polygonNumber == None or 
											(transistorNumber == 4 and polygonNumber == 4)):
				#print ("entering non-zero region 6")
				I = self.Sp*(self.Kp)*(gs - self.Vtp - ds/2.0)*ds
				firDerSrc = self.Sp*self.Kp*(src - gate + self.Vtp)
				firDerGate = self.Sp*self.Kp*(drain - src)
				firDerDrain = self.Sp*self.Kp*(gate - self.Vtp - drain)
				secDerSrc = self.Sp*self.Kp
				secDerGate = 0.0
				secDerDrain = -self.Sp*self.Kp
				secDerSrcGate = self.Sp*self.Kp
				secDerSrcDrain = 0.0
				secDerGateDrain = self.Sp*self.Kp

			
			IpLeak = ds*(2 - (gs - self.Vtp)/self.Vdd)*(self.gp*1e-4)
			firDerLeakSrc = (self.gp*1e-4)*((ds)*(1.0/self.Vdd) - (2 - (gs - self.Vtp)/self.Vdd))
			firDerLeakGate = (self.gp*1e-4)*(-ds/self.Vdd)
			firDerLeakDrain = (self.gp*1e-4)*(2 - (gs - self.Vtp)/self.Vdd)
			secDerLeakSrc = (-2*1e-4)*(self.gp/self.Vdd)
			secDerLeakGate = 0.0
			secDerLeakDrain = 0.0
			secDerLeakSrcGate = (self.gp*1e-4)*(1/self.Vdd)
			secDerLeakSrcDrain = (self.gp*1e-4)*(1/self.Vdd)
			secDerLeakGateDrain = (self.gp*1e-4)*(-1/self.Vdd)

		
		#print ("secDerLeakSrc", secDerLeakSrc, "secDerLeakGate", secDerLeakGate)
		#print ("before secDerSrc", secDerSrc, "secDerGate", secDerGate)
		#print ("I before leak", I)
		I += IpLeak
		#print ("I after leak", I)
		firDerSrc += firDerLeakSrc
		firDerGate += firDerLeakGate
		#print ("firDerDrain before leak", firDerDrain)
		firDerDrain += firDerLeakDrain
		#print ("firDerDrain after leak", firDerDrain)
		secDerSrc += secDerLeakSrc
		secDerGate += secDerLeakGate
		secDerDrain += secDerLeakDrain
		secDerSrcGate += secDerLeakSrcGate
		secDerSrcDrain += secDerLeakSrcDrain
		secDerGateDrain += secDerLeakGateDrain
		#print ("after secDerSrc", secDerSrc, "secDerGate", secDerGate)
		
		#print ("firDerSrc", firDerSrc, "firDerGate", firDerGate, "firDerDrain", firDerDrain)
		return [I, firDerSrc, firDerGate, firDerDrain, secDerSrc, secDerGate, secDerDrain, secDerSrcGate, secDerSrcDrain, secDerGateDrain]

	def intersectSurfPlaneFunDer(self, Vin, Vout, plane, transistorNumber):
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
		if transistorNumber == 1 or transistorNumber == 4:
			S, K, Vt = self.Sn, self.Kn, self.Vtn
			g = self.gn
			firstCutoff = Vin > Vout and Vin - Vout >= self.inputVoltage - Vout - Vt
			secondCutoff = Vin > Vout and Vin - Vout <= self.inputVoltage - Vout - Vt
			thirdCutoff = Vin <= Vout and Vout - Vin >= self.inputVoltage - Vin - Vt
			forthCutoff = Vin <= Vout and Vout - Vin <= self.inputVoltage - Vin - Vt
			if transistorNumber == 4:
				S, K, Vt = self.Sp, self.Kp, self.Vtp
				g = self.gp
				firstCutoff = Vin < Vout and Vin - Vout <= self.inputVoltage - Vout - Vt
				secondCutoff = Vin < Vout and Vin - Vout >= self.inputVoltage - Vout - Vt
				thirdCutoff = Vin >= Vout and Vout - Vin <= self.inputVoltage - Vin - Vt
				forthCutoff = Vin >= Vout and Vout - Vin >= self.inputVoltage - Vin - Vt
			
			if firstCutoff:
				if m is None:
					I += -0.5*S*K*(self.inputVoltage - Vout - Vt)*(self.inputVoltage - Vout - Vt)
					firDers[1] += S*K*(self.inputVoltage - Vout - Vt)
					derTypes[1] = True
					ILeak = -(g*1e-4)*(d - Vout)*(2 + (self.inputVoltage - Vt - Vout)/self.Vdd)
					derLeak = -(g*1e-4)*((d - Vout)/(-self.Vdd) - (2 + (self.inputVoltage - Vt - Vout)/self.Vdd))
					if transistorNumber == 4:
						ILeak = -(g*1e-4)*(d - Vout)*(2 - (self.inputVoltage - Vt - Vout)/self.Vdd)
						derLeak = -(g*1e-4)*((d - Vout)/(self.Vdd) - (2 - (self.inputVoltage - Vt - Vout)/self.Vdd))
					I += ILeak
					firDers[1] += derLeak
				else:
					I += -0.5*S*K*(self.inputVoltage - m*Vin - d - Vt)*(self.inputVoltage - m*Vin - d - Vt)
					firDers[0] += m*S*K*(self.inputVoltage - m*Vin - d -Vt)
					derTypes[0] = True
					ILeak = -(g*1e-4)*(Vin - m*Vin - d)*(2 + (self.inputVoltage - Vt - m*Vin - d)/self.Vdd)
					derLeak = -(g*1e-4)*((Vin - m*Vin - d)*(-m/self.Vdd) + (2 + (self.inputVoltage - Vt - m*Vin - d)/self.Vdd)*(1 - m))
					if transistorNumber == 4:
						ILeak = -(g*1e-4)*(Vin - m*Vin - d)*(2 - (self.inputVoltage - Vt - m*Vin - d)/self.Vdd)
						derLeak = -(g*1e-4)*((Vin - m*Vin - d)*(m/self.Vdd) + (2 - (self.inputVoltage - Vt - m*Vin - d)/self.Vdd)*(1 - m))
					I += ILeak
					firDers[0] += derLeak
			elif secondCutoff:
				if m is None:
					I += -S*K*(self.inputVoltage - Vout - Vt - (d - Vout)/2.0)*(d - Vout)
					firDers[1] += -S*K*(-(self.inputVoltage - Vout - Vt - (d - Vout)/2.0) - 0.5*(d - Vout))
					derTypes[1] = True
					ILeak = -(g*1e-4)*(d - Vout)*(2 + (self.inputVoltage - Vt - Vout)/self.Vdd)
					derLeak = -(g*1e-4)*((d - Vout)/(-self.Vdd) - (2 + (self.inputVoltage - Vt - Vout)/self.Vdd))
					if transistorNumber == 4:
						ILeak = -(g*1e-4)*(d - Vout)*(2 - (self.inputVoltage - Vt - Vout)/self.Vdd)
						derLeak = -(g*1e-4)*((d - Vout)/(self.Vdd) - (2 - (self.inputVoltage - Vt - Vout)/self.Vdd))
					I += ILeak
					firDers[1] += derLeak
				else:
					I += -S*K*(self.inputVoltage - m*Vin - d - Vt - (Vin - m*Vin - d)/2.0)*(Vin - m*Vin - d)
					firDers[0] += -S*K*((1 - m)*(self.inputVoltage - m*Vin -d - Vt - (Vin - m*Vin - d)/2.0) + 
										(-m - (1-m)/2.0)*(Vin - m*Vin - d))
					derTypes[0] = True
					ILeak = -(g*1e-4)*(Vin - m*Vin - d)*(2 + (self.inputVoltage - Vt - m*Vin - d)/self.Vdd)
					derLeak = -(g*1e-4)*((Vin - m*Vin - d)*(-m/self.Vdd) + (2 + (self.inputVoltage - Vt - m*Vin - d)/self.Vdd)*(1 - m))
					if transistorNumber == 4:
						ILeak = -(g*1e-4)*(Vin - m*Vin - d)*(2 - (self.inputVoltage - Vt - m*Vin - d)/self.Vdd)
						derLeak = -(g*1e-4)*((Vin - m*Vin - d)*(m/self.Vdd) + (2 - (self.inputVoltage - Vt - m*Vin - d)/self.Vdd)*(1 - m))
					I += ILeak
					firDers[0] += derLeak
			elif thirdCutoff:
				if m is None:
					I += 0.5*S*K*(self.inputVoltage - d - Vt)*(self.inputVoltage - d - Vt)
					derTypes[1] = True
					ILeak = (g*1e-4)*(Vout - d)*(2 + (self.inputVoltage - Vt - d)/self.Vdd)
					derLeak = (g*1e-4)*(2 + (self.inputVoltage - Vt - d)/self.Vdd)
					if transistorNumber == 4:
						ILeak = (g*1e-4)*(Vout - d)*(2 - (self.inputVoltage - Vt - d)/self.Vdd)
						derLeak = (g*1e-4)*(2 - (self.inputVoltage - Vt - d)/self.Vdd)
					I += ILeak
					firDers[1] += derLeak
				else:
					I += 0.5*S*K*(self.inputVoltage - Vin - Vt)*(self.inputVoltage - Vin - Vt)
					firDers[0] += -S*K*(self.inputVoltage - Vin - Vt)
					derTypes[0] = True
					ILeak = (g*1e-4)*(m*Vin + d - Vin)*(2 + (self.inputVoltage - Vt - Vin)/self.Vdd)
					derLeak = (g*1e-4)*((m*Vin + d - Vin)*(-1/self.Vdd) + (2 + (self.inputVoltage - Vt - Vin)/self.Vdd)*(m - 1))
					if transistorNumber == 4:
						ILeak = (g*1e-4)*(m*Vin + d - Vin)*(2 - (self.inputVoltage - Vt - Vin)/self.Vdd)
						derLeak = (g*1e-4)*((m*Vin + d - Vin)*(1/self.Vdd) + (2 - (self.inputVoltage - Vt - Vin)/self.Vdd)*(m - 1))
					I += ILeak
					firDers[0] += derLeak
			elif forthCutoff:
				if m is None:
					I += S*K*(self.inputVoltage - d - Vt - (Vout - d)/2.0)*(Vout - d)
					firDers[1] += S*K*((self.inputVoltage - d - Vt - (Vout - d)/2.0) - 0.5*(Vout - d)/2.0)
					derTypes[1] = True
					ILeak = (g*1e-4)*(Vout - d)*(2 + (self.inputVoltage - Vt - d)/self.Vdd)
					derLeak = (g*1e-4)*(2 + (self.inputVoltage - Vt - d)/self.Vdd)
					if transistorNumber == 4:
						ILeak = (g*1e-4)*(Vout - d)*(2 - (self.inputVoltage - Vt - d)/self.Vdd)
						derLeak = (g*1e-4)*(2 - (self.inputVoltage - Vt - d)/self.Vdd)
					I += ILeak
					firDers[1] += derLeak
				else:
					I += S*K*(self.inputVoltage - Vin - Vt - (m*Vin + d - Vin)/2.0)*(m*Vin + d - Vin)
					firDers[0] += S*K*((self.inputVoltage - Vin - Vt - (m*Vin + d - Vin)/2.0)*(m-1) + 
										(m*Vin + d - Vin)*(-1 - 0.5*(m-1)))
					derTypes[0] = True
					ILeak = (g*1e-4)*(m*Vin + d - Vin)*(2 + (self.inputVoltage - Vt - Vin)/self.Vdd)
					derLeak = (g*1e-4)*((m*Vin + d - Vin)*(-1/self.Vdd) + (2 + (self.inputVoltage - Vt - Vin)/self.Vdd)*(m - 1))
					if transistorNumber == 4:
						ILeak = (g*1e-4)*(m*Vin + d - Vin)*(2 - (self.inputVoltage - Vt - Vin)/self.Vdd)
						derLeak = (g*1e-4)*((m*Vin + d - Vin)*(1/self.Vdd) + (2 - (self.inputVoltage - Vt - Vin)/self.Vdd)*(m - 1))
					I += ILeak
					firDers[0] += derLeak

		if transistorNumber == 2 or transistorNumber == 5:
			S, K, Vt = self.Sn, self.Kn, self.Vtn
			g = self.gn
			drain = self.Vdd
			firstCutoff = Vin > drain and Vin - drain >= Vout - drain - Vt
			secondCutoff = Vin > drain and Vin - drain <= Vout - drain - Vt
			thirdCutoff = drain - Vin >= Vout - Vin - Vt
			forthCutoff = drain - Vin <= Vout - Vin - Vt
			if transistorNumber == 5:
				S, K, Vt = self.Sp, self.Kp, self.Vtp
				g = self.gp
				drain = 0.0
				firstCutoff = Vin < drain and Vin - drain <= Vout - drain - Vt
				secondCutoff = Vin < drain and Vin - drain >= Vout - drain - Vt
				thirdCutoff = drain - Vin <= Vout - Vin - Vt
				forthCutoff = drain - Vin >= Vout - Vin - Vt
			
			if firstCutoff:
				if m is None:
					I += -0.5*S*K*(Vout - d - Vt)*(Vout - d - Vt)
					firDers[1] += -S*K*(Vout - d - Vt)
					derTypes[1] = True 
					ILeak = -(g*1e-4)*(d - drain)*(2 + (Vout - Vt - drain)/self.Vdd)
					derLeak = -(g*1e-4)*(d - drain)*(1/self.Vdd)
					if transistorNumber == 5:
						ILeak = -(g*1e-4)*(d - drain)*(2 - (Vout - Vt - drain)/self.Vdd)
						derLeak = -(g*1e-4)*(d - drain)*(-1/self.Vdd)
					I += ILeak
					firDers[1] += derLeak
				else:
					I += -0.5*S*K*(m*Vin + d - drain - Vt)*(m*Vin + d - drain - Vt)
					firDers[0] += -S*K*(m*Vin + d - self.Vdd - Vt)*(m)
					derTypes[0] = True
					ILeak = -(g*1e-4)*(Vin - drain)*(2 + (m*Vin + d - Vt - drain)/self.Vdd)
					derLeak = -(g*1e-4)*((Vin - drain)*(m/self.Vdd) + (2 + (m*Vin + d - Vt - drain)/self.Vdd))
					if transistorNumber == 5:
						ILeak = -(g*1e-4)*(Vin - drain)*(2 - (m*Vin + d - Vt - drain)/self.Vdd)
						derLeak = -(g*1e-4)*((Vin - drain)*(-m/self.Vdd) + (2 - (m*Vin + d - Vt - drain)/self.Vdd))
					I += ILeak
					firDers[0] += derLeak
			elif secondCutoff:
				if m is None:
					I += -S*K*(Vout - drain - Vt - (d - drain)/2.0)*(d - drain)
					firDers[1] += -S*K*(d - drain)
					derTypes[1] = True
					ILeak = -(g*1e-4)*(d - drain)*(2 + (Vout - Vt - drain)/self.Vdd)
					derLeak = -(g*1e-4)*(d - drain)*(1/self.Vdd)
					if transistorNumber == 5:
						ILeak = -(g*1e-4)*(d - drain)*(2 - (Vout - Vt - drain)/self.Vdd)
						derLeak = -(g*1e-4)*(d - drain)*(-1/self.Vdd)
					I += ILeak
					firDers[1] += derLeak
				else:
					I += -S*K*(m*Vin + d - drain - Vt - (Vin - drain)/2.0)*(Vin - drain)
					firDers[0] += -S*K*((m*Vin + d - drain - Vt - (Vin - drain)/2.0) + (Vin - drain)*(m - 0.5))
					derTypes[0] = True
					ILeak = -(g*1e-4)*(Vin - drain)*(2 + (m*Vin + d - Vt - drain)/self.Vdd)
					derLeak = -(g*1e-4)*((Vin - drain)*(m/self.Vdd) + (2 + (m*Vin + d - Vt - drain)/self.Vdd))
					if transistorNumber == 5:
						ILeak = -(g*1e-4)*(Vin - drain)*(2 - (m*Vin + d - Vt - drain)/self.Vdd)
						derLeak = -(g*1e-4)*((Vin - drain)*(-m/self.Vdd) + (2 - (m*Vin + d - Vt - drain)/self.Vdd))
					I += ILeak
					firDers[0] += derLeak

			elif thirdCutoff:
				if m is None:
					I += 0.5*S*K*(Vout - d - Vt)*(Vout - d - Vt)
					firDers[1] += S*K*(Vout - d - Vt)
					derTypes[1] = True
					ILeak = (g*1e-4)*(drain - d)*(2 + (Vout - Vt - d)/self.Vdd)
					derLeak = (g*1e-4)*(drain - d)*(1/self.Vdd)
					if transistorNumber == 5:
						ILeak = (g*1e-4)*(drain - d)*(2 - (Vout - Vt - d)/self.Vdd)
						derLeak = (g*1e-4)*(drain - d)*(-1/self.Vdd)
					I += ILeak
					derTypes[1] += derLeak
				else:
					I += 0.5*S*K*(m*Vin + d - Vin - Vt)*(m*Vin + d - Vin - Vt)
					firDers[0] += -S*K*(m*Vin + d - Vin - Vt)*(m-1)
					derTypes[0] = True
					ILeak = (g*1e-4)*(drain - Vin)*(2 + (m*Vin + d - Vt - Vin)/self.Vdd)
					derLeak = (g*1e-4)*((drain - Vin)*(m-1)/self.Vdd - (2 + (m*Vin + d - Vt - Vin)/self.Vdd))
					if transistorNumber == 5:
						ILeak = (g*1e-4)*(drain - Vin)*(2 - (m*Vin + d - Vt - Vin)/self.Vdd)
						derLeak = (g*1e-4)*((drain - Vin)*(-m+1)/self.Vdd - (2 - (m*Vin + d - Vt - Vin)/self.Vdd))
					I += ILeak
					firDers[0] += derLeak
			elif forthCutoff:
				if m is None:
					I += S*K*(Vout - d - Vt - (drain - d)/2.0)*((drain - d)/2.0)
					firDers[1] += S*K*(drain - d)
					derTypes[1] = True
					ILeak = (g*1e-4)*(drain - d)*(2 + (Vout - Vt - d)/self.Vdd)
					derLeak = (g*1e-4)*(drain - d)*(1/self.Vdd)
					if transistorNumber == 5:
						ILeak = (g*1e-4)*(drain - d)*(2 - (Vout - Vt - d)/self.Vdd)
						derLeak = (g*1e-4)*(drain - d)*(-1/self.Vdd)
					I += ILeak
					derTypes[1] += derLeak
				else:
					I += 0.5*S*K*(m*Vin + d - Vin - Vt - (drain - Vin)/2.0)*(drain - Vin)
					firDers[0] += S*K*(-(m*Vin + d - Vin - Vt - (drain - Vin)/2.0) + 
						(drain - Vin)*(m - 0.5))
					derTypes[0] = True
					ILeak = (g*1e-4)*(drain - Vin)*(2 + (m*Vin + d - Vt - Vin)/self.Vdd)
					derLeak = (g*1e-4)*((drain - Vin)*(m-1)/self.Vdd - (2 + (m*Vin + d - Vt - Vin)/self.Vdd))
					if transistorNumber == 5:
						ILeak = (g*1e-4)*(drain - Vin)*(2 - (m*Vin + d - Vt - Vin)/self.Vdd)
						derLeak = (g*1e-4)*((drain - Vin)*(-m+1)/self.Vdd - (2 - (m*Vin + d - Vt - Vin)/self.Vdd))
					I += ILeak
					firDers[0] += derLeak

		return [I, firDers, derTypes]

	
	def triangleBounds(self, Vin, Vout, Vlow, Vhigh, transistorNumber):
		[INumLow, _, dLow, _, secDerLow, _] = self.currentFun(None, Vlow, transistorNumber, None)
		[INumHigh, _, dHigh, _, secDerHigh, _] = self.currentFun(None, Vhigh, transistorNumber, None)
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
	def ICrossRegConstraint(self,I, Vin, Vout, patch, transistorNumber):
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
		for i in range(len(self.polygonRegs[transistorNumber])):
			'''if i!=5:
				continue'''
			polygon = self.polygonRegs[transistorNumber][i]
			intersectPolyRing = None
			intersectPoly = polygon.Intersection(patchPolygon)
			if intersectPoly.GetGeometryName() != "LINESTRING":
				#print ("Error here?", intersectPoly.GetGeometryName())
				intersectPolyRing = intersectPoly.GetGeometryRef(0)
				#print ("Or here?")
			if intersectPolyRing is not None:
				intersectingPoints = []
				for pi in range(intersectPolyRing.GetPointCount()-1):
					intersectingPoints.append((intersectPolyRing.GetPoint(pi)[0], intersectPolyRing.GetPoint(pi)[1]))
				intersect = np.array(intersectingPoints)
				regConstraints,regPoints = self.IRegConstraint(I, Vin, Vout, intersect,transistorNumber,i)
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

	def saddleConvexHull(self, boundaryPlanes, boundaryPts, transistorNumber):
		#print ("saddleConvexHull", transistorNumber)
		feasiblePoints = []
		for pi in range(len(boundaryPlanes)):
			plane = boundaryPlanes[pi][0]
			point1 = boundaryPts[pi][0]
			point2 = boundaryPts[pi][1]
			funValue1, firDers1, derTypes1 = self.intersectSurfPlaneFunDer(point1[0], point1[1], plane, transistorNumber)
			funValue2, firDers2, derTypes2 = self.intersectSurfPlaneFunDer(point2[0], point2[1], plane, transistorNumber)
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
	def IRegConstraint(self, I, Vin, Vout, patch, transistorNumber, polygonNumber):
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
			[INum[i], firDerIn[i], firDerOut[i], secDerIn[i], secDerOut[i], secDerInOut] = self.currentFun(patch[i,0], patch[i,1],transistorNumber, polygonNumber)
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
		if self.secDerSigns[transistorNumber][polygonNumber] == "zer":
			patchVertsInsideZeroReg = True
		elif self.secDerSigns[transistorNumber][polygonNumber] == "neg":
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

		if self.secDerSigns[transistorNumber][polygonNumber] != "sad":
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
			feasiblePoints = self.saddleConvexHull(boundaryPlanes, boundaryPts, transistorNumber)
			feasiblePoints = np.array(feasiblePoints)
			if len(feasiblePoints) >= 1:
				feasiblePoints = np.unique(feasiblePoints, axis = 0)

			#print ("feasiblePoints", feasiblePoints)
			overallConstraint = self.convexHullConstraints(feasiblePoints, I, Vin, Vout)
		
		#print ("regionFeasiblePoints")
		#print (feasiblePoints)
		#print ("len(feasiblePoints)", len(feasiblePoints))

		return overallConstraint, feasiblePoints



	def oscNum(self,V):
		transCurs = np.zeros((6))
		transCurs[0] = self.currentFun(None, V[1], 0, None)[0]
		transCurs[1] = self.currentFun(V[1], V[0], 1, None)[0]
		transCurs[2] = self.currentFun(V[1], V[0], 2, None)[0]
		transCurs[3] = self.currentFun(None, V[0], 3, None)[0]
		transCurs[4] = self.currentFun(V[2], V[0], 4, None)[0]
		transCurs[5] = self.currentFun(V[2], V[0], 5, None)[0]
		#print ("transCurs", transCurs)
		nodeCurs = np.zeros((3))
		nodeCurs[0] = -transCurs[4] - transCurs[1]
		nodeCurs[1] = -transCurs[0] + transCurs[1] + transCurs[2]
		nodeCurs[2] = -transCurs[3] + transCurs[5] + transCurs[4]
		return [None, None, nodeCurs]

	def jacobian(self,V):
		#print ("calculating jacobian")
		jac = np.zeros((3,3))
		transCurs = np.zeros((6))
		transCurDerIns = np.zeros((6))
		transCurDerOuts = np.zeros((6))
		transCurs[0], transCurDerIns[0], transCurDerOuts[0] = self.currentFun(None, V[1], 0, None)[:3]
		transCurs[1], transCurDerIns[1], transCurDerOuts[1] = self.currentFun(V[1], V[0], 1, None)[:3]
		transCurs[2], transCurDerIns[2], transCurDerOuts[2] = self.currentFun(V[1], V[0], 2, None)[:3]
		transCurs[3], transCurDerIns[3], transCurDerOuts[3] = self.currentFun(None, V[0], 3, None)[:3]
		transCurs[4], transCurDerIns[4], transCurDerOuts[4] = self.currentFun(V[2], V[0], 4, None)[:3]
		transCurs[5], transCurDerIns[5], transCurDerOuts[5] = self.currentFun(V[2], V[0], 5, None)[:3]

		#print ("transCurDerOuts[4]", transCurDerOuts[4])
		jac[0,0] = -transCurDerOuts[4] - transCurDerOuts[1]
		jac[0,1] = -transCurDerIns[1]
		jac[0,2] = -transCurDerIns[4]

		jac[1,0] = transCurDerOuts[1] + transCurDerOuts[2]
		jac[1,1] = -transCurDerOuts[0] + transCurDerIns[1] + transCurDerIns[2]
		jac[1,2] = 0.0

		jac[2,0] = -transCurDerOuts[3] + transCurDerOuts[5] + transCurDerOuts[4]
		jac[2,1] = 0.0
		jac[2,2] = transCurDerIns[5] + transCurDerIns[4]

		return jac

	def jacobianInterval(self, bounds):
		lowerBound = bounds[:,0]
		upperBound = bounds[:,1]
		lenV  = len(lowerBound)
		jac = np.zeros((lenV, lenV, 2))

		transCurDerIns0 = np.zeros((2))
		transCurDerOuts0 = np.zeros((2))

		transCurDerIns1 = np.zeros((4))
		transCurDerOuts1 = np.zeros((4))

		transCurDerIns2 = np.zeros((4))
		transCurDerOuts2 = np.zeros((4))

		transCurDerIns3 = np.zeros((2))
		transCurDerOuts3 = np.zeros((2))

		transCurDerIns4 = np.zeros((4))
		transCurDerOuts4 = np.zeros((4))

		transCurDerIns5 = np.zeros((4))
		transCurDerOuts5 = np.zeros((4))

		transCurs0L, transCurDerIns0[0], transCurDerOuts0[0] = self.currentFun(None, lowerBound[1], 0, None)[:3]
		transCurs0H, transCurDerIns0[1], transCurDerOuts0[1] = self.currentFun(None, upperBound[1], 0, None)[:3]
		
		transCurs1LL, transCurDerIns1[0], transCurDerOuts1[0] = self.currentFun(lowerBound[1], lowerBound[0], 1, None)[:3]
		transCurs1LH, transCurDerIns1[1], transCurDerOuts1[1] = self.currentFun(lowerBound[1], upperBound[0], 1, None)[:3]
		transCurs1HL, transCurDerIns1[2], transCurDerOuts1[2] = self.currentFun(upperBound[1], lowerBound[0], 1, None)[:3]
		transCurs1HH, transCurDerIns1[3], transCurDerOuts1[3] = self.currentFun(upperBound[1], upperBound[0], 1, None)[:3]
		
		transCurs2LL, transCurDerIns2[0], transCurDerOuts2[0] = self.currentFun(lowerBound[1], lowerBound[0], 2, None)[:3]
		transCurs2LH, transCurDerIns2[1], transCurDerOuts2[1] = self.currentFun(lowerBound[1], upperBound[0], 2, None)[:3]
		transCurs2HL, transCurDerIns2[2], transCurDerOuts2[2] = self.currentFun(upperBound[1], lowerBound[0], 2, None)[:3]
		transCurs2HH, transCurDerIns2[3], transCurDerOuts2[3] = self.currentFun(upperBound[1], upperBound[0], 2, None)[:3]
		
		transCurs3L, transCurDerIns3[0], transCurDerOuts3[0] = self.currentFun(None, lowerBound[0], 3, None)[:3]
		transCurs3H, transCurDerIns3[1], transCurDerOuts3[1] = self.currentFun(None, upperBound[0], 3, None)[:3]
		
		transCurs4LL, transCurDerIns4[0], transCurDerOuts4[0] = self.currentFun(lowerBound[2], lowerBound[0], 4, None)[:3]
		transCurs4LH, transCurDerIns4[1], transCurDerOuts4[1] = self.currentFun(lowerBound[2], upperBound[0], 4, None)[:3]
		transCurs4HL, transCurDerIns4[2], transCurDerOuts4[2] = self.currentFun(upperBound[2], lowerBound[0], 4, None)[:3]
		transCurs4HH, transCurDerIns4[3], transCurDerOuts4[3] = self.currentFun(upperBound[2], upperBound[0], 4, None)[:3]
		
		transCurs5LL, transCurDerIns5[0], transCurDerOuts5[0] = self.currentFun(lowerBound[2], lowerBound[0], 5, None)[:3]
		transCurs5LH, transCurDerIns5[1], transCurDerOuts5[1] = self.currentFun(lowerBound[2], upperBound[0], 5, None)[:3]
		transCurs5LL, transCurDerIns5[2], transCurDerOuts5[2] = self.currentFun(upperBound[2], lowerBound[0], 5, None)[:3]
		transCurs5HH, transCurDerIns5[3], transCurDerOuts5[3] = self.currentFun(upperBound[2], upperBound[0], 5, None)[:3]

		jac[:,:,0] = np.ones((lenV, lenV))*float("inf")
		jac[:,:,1] = -np.ones((lenV, lenV))*float("inf")

		# Figure out min and max node currents
		for i in range(4):
			for j in range(4):
				termVal = -transCurDerOuts4[i] - transCurDerOuts1[j]
				jac[0,0,0] = min(jac[0,0,0], termVal)
				jac[0,0,1] = max(jac[0,0,1], termVal)

		for i in range(4):
			jac[0,1,0] = min(jac[0,1,0], -transCurDerIns1[i])
			jac[0,1,1] = max(jac[0,1,1], -transCurDerIns1[i])

		for i in range(4):
			jac[0,2,0] = min(jac[0,2,0], -transCurDerIns4[i])
			jac[0,2,1] = max(jac[0,2,1], -transCurDerIns4[i])

		for i in range(4):
			for j in range(4):
				termVal = transCurDerOuts1[i] + transCurDerOuts2[j]
				jac[1,0,0] = min(jac[1,0,0], termVal)
				jac[1,0,1] = max(jac[1,0,1], termVal)

		for i in range(2):
			for j in range(4):
				for k in range(4):
					termVal = -transCurDerOuts0[i] + transCurDerIns1[j] + transCurDerIns2[k]
					jac[1,1,0] = min(jac[1,1,0], termVal)
					jac[1,1,1] = max(jac[1,1,1], termVal)

		jac[1,2,:] = [0.0, 0.0]

		for i in range(2):
			for j in range(4):
				for k in range(4):
					termVal = -transCurDerOuts3[i] + transCurDerOuts5[j] + transCurDerOuts4[k]
					jac[2,0,0] = min(jac[2,2,0], termVal)
					jac[2,0,1] = max(jac[2,0,1], termVal)

		jac[2,1,:] = [0.0, 0.0]

		for i in range(4):
			for j in range(4):
				termVal = transCurDerIns5[i] + transCurDerIns4[j]
				jac[2,2,0] = min(jac[2,2,0], termVal)
				jac[2,2,1] = max(jac[2,2,1], termVal)

		return jac


	# numerical approximation where i sample jacobian in the patch by bounds
	# might need a more analytical way of solving this later
	'''def jacobianInterval(self,bounds):
		lowerBound = bounds[:,0]
		upperBound = bounds[:,1]
		lenV = len(lowerBound)
		jacSamples = np.zeros((lenV, lenV,8))

		jacSamples[:,:,0] = self.jacobian([lowerBound[0], lowerBound[1], lowerBound[2]])
		jacSamples[:,:,1] = self.jacobian([lowerBound[0], lowerBound[1], upperBound[2]])
		jacSamples[:,:,2] = self.jacobian([lowerBound[0], upperBound[1], lowerBound[2]])
		jacSamples[:,:,3] = self.jacobian([upperBound[0], lowerBound[1], lowerBound[2]])

		jacSamples[:,:,4] = self.jacobian([lowerBound[0], upperBound[1], upperBound[2]])
		jacSamples[:,:,5] = self.jacobian([upperBound[0], lowerBound[1], upperBound[2]])
		jacSamples[:,:,6] = self.jacobian([lowerBound[0], upperBound[1], upperBound[2]])
		jacSamples[:,:,7] = self.jacobian([upperBound[0], upperBound[1], upperBound[2]])

		jac = np.zeros((lenV, lenV, 2))
		jac[:,:,0] = np.ones((lenV, lenV))*float("inf")
		jac[:,:,1] = -np.ones((lenV, lenV))*float("inf")

		for ji in range(8):
			for i in range(lenV):
				for j in range(lenV):
					jac[i,j,0] = min(jac[i,j,0], jacSamples[i,j,ji])
					jac[i,j,1] = max(jac[i,j,1], jacSamples[i,j,ji])

		#print ("jac")
		#print (jac)
		return jac'''

	def linearConstraints(self, hyperRectangle):
		#print ("linearConstraints hyperRectangle")
		#print (hyperRectangle)
		solvers.options["show_progress"] = False
		allConstraints = ""

		# transistor currents
		for i in range(len(self.tIs)):
			if i == 0:
				allConstraints += self.triangleBounds(self.xs[1], self.tIs[i], hyperRectangle[1][0], hyperRectangle[1][1], i)
			elif i == 3:
				allConstraints += self.triangleBounds(self.xs[0], self.tIs[i], hyperRectangle[0][0], hyperRectangle[0][1], i)
			else:
				inIndex, outIndex = 1, 0
				if i == 4 or i == 5:
					inIndex, outIndex = 2, 0
				#print ("transistorNumber", i)
				#print ("inIndex", inIndex, "outIndex", outIndex)
				patch = np.zeros((4,2))
				patch[0,:] = [hyperRectangle[inIndex][0], hyperRectangle[outIndex][0]]
				patch[1,:] = [hyperRectangle[inIndex][1], hyperRectangle[outIndex][0]]
				patch[2,:] = [hyperRectangle[inIndex][1], hyperRectangle[outIndex][1]]
				patch[3,:] = [hyperRectangle[inIndex][0], hyperRectangle[outIndex][1]]
				hyperTooSmall = True
				for fi in range(3):
					diff = np.absolute(patch[fi,:] - patch[fi+1,:])
					#print ("fwdDiff", diff)
					if np.greater(diff, np.ones(diff.shape)*1e-5).any():
						hyperTooSmall = False
						break
				if not(hyperTooSmall):
					constraints = self.ICrossRegConstraint(self.tIs[i], self.xs[inIndex], self.xs[outIndex], patch, i)
					allConstraints += constraints
				else:
					#print ("hyperTooSmall", patch)
					pass
		allConstraints += "1 " + self.nIs[0] + " + " + "1 "+self.tIs[4] + " + " + "1 " + self.tIs[1] + " >= 0\n"
		allConstraints += "1 " + self.nIs[0] + " + " + "1 "+self.tIs[4] + " + " + "1 " + self.tIs[1] + " <= 0\n"

		allConstraints += "1 " + self.nIs[1] + " + " + "1 "+self.tIs[0] + " + " + "-1 " + self.tIs[1] + " + " + "-1 " + self.tIs[2]+" >= 0\n"
		allConstraints += "1 " + self.nIs[1] + " + " + "1 "+self.tIs[0] + " + " + "-1 " + self.tIs[1] + " + " + "-1 " + self.tIs[2]+" <= 0\n"

		allConstraints += "1 " + self.nIs[2] + " + " + "1 "+self.tIs[3] + " + " + "-1 " + self.tIs[5] + " + " + "-1 " + self.tIs[4]+" >= 0\n"
		allConstraints += "1 " + self.nIs[2] + " + " + "1 "+self.tIs[3] + " + " + "-1 " + self.tIs[5] + " + " + "-1 " + self.tIs[4]+" <= 0\n"

		for i in range(3):
			allConstraints += "1 " + self.nIs[i] + " >= 0\n"
			allConstraints += "1 " + self.nIs[i] + " <= 0\n"

		'''allConstraints += "1 " + self.xs[0] + " >= 0.5515\n"
		allConstraints += "1 " + self.xs[0] + " <= 0.5516\n"
		allConstraints += "1 " + self.xs[1] + " >= 0.1185\n"
		allConstraints += "1 " + self.xs[1] + " <= 0.1186\n"
		allConstraints += "1 " + self.xs[2] + " >= 1.4812\n"
		allConstraints += "1 " + self.xs[2] + " <= 1.4813\n"

		#allConstraints += "1 " + self.tIs[0] + " >= 0.1615\n"
		#allConstraints += "1 " + self.tIs[0] + " <= 0.1616\n"
		allConstraints += "1 " + self.tIs[1] + " >= 0.1586\n"
		allConstraints += "1 " + self.tIs[1] + " <= 0.1587\n"
		allConstraints += "1 " + self.tIs[2] + " >= 0.0029\n"
		allConstraints += "1 " + self.tIs[2] + " <= 0.0030\n"
		allConstraints += "1 " + self.tIs[3] + " >= -0.7209\n"
		allConstraints += "1 " + self.tIs[3] + " <= -0.7204\n"
		allConstraints += "1 " + self.tIs[4] + " >= -0.1587\n"
		allConstraints += "1 " + self.tIs[4] + " <= -0.1585\n"
		allConstraints += "1 " + self.tIs[5] + " >= -0.5620\n"
		allConstraints += "1 " + self.tIs[5] + " <= -0.5618\n"'''
		'''allConstraintList = allConstraints.splitlines()
		allConstraints = ""
		for i in range(len(allConstraintList)):
			#if i > 33 or i < 28:
			#if i <= 29:
			allConstraints += allConstraintList[i] + "\n"
		print ("numConstraints ", len(allConstraintList))'''
		#print ("allConstraints")
		#print (allConstraints)
		variableDict, A, B = lpUtils.constructCoeffMatrices(allConstraints)
		#print ("Amat", A)
		#print ("Bmat", B)
		newHyperRectangle = np.copy(hyperRectangle)
		feasible = True
		for i in range(len(self.xs)):
			#print ("min max ", i)
			minObjConstraint = "min 1 " + self.xs[i]
			maxObjConstraint = "max 1 " + self.xs[i]
			Cmin = lpUtils.constructObjMatrix(minObjConstraint,variableDict)
			Cmax = lpUtils.constructObjMatrix(maxObjConstraint,variableDict)
			minSol, maxSol = None, None
			try:
				minSol = solvers.lp(Cmin,A,B)
			except ValueError:
				#print ("weird constraints", allConstraints)
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






