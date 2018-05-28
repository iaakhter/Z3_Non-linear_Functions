from circuit import *
import numpy as np

nfet = MosfetModel('nfet', 0.4, 270.0e-6, 3.0)
m1 = Mosfet(0, 1, 2, nfet)
v = np.zeros(3)
vg = np.zeros(19)
ids = np.zeros(len(vg))

v[0] = 0.0;
v[2] = 1.8;
for i in range(len(vg)):
	vg[i] = 1.8*i/(len(vg)-1)
	v[1] = vg[i]
	ids[i] = m1.ids(v)

# try printing ids and see if it looks reasonable
print 'ids_n, vs=0, vd=1.8, sweep vg 0 to 1.8 step 0.1, ids_n = ' + str(ids)

# try a pfet
pfet = MosfetModel('pfet', -0.4, -90.0e-6, 6.0)
m2 = Mosfet(3, 1, 2, pfet)
v = np.array([0.0, 0.0, 0.0, 1.8])
for i in range(len(vg)):
	v[1] = vg[i]
	ids[i] = m2.ids(v)

# try printing ids and see if it looks reasonable
print 'ids_p, vs=1.8, vd=0.0, sweep vg 0 to 1.8 step 0.1, ids_p = ' + str(ids)

# try an inverter
c = Circuit([m1, m2])
v = np.array([0.0, 0.0, 0.9, 1.8])
vin = vg
i_out = np.zeros(len(vg))
for i in range(len(vin)):
	v[1] = vin[i]
	i_out[i] = c.f(v)[2]

print 'Vout = ' + str(v[2]) + ': i_out = ' + str(i_out)
v[2] = 0.0
for i in range(len(vin)):
	v[1] = vin[i]
	i_out[i] = c.f(v)[2]

print 'Vout = ' + str(v[2]) + ': i_out = ' + str(i_out)
v[2] = 1.8
for i in range(len(vin)):
	v[1] = vin[i]
	i_out[i] = c.f(v)[2]

print 'Vout = ' + str(v[2]) + ': i_out = ' + str(i_out)

# need to test the grad_ids code
v = np.array([0.0, 0.8, 1.3, 1.8])
m1.grad_ids(v)
m2.grad_ids(v)

# The two examples above didn't crash.  Let's try some more cases and 
#   compare with the values numerical differencing
#######   gradients for the nfet
#	derivatives wrt Vs
delta = 0.01
for VsInt in range(0, 19, 3):
	Vs = VsInt/10.0
	for VgInt in range(0, 19, 3):
		Vg = VgInt/10.0
		for VdInt in range(0, 19, 3):
			Vd = VdInt/10.0
			v = np.array([Vs, Vg, Vd, 1.8])
			Ids = m1.ids(v)
			g = m1.grad_ids(v)
			dvs1 = np.array([Vs-(delta/2.0), Vg, Vd, 1.8])
			dvs2 = np.array([Vs+(delta/2.0), Vg, Vd, 1.8])
			ds = (m1.ids(dvs2) - m1.ids(dvs1))/delta
			if(abs(ds-g[0]) > 1e-2*abs(g[0])):
				print 'oops: v = ' + str(v) + ', g[0] = ' + str(g[0]) + ', ds = ' + str(ds)

#   derivates wrt Vg
delta = 0.01
for VsInt in range(0, 19, 3):
	Vs = VsInt/10.0
	for VgInt in range(0, 19, 3):
		Vg = VgInt/10.0
		for VdInt in range(0, 19, 3):
			Vd = VdInt/10.0
			v = np.array([Vs, Vg, Vd, 1.8])
			Ids = m1.ids(v)
			g = m1.grad_ids(v)
			dvg1 = np.array([Vs, Vg-(delta/2.0), Vd, 1.8])
			dvg2 = np.array([Vs, Vg+(delta/2.0), Vd, 1.8])
			dg = (m1.ids(dvg2) - m1.ids(dvg1))/delta
			if(abs(dg-g[1]) > 1e-2*abs(g[1])):
				print 'oops: v = ' + str(v) + ', g[1] = ' + str(g[1]) + ', dg = ' + str(dg)

#   derivates wrt Vd
delta = 0.01
for VsInt in range(0, 19, 3):
	Vs = VsInt/10.0
	for VgInt in range(0, 19, 3):
		Vg = VgInt/10.0
		for VdInt in range(0, 19, 3):
			Vd = VdInt/10.0
			v = np.array([Vs, Vg, Vd, 1.8])
			Ids = m1.ids(v)
			g = m1.grad_ids(v)
			dvd1 = np.array([Vs, Vg, Vd-(delta/2.0), 1.8])
			dvd2 = np.array([Vs, Vg, Vd+(delta/2.0), 1.8])
			dd= (m1.ids(dvd2) - m1.ids(dvd1))/delta
			if(abs(dd-g[2]) > 1e-2*abs(g[2])):
				print 'oops: v = ' + str(v) + ', g[2] = ' + str(g[2]) + ', dd = ' + str(dd)

#######   gradients for the pfet
#    derivatives wrt Vs
delta = 0.01
for VsInt in range(0, 19, 3):
	Vs = VsInt/10.0
	for VgInt in range(0, 19, 3):
		Vg = VgInt/10.0
		for VdInt in range(0, 19, 3):
			Vd = VdInt/10.0
			#v = np.array([Vs, Vg, Vd, 1.8])
			v = np.array([1.8, Vg, Vd, Vs])
			Ids = m2.ids(v)
			g = m2.grad_ids(v)
			dvs1 = np.array([1.8, Vg, Vd, Vs-(delta/2.0)])
			dvs2 = np.array([1.8, Vg, Vd, Vs+(delta/2.0)])
			ds = (m2.ids(dvs2) - m2.ids(dvs1))/delta
			if(abs(ds-g[0]) > 1e-2*abs(g[0])):
				print 'oops: v = ' + str(v) + ', g[0] = ' + str(g[0]) + ', ds = ' + str(ds)

#   derivates wrt Vg
delta = 0.01
for VsInt in range(0, 19, 3):
	Vs = VsInt/10.0
	for VgInt in range(0, 19, 3):
		Vg = VgInt/10.0
		for VdInt in range(0, 19, 3):
			Vd = VdInt/10.0
			#v = np.array([Vs, Vg, Vd, 1.8])
			v = np.array([1.8, Vg, Vd, Vs])
			Ids = m2.ids(v)
			g = m2.grad_ids(v)
			dvg1 = np.array([1.8, Vg-(delta/2.0), Vd, Vs])
			dvg2 = np.array([1.8, Vg+(delta/2.0), Vd, Vs])
			dg = (m2.ids(dvg2) - m2.ids(dvg1))/delta
			if(abs(dg-g[1]) > 1e-2*abs(g[1])):
				print 'oops: v = ' + str(v) + ', g[1] = ' + str(g[1]) + ', dg = ' + str(dg)

#   derivates wrt Vd
delta = 0.01
for VsInt in range(0, 19, 3):
	Vs = VsInt/10.0
	for VgInt in range(0, 19, 3):
		Vg = VgInt/10.0
		for VdInt in range(0, 19, 3):
			Vd = VdInt/10.0
			#v = np.array([Vs, Vg, Vd, 1.8])
			v = np.array([1.8, Vg, Vd, Vs])
			Ids = m2.ids(v)
			g = m2.grad_ids(v)
			dvd1 = np.array([1.8, Vg, Vd-(delta/2.0), Vs])
			dvd2 = np.array([1.8, Vg, Vd+(delta/2.0), Vs])
			dd = (m2.ids(dvd2) - m2.ids(dvd1))/delta
			if(abs(dd-g[2]) > 1e-2*abs(g[2])):
				print 'oops: v = ' + str(v) + ', g[2] = ' + str(g[2]) + ', dd = ' + str(dd)

#####   gradients when Vs, Vg, and/or Vd are intervals
#######   gradients for the nfet
#	derivatives wrt Vs
delta = 0.01
for VsInt in range(0, 16, 3):
	Vs = np.array([VsInt/10.0, (VsInt + 3)/10.0])
	for VgInt in range(0, 16, 3):
		Vg = np.array([VgInt/10.0, (VgInt + 3)/10.0])
		for VdInt in range(0, 16, 3):
			Vd = np.array([VdInt/10.0, (VdInt + 3)/10.0])
			v = np.array([Vs, Vg, Vd, 1.8])
			Ids = m1.ids(v)
			g = m1.grad_ids(v)
			#print "g", g

			sampleDelta = 0.0001
			vsSamples = np.linspace(Vs[0]+sampleDelta, Vs[1]-sampleDelta, 10)
			vgSamples = np.linspace(Vg[0]+sampleDelta, Vg[1]-sampleDelta, 10)
			vdSamples = np.linspace(Vd[0]+sampleDelta, Vd[1]-sampleDelta, 10)

			for vs in vsSamples:
				for vg in vgSamples:
					for vd in vdSamples:
						dvs1 = np.array([vs-(delta/2.0), vg, vd, 1.8])
						dvs2 = np.array([vs+(delta/2.0), vg, vd, 1.8])
						ds = (m1.ids(dvs2) - m1.ids(dvs1))/delta
						#if(abs(ds-g[0]) > 1e-2*abs(g[0])):
						if(((ds < g[0][0]) or (ds > g[0][1])) and not (tiny_p(ds - g[0][0]) or tiny_p(ds - g[0][1]))):
							print 'oops gradient interval for v = ' + str(v) +', with sample v ' + str([vs, vg, vd]) + ', g[0] = ' + str(g[0]) + ', ds = ' + str(ds)
							print 'ds - g[0][0] = ' + str(ds - g[0][0]) + ', tiny_p(ds - g[0][0]) = ' + str(tiny_p(ds - g[0][0]))
							print 'ds - g[0][1] = ' + str(ds - g[0][1]) + ', tiny_p(ds - g[0][1]) = ' + str(tiny_p(ds - g[0][1]))
