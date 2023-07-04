# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:46:41 2022

@author: Matthieu Nougaret

Second method to create a random racetrak: find the convexe envelope of a
random dots cloud.
"""

import numpy as np
from general import SmoothCircuit, UniqueOrder
#=============================================================================

def JarvisWalk(points):
	"""
	Function to find the convex envelope of a dot cloud. It uses scalar
	product to find the correct angle.
	\math{
	   \theta = arccos(\fraf{x_{v1}*x_{v2}+y_{v1}*y_{v2}}{||v_1||*||v_2||})
	   }

	Parameters
	----------
	points : np.ndarray
		A 2-dimensions numpy array. It countains the positions of the dots.

	Returns
	-------
	sort : np.ndarray
		A 2-dimensions numpy array. It countains the positions of the dots
		forming the convexe envelope.

	"""
	sort = []
	starter = points[points[:, 0] == np.min(points[:, 0])][0]
	# selects the "left most" => dot that has the lowest ordinate
	sort.append(list(starter))
	# u vector
	u = np.array([0, 1])
	current = np.copy(starter)
	stop = False
	# Safe stoping counter
	c = 0
	while stop != True:
		v = points-current
		prod_sca = v[:, 0]*u[0]+v[:, 1]*u[1]
		norme_v = np.sum(v**2, axis=1)**.5
		norme_u = (u[0]**2 + u[1]**2)**.5
		angles = np.arccos(prod_sca/(norme_u*norme_v))
		sort.append(list(points[np.nanargmin(angles)]))
		u = v[np.nanargmin(angles)]
		current = points[np.nanargmin(angles)]
		c += 1
		# Stop when the starter nod is once again reach
		if np.sum(current == starter) == 2:
			stop = True

		# Safe stoping
		if c > (len(points)+10):
			print('')
			print(points)
			print('')
			stop = True

	return sort

def ComplexIt(P, mind, divd=4):
	"""
	Function to break the line witch length is superior to mind.

	Parameters
	----------
	P : np.ndarray
		A 2-dimensions numpy array witch countain the positions of the dots
		forming the convexe envelope.
	mind : int
		Maximum length above which the segments will be broken.
	divd : int|float, optional
		Value that divide the distance between the new position and the curve.
		Higger it will be, lowwer the distance between the new position and
		the curve will be. The default is 4. It have to be > 0.

	Returns
	-------
	Course : np.ndarray
		A 2-dimensions numpy array witch countain the positions of the dots
		forming the racetrack.

	"""
	Course = []
	for i in range(len(P)-1):
		Course.append(list(P[i]))
		d = ((P[i+1, 0]-P[i, 0])**2+(P[i+1, 1]-P[i, 1])**2)**0.5
		if d >= mind:
			r1 = d/2+d/divd ; r2 = d/2+d/divd
			if (P[i+1, 1]-P[i, 1]) != 0:
				a = (-P[i, 0]**2 -P[i, 1]**2 +P[i+1, 0]**2 +P[i+1, 1]**2 +
							  r1**2-r2**2)/(2*(P[i+1, 1]-P[i, 1]))
				d = (P[i+1, 0]-P[i, 0])/(P[i+1, 1]-P[i, 1])
				A = d**2 + 1
				B = -2*P[i, 0] +2*P[i, 1]*d -2*a*d
				C = P[i, 0]**2 +P[i, 1]**2 -2*P[i, 1]*a + a**2 -r1**2
				Delta = B**2 - 4*A*C
				XI1 = (-B-np.sqrt(Delta))/(2*A)
				YI1 = a-XI1*d
				XI2 = (-B+np.sqrt(Delta))/(2*A)
				YI2 = a-XI2*d
			elif ((P[i, 1]-P[i+1, 1])==0) & ((P[i, 0]-P[i+1, 0])!=0):
				XI1 = -(r1**2 -r2**2 -P[i, 0]**2 +P[i+1, 0]**2)/(2*(
					P[i, 0]-P[i+1, 0]))
				XI2 = -(r1**2 -r2**2 -P[i, 0]**2 +P[i+1, 0]**2)/(2*(
					P[i, 0]-P[i+1, 0]))
				A = 1 ; B = -2*P[i, 1]
				C1 = P[i, 1]**2 + (XI1-P[i, 0])**2 -r1**2
				Delta = B**2 - 4*A*C1
				YI1 = (-B-np.sqrt(Delta))/(2*A)
				YI2 = (-B+np.sqrt(Delta))/(2*A)
			sol = [[XI1,YI1], [XI2,YI2]][np.random.randint(0, 2)]
			Course.append(sol)
		else:
			pass
	Course.append(list(P[-1]))
	return Course

def MakeHullCircuit(Size, Nrand, Ninterp, mxcut, pccent, Ddiv, Ks, wid):
	"""
	Function to create random racetrack from the convex envelope of a random
	cloud of dots.

	Parameters
	----------
	Size : int
		Size of the final square map (number of cells).
	Nrand : int
		Number of random dots put on the map.
	Ninterp : int
		Number of dots generated for each segments during interpolation.
	mxcut : int|float
		Length above which the lines are broken.
	pccent : float
		Percentage of centering of the circuit. The stronger the circuit, the
		less the circuit will approach the edges.
	Ddiv : float
		Denominator of adding additional distance.
	Ks : int
		Size of the smoothing core.
	wid : int
		Width of the road in pixel on the final map. wid have to be > 1.

	Returns
	-------
	Course : np.ndarray
		Map of the final racetrack in 3-dimensions numpy.array. The dimensions
		are (high, width, rgb code).

	"""
	Road = np.zeros((Size, Size, 3))
	Road[:, :, 1] = .8
	Pts = np.random.uniform(Size*pccent, Size-Size*pccent, (Nrand, 2))
	Hull = JarvisWalk(Pts)
	Hull = ComplexIt(np.array(Hull), mxcut, Ddiv)
	Hull = np.array(Hull+[Hull[1]])
	Course = SmoothCircuit(Hull, Ninterp, Ks)
	Idn = Course.astype(int)
	if wid%2 == 0:
		lw1 = int(wid/2) ; lw2 = int(wid/2)
	else:
		lw1 = int(wid/2) ; lw2 = int(wid/2)+1
	for i in range(len(Idn)):
		Road[Idn[i, 0]-lw1:Idn[i, 0]+lw2,
			 Idn[i, 1]-lw1:Idn[i, 1]+lw2] = [.4, .4, .4]
	Sear = np.argwhere(Road[:, :, 1] == .8)
	for i in range(len(Sear)):
		area = Road[Sear[i, 0]-2:Sear[i, 0]+3, Sear[i, 1]-2:Sear[i, 1]+3]
		if np.sum(area[:, :, 1] == .4):
			Road[Sear[i, 0], Sear[i, 1]] = [.6, 0, 0]
	Ulin = UniqueOrder(Idn)
	return Road, Ulin
