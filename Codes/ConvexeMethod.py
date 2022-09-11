# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:46:41 2022

@author: Matthieu Nougaret

Second method to create a random racetrak: find the convexe envelope of a
random dots cloud.
"""

import numpy as np
from general import SmoothCircuit
#=============================================================================

def JarvisWalk(Points):
	"""
	Function to fin the convex envelope of the random cloud.

	Parameters
	----------
	Points : np.ndarray
		A 2-dimensions numpy array. It countains the positions of the dots.

	Returns
	-------
	cop : np.ndarray
		A 2-dimensions numpy array. It countains the positions of the dots
		forming the convexe envelope.

	"""
	E = [] ; cop = [] ; nd = len(Points) ; step = 0
	E.append(list(Points[Points[:, 0] == np.min(Points[:, 0])][0]))
	Stop=False ; verti = E[0][0]
	while Stop != True:
		angles = np.zeros(nd)
		if len(E)== 1:
			for i in range(len(Points)):
				a = ((E[0][0]-verti)**2 +(E[0][1]-verti)**2)**0.5
				b = ((Points[i, 0]-E[0][0])**2 +(Points[i, 1]-E[0][1])**2)**0.5
				c = ((verti-Points[i, 0])**2 +(verti-Points[i, 1])**2)**0.5
				gamma = np.rad2deg(np.arccos(-(c**2 -a**2 -b**2)/(2*a*b)))
				angles[i] = 360-gamma
			Px = np.nanargmin(angles)
			E.append(list(Points[Px]))
		else:
			for i in range(len(Points)):
				a = ((E[step][0]-E[step-1][0])**2 +
					 (E[step][1]-E[step-1][1])**2)**0.5
				b = ((Points[i, 0]-E[step][0])**2 +
					 (Points[i, 1]-E[step][1])**2)**0.5
				c = ((E[step-1][0]-Points[i, 0])**2 +
					 (E[step-1][1]-Points[i, 1])**2)**0.5
				gamma = np.rad2deg(np.arccos(-(c**2 -a**2 -b**2)/(2*a*b)))
				angles[i] = 360-gamma
			Px = np.arange(nd)[angles < 360]
			Px = Px[np.nanargmin(angles[angles < 360])]
			E.append(list(Points[Px]))
		if E[step] in cop:
			cop.append(E[step]) ; Stop = True
		else:
			cop.append(E[step]) ; step += 1
	return cop

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
		area = Road[Sear[i, 0]-lw1:Sear[i, 0]+lw2,
					Sear[i, 1]-lw1:Sear[i, 1]+lw2]
		if np.sum(area[:, :, 1] == .4):
			Road[Sear[i, 0], Sear[i, 1]] = [.6, 0, 0]
	return Road
