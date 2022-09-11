# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 21:43:06 2022

@author: Matthieu Nougaret

This script countain functions to create random racetrack.

(1)http://blog.meltinglogic.com/2013/12/how-to-generate-procedural-racetracks/

"""
import numpy as np
import matplotlib.pyplot as plt
import modulenougaret.FonctionsCours336 as fc3
import modulenougaret.FonctionPersos as fp
import modulenougaret.Aleatoire as al
import warnings
#=============================================================================
def SmoothCircuit(Points, Nit, Ks):
	Course = np.zeros((int((len(Points)-1)*Nit), 2))
	for j in range(len(Points)-1):
		xj = np.linspace(Points[j, 0], Points[j+1, 0], Nit)
		yj = np.linspace(Points[j, 1], Points[j+1, 1], Nit)
		Course[j*Nit:(j+1)*Nit] = np.array([xj, yj]).T
	Course[:, 0] = fp.VectorialSmooth(Course[:, 0], ks=Ks)
	Course[:, 1] = fp.VectorialSmooth(Course[:, 1], ks=Ks)
	Course = Course[int(Nit/2):-int(Nit/2)]
	return Course

def MakeBroke_circleCircuit(d_angle, Std, Nit, Ks, Size, pcent, wid):
	"""
	Parameters
	----------
	d_angle : int|float
		Spacing in degrees between each circle break.
	Std : int|float
		Standard deviation of the centered normal noise that changes the
		distance to the center of the initial points that make up the polygon.
	Nit : int
		Number of points generated for each segments during interpolation.
	Ks : int
		Size of the smoothing core.
	Size : int
		Size of the final square map (number of cells).
	pcent : float
		Percentage of centering of the circuit. The stronger the circuit, the
		less the circuit will approach the edges.
	wid : int
		Width of the road of the racetrack. wid have to be > 1.

	Returns
	-------
	Color : np.ndarray
		Map of the final racetrack in 3-dimensions numpy.array. The dimensions
		are (high, width, rgb code).
	"""
	Teta = np.deg2rad(np.arange(0, 360, d_angle))
	R = 1+np.random.normal(0, Std, len(Teta))
	Coo = np.array([np.cos(Teta)*R, np.sin(Teta)*R]).T
	Coo2 = np.concatenate((Coo, Coo[:2]))
	Course = SmoothCircuit(Coo2, Nit, Ks)
	Circ = np.zeros((Size, Size)) ; I = np.copy(Course)
	I[:, 0] = (I[:,0]-np.min(I[:,0]))/np.max(I[:,0]-np.min(I[:,0]))
	I[:, 1] = (I[:,1]-np.min(I[:,1]))/np.max(I[:,1]-np.min(I[:,1]))
	I[:, 0] = (I[:, 0]*int(Size-Size*pcent))+int(Size*(pcent/2))
	I[:, 1] = (I[:, 1]*int(Size-Size*pcent))+int(Size*(pcent/2))
	I = I.astype(int)
	if wid%2 == 0:
		lw1 = int(wid/2) ; lw2 = int(wid/2)
	else:
		lw1 = int(wid/2)+1 ; lw2 = int(wid/2)
	for u in range(len(I)):
		Circ[I[u, 0]-lw1:I[u, 0]+lw2, I[u, 1]-lw1:I[u, 1]+lw2] = 1
	Color = np.zeros((Size, Size, 3))
	Color[Circ == 0] = [0, 0.8, 0]
	Color[Circ == 1] = [.4, .4, .4]
	lok = np.argwhere(Circ == 0)
	for i in range(len(lok)):
		area = Circ[lok[i, 0]-lw1:lok[i, 0]+lw2, lok[i, 1]-lw1:lok[i, 1]+lw2]
		if np.sum(area) > 0:
			Color[lok[i, 0], lok[i, 1]] = [.6, 0, 0]
	return Color

def JarvisWalk(Points):
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

def ComplexIt_1(P, mind, divd=4):
	Course = []
	for i in range(len(P)-1):
		Course.append(list(P[i]))
		d = ((P[i+1, 0]-P[i, 0])**2+(P[i+1, 1]-P[i, 1])**2)**0.5
		if d >= mind:
			dx = (P[i+1, 0]+P[i, 0])/2 ; dy = (P[i+1, 1]+P[i, 1])/2
			a = (P[i+1, 1]-P[i, 1])/(P[i+1, 0]-P[i, 0])
			b = P[i, 1]-a*P[i, 0] ; acut = -a
			bcut = dy-acut*dx
			A = 1+acut**2
			B = 2*(-dx+acut*bcut-acut*dy)
			C = dx**2+bcut**2+dy**2-2*bcut*dy-(d/divd)**2
			delta = B**2-4*A*C
			sol = [(-B-np.sqrt(delta))/(2*A), (-B+np.sqrt(delta))/(2*A)]
			nwx = sol[np.random.randint(0, 2)]
			nwy = nwx*acut+bcut
			Course.append([nwx, nwy])
		else:
			pass
	Course.append(list(P[-1]))
	return Course

def ComplexIt_2(P, mind, divd=4):
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
	#Jarvis
	Hull = JarvisWalk(Pts)
	Hull = ComplexIt_2(np.array(Hull), mxcut, Ddiv)
	Hull = np.array(Hull+[Hull[1]])
	Course = SmoothCircuit(Hull, Ninterp, Ks)
	Idn = Course.astype(int)
	if wid%2 == 0:
		lw1 = int(wid/2) ; lw2 = int(wid/2)
	else:
		lw1 = int(wid/2)+1 ; lw2 = int(wid/2)
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

def Exploration(Shape, init):
	Reca = np.zeros((Shape[0]+2, Shape[1]+2))-1
	Reca[1:-1, 1:-1] = 0 ; Road = []
	initR = np.copy(init)+1
	Reca[initR[0], initR[1]] = 1
	var = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])
	Road.append(list(np.copy(initR)))
	Stop = False
	while Stop != True:
		news = initR+var
		news = news[(news[:, 0] > 0)&(news[:, 0] < Shape[0]+1)&
					(news[:, 1] > 0)&(news[:, 1] < Shape[1]+1)]
		value = Reca[news[:, 0], news[:, 1]]
		news = news[value == 0]
		if len(news) == 0:
			Reca[initR[0], initR[1]] = 1
			Stop = True
		else:
			PolAr = np.copy(Reca)
			PolAr[init[0]+1, init[1]+1] = 0
			polyval = al.Polygonize(PolAr)
			if len(polyval[1]) > 3:
				IniAreaVal = polyval[0][init[0]+1, init[1]+1]
				vpolval = polyval[0][news[:, 0], news[:, 1]]
				news = news[vpolval == IniAreaVal]
			if len(news) == 0:
				Reca[initR[0], initR[1]] = 1
				Stop = True
			else:
				r = np.random.randint(0, len(news))
				initR = news[r]
				Road.append(list(np.copy(initR)))
				Reca[initR[0], initR[1]] = 1
	Road.append(list(np.copy(init)+1))
	Road.append(Road[1])
	Road = np.array(Road)-1
	return Road

def MakeExplorerCircuit(Shape, init, Nit, Ks, Size, wid):
	"""
	Parameters
	----------
	Shape : tuple|list|np.ndarray
		Shape of the map that will be used for the exploration stage. It have
		to be a 1-dimension vector of length 2.
	init : list|np.ndarray
		Initial position during the eploration stage. It have to be a 
		1-dimension vector of length 2.
	Nit : int
		Number of dots that are adds during the smothing of th path. It have
		to be > 2.
	Ks : int
		Size of the smoothing core.
	Size : int
		Size of the final square map (number of cells).
	wid : int
		Width of the road in pixel on the final map. wid have to be > 1.

	Returns
	-------
	Color : np.ndarray
		Map of the final racetrack in 3-dimensions numpy.array. The dimensions
		are (high, width, rgb code).

	"""
	Road = Exploration(Shape, init)
	Course = SmoothCircuit(Road, Nit, Ks)
	Circ = np.zeros((Size, Size))
	I = np.copy(Course)
	I[: ,0] = I[: ,0]/Shape[0] - np.min(I[: ,0]/Shape[0])
	I[:, 1] = I[:, 1]/Shape[1] - np.min(I[:, 1]/Shape[1])
	I[: ,0] = (I[: ,0]+(1-np.max(I[: ,0]))/2)*Size
	I[: ,1] = (I[: ,1]+(1-np.max(I[: ,1]))/2)*Size
	I = I.astype(int)
	if wid%2 == 0:
		lw1 = int(wid/2) ; lw2 = int(wid/2)
	else:
		lw1 = int(wid/2)+1 ; lw2 = int(wid/2)
	for u in range(len(I)):
		Circ[I[u, 0]-lw1:I[u, 0]+lw2, I[u, 1]-lw1:I[u, 1]+lw2] = 1
	Color = np.zeros((Size, Size, 3))
	Color[Circ == 0] = [0, 0.8, 0]
	Color[Circ == 1] = [.4, .4, .4]
	lok = np.argwhere(Circ == 0)
	for i in range(len(lok)):
		area = Circ[lok[i, 0]-2:lok[i, 0]+3, lok[i, 1]-2:lok[i, 1]+3]
		if np.sum(area) > 0:
			Color[lok[i, 0], lok[i, 1]] = [.6, 0, 0]
	return Color
#=============================================================================

way = [3]

if 1 in way:
	for i in range(10):
		Road = MakeBroke_circleCircuit(20, 0.4, 100, 81, 250, 0.1, 4)
		plt.figure(figsize=(7, 7))
		plt.imshow(Road)
		plt.show()

#http://blog.meltinglogic.com/2013/12/how-to-generate-procedural-racetracks/
if 2 in way:
	warnings.filterwarnings("ignore")
	for cir in range(10):
		
		Road = MakeHullCircuit(250, 15, 100, 50, 0.18, 12, 67, 4)
		plt.figure(figsize=(7, 7))
		plt.imshow(Road)
		plt.show()

if 3 in way:
	for cir in range(10):
		Forme = [6, 6] ; Sz = 250 ; larg = 5 ; ni = 100 ; ks = 103
		Depart = np.array([np.random.randint(0, Forme[0]),
						   np.random.randint(0, Forme[1])])
		smt = MakeExplorerCircuit(Forme, Depart, ni, ks, Sz, larg)

		plt.figure(figsize=(6, 6))
		plt.imshow(smt)
		plt.show()