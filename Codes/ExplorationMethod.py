# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 18:00:03 2022

@author:  Matthieu Nougaret

Tird method to create a random racetrak: it explore a 2d np.array, the convert
it to the wrigth size and position.
"""

import numpy as np
import warnings
from general import SmoothCircuit
#=============================================================================

def SelectCroix(Array, x, y):
	"""
	Function to select 4 values on a 2d np.array with a specific structure.

	Parameters
	----------
	Array : np.ndarray
		The 2-dimensions numpy array on witch we want to get the values.
	x : int
		X position of the center of the cross.
	y : TYPE
		Y position of the center of the cross.

	Returns
	-------
	Select : np.ndarray
		A 1-dimension np.array of length 4 in witch we have the wanted values.

	"""
	Select = np.array([Array[x-1,y],Array[x,y-1],Array[x,y+1],Array[x+1,y]])
	return Select

def Polyg(Array, First):
	"""
	Function to cut a polygon into a 2d np.array.

	Parameters
	----------
	Array : np.array
		A 2-dimensions numpy array.
	First : list
		coordinates from where we began to construct the polygon.

	Returns
	-------
	Arr: np.ndarray
		A 2-dimensions numpy array, copy of 'Array', but with a new value
		where the polygon at witch owned the position 'First'.

	"""
	Arr = np.full((Array.shape[0]+2, Array.shape[1]+2), -1)
	Arr[1:-1, 1:-1] = Array
	FillValue = np.max(Arr)+1
	open_liste = [First]
	tmp_open_liste = []
	close_liste = []
	PolygValue = Arr[open_liste[0][0], open_liste[0][1]]
	Arr[open_liste[0][0], open_liste[0][1]] = FillValue
	VarList = [1]
	c = 0
	while len(open_liste) > 0:
		try :
			c+=1
			for n in range(len(open_liste)):
				S = SelectCroix(Arr, open_liste[n][0], open_liste[n][1])
				for i in range(len(S)):
					if S[i] == PolygValue:
						if i == 0:
							xy = [open_liste[n][0]-1, open_liste[n][1]]
						elif i == 1:
							xy = [open_liste[n][0], open_liste[n][1]-1]
						elif i == 2:
							xy = [open_liste[n][0], open_liste[n][1]+1]
						elif i == 3:
							xy = [open_liste[n][0]+1, open_liste[n][1]]
						if xy not in close_liste:
							if xy not in tmp_open_liste:
								tmp_open_liste.append(xy)
								Arr[xy[0], xy[1]] = FillValue
			close_liste.append(open_liste[0])
			open_liste.remove(open_liste[0])
			open_liste = tmp_open_liste.copy()
			tmp_open_liste.remove(tmp_open_liste[0])
			VarList.append(len(open_liste))
			if VarList[c] < VarList[c-1]:
				break
		except:
			break
	return Arr[1:-1, 1:-1]

def Polygonize(Array, verbose=False):
	"""
	Function to find the differents poylgons on a 2d np.array created by
	groups of cells of same values.

	Parameters
	----------
	Array : np.ndarray
		A 2-dimensions numpy array on witch the function will search the
		polygons.
	verbose : bool, optional
		If True, some news on the number of polygones begin created. The
		default is False.

	Returns
	-------
	Y : np.ndarray
		A 2-dimensions numpy array on witch the function have cut out polygons
		by filling them with unique values.
	Polygones : dict
		Dictionary of dictionary. the intern dict countains: {'the value that
		fill the n-th polygon', 'the number of cells forming part of the n-th
		polygon', 'the original value that was filling the n-th polygon'}.

	"""
	if verbose == True:
		print("Creating the polygons map")
		print("polygon 1")
	Y = Polyg(Array, [1, 1])
	FREO = list(np.argwhere(Y == Array)[0]+1)
	if verbose == True:
		print("polygon 2")
	Y = Polyg(Y, FREO)
	c = 3
	while len(FREO) > 0:
		FREO = np.argwhere(Y == Array)
		if len(FREO) == 0:
			break
		if verbose == True:
			print("polygon", c)
		FREO = list(FREO[0]+1)
		Y = Polyg(Y, FREO)
		c += 1
	if verbose == True:
		print("Classifing the polygons created")
	Polygones = {}
	Uniq = np.unique(Y)
	for i in range(len(Uniq)):
		Polygones["P"+str(i+1)] = dict(fill_value=Uniq[i],
								 size=len(Y[Y == Uniq[i]]),
								 OriginVal=Array[Y == Uniq[i]][0] )
	return Y, Polygones

def Exploration(Shape, init):
	"""
	Function to create the base of the racetrack.

	Parameters
	----------
	Shape : tuple|list|np.ndarray
		A 1-dimension vector of length 2. Shape of the map that will be
		randomly explored.
	init : list|np.ndarray
		A 1-dimension vector of length 2. Initial position of the exploration.

	Returns
	-------
	Road : np.ndarray
		A 2-dimensions np.array. They are the positions of the racetrack.

	"""
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
			polyval = Polygonize(PolAr)
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
	Function to create a random racetrack by the exploration of a np.ndarray
	of shape: Shape.

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