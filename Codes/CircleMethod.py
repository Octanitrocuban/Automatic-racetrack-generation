# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:35:24 2022

@author: Matthieu Nougaret

First method to create a random racetrak: break a cicle into curves.
"""

import numpy as np
from general import SmoothCircuit
#=============================================================================

def MakeBroke_circleCircuit(d_angle, Std, Nit, Ks, Size, pcent, wid):
	"""
	Function to creat a random racetrack from a circle.

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
		Size of the smoothing kernel.
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
		lw1 = int(wid/2) ; lw2 = int(wid/2)+1
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