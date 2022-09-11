# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 21:43:06 2022

@author: Matthieu Nougaret

Some general functions.

"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
#=============================================================================

def VectorialSmooth(Vect, ks=3):
	"""
	Function to smooth an 1d numpy.array.

	Parameters
	----------
	Vect : 1d numpy array
		1d vector to smooth.
	ks : int
		Kernel size.

	Raises
	------
	TypeError
		Vect is not a numpy nd array.
	TypeError
		ks not an int.

	Returns
	-------
	Smooth : numpy nd array
		Smoothed numpy nd array of Vect with the same shape.

	"""
	if type(Vect) != np.ndarray:
		raise TypeError('Vect have to be numpy.ndarray type, get : '+
				  str(type(Vect)))
	if type(ks) != int:
		raise TypeError('ks have to be an int type, get : '+str(type(ks)))

	lr = np.arange(ks, dtype=object)
	lr[0] = None
	ld = np.copy(lr)[::-1]
	ld[:-1] = -ld[:-1]
	Hi, add, lm = len(Vect), ks-1, int((ks-1)/2)
	Smooth = np.zeros((Hi+add))
	Denomi = np.zeros((Hi+add))
	for i in range(len(lr)):
		Denomi[lr[i]:ld[i]] += 1
		Smooth[lr[i]:ld[i]] += Vect
	Smooth = Smooth[lm:-lm]/Denomi[lm:-lm]
	if ks%2 == 0:
		Hi2 = Smooth.shape
		gh = Hi-Hi2
		Smooth = Smooth[:gh]
	return Smooth

def SmoothCircuit(Points, Nit, Ks):
	"""
	Function to smooth the path of the road to remoove the angular cut.

	Parameters
	----------
	Points : np.ndarray
		2d numpy array that will be smooth.
	Nit : int
		Number of points generated for each segments during interpolation.
	Ks : int
		Size of the smoothing kernel.

	Returns
	-------
	Course : np.ndarray
		DESCRIPTION.

	"""
	Course = np.zeros((int((len(Points)-1)*Nit), 2))
	for j in range(len(Points)-1):
		xj = np.linspace(Points[j, 0], Points[j+1, 0], Nit)
		yj = np.linspace(Points[j, 1], Points[j+1, 1], Nit)
		Course[j*Nit:(j+1)*Nit] = np.array([xj, yj]).T
	Course[:, 0] = VectorialSmooth(Course[:, 0], ks=Ks)
	Course[:, 1] = VectorialSmooth(Course[:, 1], ks=Ks)
	Course = Course[int(Nit/2):-int(Nit/2)]
	return Course
