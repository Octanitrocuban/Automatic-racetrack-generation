# -*- coding: utf-8 -*-
"""
@author: Matthieu Nougaret

General functions that are used by the three methods.
"""
import numpy as np
#=============================================================================
def vectorial_smooth(vector, ks=3):
	"""
	Function to smooth an 1d numpy.array.

	Parameters
	----------
	vector : numpy.ndarray
		1d vector to smooth. The smooth use uniform weight kernel.
	ks : int
		Kernel size. It can be odd or even.

	Raises
	------
	TypeError
		vector is not a numpy nd array.
	TypeError
		ks not an int.

	Returns
	-------
	Smooth : numpy nd array
		Smoothed numpy nd array of vector with the same shape.

	Example
	-------
	[In 0] : vr = np.random.uniform(-1, 1, 10)
	[Out 0]: np.array([ 0.37852728, -0.05498603, 0.16006387, -0.51660395,
					   -0.79552948, -0.36000402, 0.95710168,  0.46257362,
						0.72266915,  0.0813731])

	[In 1] : vectorial_smooth(vr, ks=3)
	[Out 1]: np.array([ 0.16177062,  0.1612017 , -0.13717537, -0.38402318,
					   -0.55737915, -0.06614394,  0.35322376,  0.71411482,
						0.42220529,  0.40202112])

	"""
	if type(vector) != np.ndarray:
		raise TypeError('vector have to be numpy.ndarray type, get : '+
				  str(type(vector)))

	if type(ks) != int:
		raise TypeError('ks have to be an int type, get : '+str(type(ks)))

	kernel = np.arange(ks, dtype=object)
	kernel[0] = None
	kernel_inv = np.copy(kernel)[::-1]
	kernel_inv[:-1] = -kernel_inv[:-1]
	length = len(vector)
	padd = ks-1
	dilat = int((ks-1)/2)
	smooth = np.zeros((length+padd))
	denomi = np.zeros((length+padd))
	for i in range(len(kernel)):
		denomi[kernel[i]:kernel_inv[i]] += 1
		smooth[kernel[i]:kernel_inv[i]] += vector

	smooth = smooth[dilat:-dilat]/denomi[dilat:-dilat]
	if ks%2 == 0:
		length_2 = smooth.shape
		out_pad = length-length_2
		smooth = smooth[:out_pad]

	return smooth

def smooth_circuit(points, n_iter, kernel_size):
	"""
	Function to interpolate and smooth the path. The goal is to remove
	of the road to the angular cut.

	Parameters
	----------
	points : np.ndarray
		2d numpy array that will be smooth.
	n_iter : int
		Number of points generated for each segments during interpolation.
	kernel_size : int
		Size of the smoothing kernel.

	Returns
	-------
	course : np.ndarray
		Smoothed and interpolated path.

	Example
	-------
	[In 0] : path = np.arange(4)[:, np.newaxis]+np.random.rand(4, 2)
	[Out 0]: np.array([[0.90820983, 0.9398832 ], [1.44765178, 1.0790727 ]
					   [2.27609303, 2.14004349], [3.51226993, 3.66886917]]])

	[In 1] : smooth_circuit(path, 5, 3)
	[Out 1]: np.array([[1.1779308 , 1.00947795], [1.31279129, 1.04427533],
					   [1.40269829, 1.06747358], [1.51668855, 1.16748694],
					   [1.65476209, 1.3443154 ], [1.86187241, 1.60955809],
					   [2.06898272, 1.87480079], [2.20705626, 2.05162925],
					   [2.37910777, 2.26744563], [2.58513726, 2.52224991],
					   [2.89418148, 2.90445633]])

	"""
	course = np.zeros((int((len(points)-1)*n_iter), 2))
	for j in range(len(points)-1):
		xj = np.linspace(points[j, 0], points[j+1, 0], n_iter)
		yj = np.linspace(points[j, 1], points[j+1, 1], n_iter)
		course[j*n_iter:(j+1)*n_iter] = np.array([xj, yj]).T

	course[:, 0] = vectorial_smooth(course[:, 0], ks=kernel_size)
	course[:, 1] = vectorial_smooth(course[:, 1], ks=kernel_size)
	course = course[int(n_iter/2):-int(n_iter/2)]
	return course
