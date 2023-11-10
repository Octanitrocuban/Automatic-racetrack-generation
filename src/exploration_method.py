# -*- coding: utf-8 -*-
"""
@author:  Matthieu Nougaret

Third method to create a random racetrak: it explore a 2d np.array, the
convert it to the wrigth size and position.
"""

import numpy as np
import warnings
from general import smooth_circuit
#=============================================================================
def polyg_posi_vec_table(array, start_posi):
	"""
	Function to have a representation map to select a polygon on a 2
	dimensionals numpy.ndarray.

	Parameters
	----------
	array : numpy.ndarray
		The 2 dimensionals numpy.ndarray to explore.
	start_posi : numpy.ndarray
		Starting position of the exploration. It must have the following
		shape np.array([[xi, yi]]).

	Returns
	-------
	repre_map : numpy.ndarray
		A 2 dimensionals numpy.ndarray in which the detected polygon
		correspond at the cells which have the heigher value of the array.

	Exemple
	-------
	In [0] : _
	Out [0]: array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
					 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
					 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
					 [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
					 [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
	In [1] : polyg_posi_vec_table(_, np.array([[1, 9]]))
	Out [1]: array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1],
					 [1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1],
					 [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
					 [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
					 [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1],
					 [1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
					 [1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
					 [1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 1, 1, 1],
					 [1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1],
					 [1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					 [1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	shp = array.shape
	repre_map = np.copy(array)
	posi = np.copy(start_posi)
	# value at the starting cell which will be targeted for clipping
	v = repre_map[posi[:, 0], posi[:, 1]]
	# value that will be usedd to indicate the cells selected as being
	# part of the current clipped polygon
	vfil = np.max(array)+1
	stop = False
	while stop != True:
		# stored position are used to fill the cell
		repre_map[posi[:, 0], posi[:, 1]] = vfil
		# select neigboor cells
		posi = np.array([[posi[:, 0]-1, posi[:, 1]  ],
						 [posi[:, 0]+1, posi[:, 1]  ],
						 [posi[:, 0]  , posi[:, 1]-1],
						 [posi[:, 0]  , posi[:, 1]+1]])

		# reshape them
		posi = np.concatenate(posi, axis=1).T
		# to avoid exponentional repetition
		# to keep only existing cell
		posi = posi[(posi[:, 0] >= 0)&(posi[:, 1] >= 0)&(
					 posi[:, 0] < shp[0])&(posi[:, 1] < shp[1])]

		# to keep unexplored cell which are part of the clipping polygon
		posi = posi[repre_map[posi[:, 0], posi[:, 1]] == v]
		# to have non repetition of the position
		posi = np.unique(posi, axis=0)

		# stop when it can not found any other connected cell with the
		# starting value i.e. it had clipped the current polygon
		if len(posi) == 0:
			stop = True

	return repre_map

def polygonize(array):
	"""
	Function to find the differents poylgons on a 2d np.array created by
	groups of cells of same values.

	Parameters
	----------
	array : np.ndarray
		A 2-dimensions numpy array on witch the function will search the
		polygons.

	Returns
	-------
	first_ground : np.ndarray
		A 2-dimensions numpy array on witch the function have cut out
		polygons by filling them with unique values.

	Example
	-------
	In [0] : _
	Out [0]: array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
					 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
					 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
					 [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
					 [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
	In [1] : polygonize(_)
	Out [1]: array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
					 [2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2],
					 [2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2],
					 [2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
					 [2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2],
					 [2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2],
					 [2, 2, 3, 3, 3, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2],
					 [2, 2, 3, 3, 3, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2],
					 [2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 5, 2, 2, 2],
					 [2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 5, 5, 5, 2, 2],
					 [2, 2, 2, 3, 3, 3, 2, 2, 2, 5, 5, 5, 5, 2, 2],
					 [2, 2, 3, 3, 3, 3, 2, 2, 2, 5, 5, 5, 5, 2, 2],
					 [2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 5, 5, 2, 2, 2],
					 [2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
					 [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])

	"""
	first_ground = polyg_posi_vec_table(array, np.array([[0, 0]]))
	next_p = np.argwhere(first_ground == array)
	while len(next_p) > 0:
		next_p = np.array([next_p[0]])
		first_ground = polyg_posi_vec_table(first_ground, next_p)
		# first_ground == array will return a boolean 2d np.ndarray where
		# the cells beeing equal to True mean that they were not explore
		# yet by the polyg_posi_vec_table function.
		next_p = np.argwhere(first_ground == array)

	return first_ground

def exploration(shape, init):
	"""
	Function to create the base of the racetrack.

	Parameters
	----------
	shape : tuple
		A 1-dimension vector of length 2. Shape of the map that will be
		randomly explored.
	init : list|np.ndarray
		A 1-dimension vector of length 2. Initial position of the
		exploration.

	Returns
	-------
	Road : np.ndarray
		A 2-dimensions np.array. They are the positions of the racetrack.

	Example
	-------
	In [0] : exploration((5, 5), np.array([1, 1]))
	Out [0]: array([[1, 1], [0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4],
					[3, 4], [3, 3], [4, 3], [4, 2], [3, 2], [3, 1], [3, 0],
					[2, 0], [2, 1], [1, 1], [0, 1]])

	"""
	recadre = np.zeros((shape[0]+2, shape[1]+2))-1
	recadre[1:-1, 1:-1] = 0
	road = []
	init_rec = np.copy(init)+1
	recadre[init_rec[0], init_rec[1]] = 1
	var = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])
	road.append(list(np.copy(init_rec)))
	stop = False
	while stop != True:
		news = init_rec+var
		news = news[(news[:, 0] > 0)&(news[:, 0] < shape[0]+1)&
					(news[:, 1] > 0)&(news[:, 1] < shape[1]+1)]

		value = recadre[news[:, 0], news[:, 1]]
		news = news[value == 0]
		if len(news) == 0:
			recadre[init_rec[0], init_rec[1]] = 1
			stop = True
		else:
			pol_ar = np.copy(recadre)
			pol_ar[init[0]+1, init[1]+1] = 0
			polyval = polygonize(pol_ar)
			# since the map we want to tessel is filled with 0 and 1, the
			# minimum of polyval will allways be between 1 and 2.
			if np.max(polyval) > 4:
				# if the plate was cut in area, to take a path that will
				# lead to the start and not end at an isolated position.
				ini_area_valid = polyval[init[0]+1, init[1]+1]
				vpolval = polyval[news[:, 0], news[:, 1]]
				# re-define news
				news = news[vpolval == ini_area_valid]

			if len(news) == 0:
				recadre[init_rec[0], init_rec[1]] = 1
				stop = True
			else:
				r = np.random.randint(0, len(news))
				init_rec = news[r]
				road.append(list(np.copy(init_rec)))
				recadre[init_rec[0], init_rec[1]] = 1

	road.append(list(np.copy(init)+1))
	road.append(road[1])
	road = np.array(road)-1
	return road

def make_explorer_circuit(shape, init, n_iter, kernel_size, width_map,
						  width_road):
	"""
	Function to create a random racetrack by the exploration of a np.ndarray
	of shape: shape.

	Parameters
	----------
	shape : tuple
		Shape of the map that will be used for the exploration stage. It have
		to be a 1-dimension vector of length 2.
	init : list|np.ndarray
		Initial position during the eploration stage. It have to be a 
		1-dimension vector of length 2.
	n_iter : int
		Number of dots that are adds during the smothing of th path. It have
		to be > 2.
	kernel_size : int
		Size of the smoothing core.
	width_map : int
		Size of the final square map (number of cells).
	width_road : int
		Width of the road in pixel on the final map. wid have to be > 1.

	Returns
	-------
	map_color : np.ndarray
		Map of the final racetrack in 3-dimensions numpy.array. The
		dimensions are (high, width, rgb code).

	Example
	-------
	In [0] : make_explorer_circuit((6, 6), np.array([1, 1]), 100, 103, 250, 5)

	"""
	road = exploration(shape, init)
	course = smooth_circuit(road, n_iter, kernel_size)
	circuit = np.zeros((width_map, width_map))
	course[: ,0] = course[: ,0]/shape[0] - np.min(course[: ,0]/shape[0])
	course[:, 1] = course[:, 1]/shape[1] - np.min(course[:, 1]/shape[1])
	course[: ,0] = (course[: ,0]+(1-np.max(course[: ,0]))/2)*width_map
	course[: ,1] = (course[: ,1]+(1-np.max(course[: ,1]))/2)*width_map
	course = course.astype(int)
	if width_road%2 == 0:
		lw1 = width_road//2
		lw2 = width_road//2
	else:
		lw1 = width_road//2 + 1
		lw2 = width_road//2

	for u in range(len(course)):
		circuit[course[u, 0]-lw1:course[u, 0]+lw2,
			course[u, 1]-lw1:course[u, 1]+lw2] = 1

	map_color = np.zeros((width_map, width_map, 3))
	map_color[circuit == 0] = [0, 0.8, 0]
	map_color[circuit == 1] = [.4, .4, .4]
	lok = np.argwhere(circuit == 0)
	for i in range(len(lok)):
		area = circuit[lok[i, 0]-2:lok[i, 0]+3, lok[i, 1]-2:lok[i, 1]+3]
		if np.sum(area) > 0:
			map_color[lok[i, 0], lok[i, 1]] = [.6, 0, 0]

	return map_color
