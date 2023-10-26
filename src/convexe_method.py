# -*- coding: utf-8 -*-
"""
@author: Matthieu Nougaret

Second method to create a random racetrak: find the convexe envelope of a
random dots cloud. I took inspiration from the blog post:
https://www.gamedeveloper.com/programming/generating-procedural-racetracks
"""

import numpy as np
from general import smooth_circuit
#=============================================================================
def jarvis_walk(points):
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

	Example
	-------
	[In 0] : vr = np.random.uniform(0, 1, (10, 2))
	[Out 0]: np.array([[0.2086457 , 0.1112382 ], [0.26972164, 0.73368871],
					   [0.49775603, 0.31530644], [0.01461559, 0.39027198],
					   [0.31238011, 0.18289725], [0.2184884 , 0.76162714],
					   [0.87422724, 0.47360825], [0.7539203 , 0.27809264],
					   [0.73860496, 0.44780447], [0.72215823, 0.40458717]])

	[In 1] : jarvis_walk(vr)
	[Out 1]: [[0.01461559, 0.39027198], [0.2184884 , 0.76162714],
			  [0.87422724, 0.47360825], [0.7539203 , 0.27809264],
			  [0.2086457 , 0.1112382 ], [0.01461559, 0.39027198]]

	"""
	sort = []
	starter = points[points[:, 0] == np.min(points[:, 0])][0]
	# selects the "left most" => dot that has the lowest ordinate
	sort.append(list(starter))
	# u vector
	u = np.array([0, 1])
	current = np.copy(starter)
	stop = False
	while stop != True:
		v = points-current
		# scalar product
		prod_sca = v[:, 0]*u[0]+v[:, 1]*u[1]
		# vector norme
		norme_v = np.sum(v**2, axis=1)**.5
		norme_u = (u[0]**2 + u[1]**2)**.5
		angles = np.arccos(prod_sca/(norme_u*norme_v))
		# store the dot that create the smallest angle
		sort.append(list(points[np.nanargmin(angles)]))
		u = v[np.nanargmin(angles)]
		current = points[np.nanargmin(angles)]
		# Stop when the starter nod is once again reach
		if np.sum(current == starter) == 2:
			stop = True

	return sort

def complexe_it(path, mind, divd=4):
	"""
	Function to break the line witch length is superior to mind.

	Parameters
	----------
	path : np.ndarray
		A 2-dimensions numpy array witch countain the positions of the dots
		forming the convexe envelope.
	mind : float
		Maximum length above which the segments will be broken.
	divd : int|float, optional
		Value that divide the distance between the new position and the
		curve. Higger it will be, lowwer the distance between the new
		position and the curve will be. The default is 4. It have to be > 0.

	Returns
	-------
	course : np.ndarray
		A 2-dimensions numpy array witch countain the positions of the dots
		forming the racetrack.

	Example
	-------
	[In 0] : path = np.array([[0.01461559, 0.39027198],
							  [0.2184884 , 0.76162714],
							  [0.87422724, 0.47360825],
							  [0.7539203 , 0.27809264],
							  [0.2086457 , 0.1112382 ],
							  [0.01461559, 0.39027198]])
	[In 1] : complexe_it(path, 0.5, divd=4)
	[Out 1]: [[ 0.01461559, 0.39027198], [0.2184884 ,  0.76162714],
			  [0.38535037 , 0.25104854], [0.87422724,  0.47360825],
			  [0.7539203  , 0.27809264], [0.57455747, -0.11015235],
			  [0.2086457  , 0.1112382 ], [0.01461559,  0.39027198]]

	"""
	course = []
	for i in range(len(path)-1):
		course.append(list(path[i]))
		dist = ((path[i+1, 0] - path[i, 0])**2+
				(path[i+1, 1] - path[i, 1])**2)**0.5
		if dist >= mind:
			r1 = dist/2 + dist/divd
			r2 = dist/2 + dist/divd
			if (path[i+1, 1]-path[i, 1]) != 0:
				a = (-path[i, 0]**2 - path[i, 1]**2 + path[i+1, 0]**2 +
					  path[i+1, 1]**2 + r1**2 - r2**2)/(
							2 * (path[i+1, 1] - path[i, 1]))

				d = (path[i+1, 0]-path[i, 0])/(path[i+1, 1]-path[i, 1])
				a_par = d**2 + 1
				b_par = -2*path[i, 0] +2*path[i, 1]*d -2*a*d
				c_par = (path[i, 0]**2 + path[i, 1]**2 - 2*path[i, 1]*a 
						+ a**2 -r1**2)
				delta = b_par**2 - 4*a_par*c_par
				half_x1 = (-b_par-np.sqrt(delta))/(2*a_par)
				half_y1 = a-half_x1*d
				half_x2 = (-b_par+np.sqrt(delta))/(2*a_par)
				half_y2 = a-half_x2*d

			elif ((path[i, 1]-path[i+1, 1]) == 0)&(
							(path[i, 0]-path[i+1, 0]) !=0 ):

				half_x1 = -(r1**2 -r2**2 -path[i, 0]**2 +path[i+1, 0]**2)/(
							2*(path[i, 0]-path[i+1, 0]))

				half_x2 = -(r1**2 -r2**2 -path[i, 0]**2 +path[i+1, 0]**2)/(
							2*(path[i, 0]-path[i+1, 0]))

				a_par = 1
				b_par = -2*path[i, 1]
				c_par = path[i, 1]**2 + (half_x1-path[i, 0])**2 -r1**2
				delta = b_par**2 - 4*a_par*c_par
				half_y1 = (-b_par - np.sqrt(delta))/(2 * a_par)
				half_y2 = (-b_par + np.sqrt(delta))/(2 * a_par)

			sol = [[half_x1, half_y1],
					[half_x2, half_y2]][np.random.randint(0, 2)]

			course.append(sol)

		else:
			pass

	course.append(list(path[-1]))
	return course

def make_hull_circuit(size, n_ini_rand, n_interp, max_edge_len, pccent,
					  denom_div, kernel_size, wid):
	"""
	Function to create random racetrack from the convex envelope of a random
	cloud of dots.

	Parameters
	----------
	size : int
		Size of the final square map (number of cells).
	n_ini_rand : int
		Number of random dots from witch the convex hull will be calculated.
	n_interp : int
		Number of dots generated for each segments during interpolation.
	max_edge_len : float
		Length above which the lines are broken into new edges.
	pccent : float
		Percentage of centering of the circuit. The stronger it is, the
		less the circuit will approach the edges.
	denom_div : float
		Denominator of adding additional distance at the break of new edges.
	kernel_size : int
		Size of the smoothing core.
	wid : int
		Width of the road in pixel on the final map. wid have to be > 1.

	Returns
	-------
	road : np.ndarray
		Map of the final racetrack in 3-dimensions numpy.array. The dimensions
		are (high, width, rgb code).

	Example
	-------
	[In 0] : make_hull_circuit(150, 15, 100, 5, 0.18, 25, 37, 5)

	"""
	road = np.zeros((size, size, 3))
	road[:, :, 1] = .8
	dots = np.random.uniform(size*pccent, size-size*pccent, (n_ini_rand, 2))
	hull = jarvis_walk(dots)
	hull = complexe_it(np.array(hull), max_edge_len, denom_div)
	hull = np.array(hull+[hull[1]])
	course = smooth_circuit(hull, n_interp, kernel_size).astype(int)
	lw1 = wid//2
	if wid%2 == 0:
		lw2 = wid//2
	else:
		lw2 = wid//2 + 1

	for i in range(len(course)):
		road[course[i, 0]-lw1:course[i, 0]+lw2,
			 course[i, 1]-lw1:course[i, 1]+lw2] = [.4, .4, .4]

	center_road = np.argwhere(road[:, :, 1] == .8)
	for i in range(len(center_road)):
		area = road[center_road[i, 0]-2:center_road[i, 0]+3,
					center_road[i, 1]-2:center_road[i, 1]+3]
		if np.sum(area[:, :, 1] == .4):
			road[center_road[i, 0], center_road[i, 1]] = [.6, 0, 0]

	return road
