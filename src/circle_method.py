# -*- coding: utf-8 -*-
"""
First method to create a random racetrak: break a cicle into curves.
"""

import numpy as np
from general import smooth_circuit
#=============================================================================
def make_broke_circle_circuit(d_angle, noise_lvl, n_iter, kernel_size, width,
							  pcent, wid):
	"""
	Function to creat a random racetrack from a circle.

	Parameters
	----------
	d_angle : int|float
		Spacing in degrees between each circle break.
	noise_lvl : float
		Standard deviation of the centered normal noise that changes the
		distance to the center of the initial points that make up the
		polygon.
	n_iter : int
		Number of points generated for each segments during interpolation.
	kernel_size : int
		Size of the smoothing kernel.
	width : int
		Size of the final square map (number of cells).
	pcent : float
		Percentage of centering of the circuit. The stronger the circuit,
		the less the circuit will approach the edges.
	wid : int
		Width of the road of the racetrack. wid have to be > 1.

	Returns
	-------
	color_map : np.ndarray
		Map of the final racetrack in 3-dimensions numpy.array. The
		dimensions are (high, width, rgb code).

	Example
	-------
	[In 0] : make_broke_circle_circuit(20, 0.4, 100, 81, 250, 0.1, 5)

	"""
	teta = np.deg2rad(np.arange(0, 360, d_angle))
	rayon = 1+np.random.normal(0, noise_lvl, len(teta))
	circle = np.array([np.cos(teta)*rayon, np.sin(teta)*rayon]).T
	circle = np.concatenate((circle, circle[:2]))
	course = smooth_circuit(circle, n_iter, kernel_size)
	carte = np.zeros((width, width))
	course[:, 0] = (course[:,0]-np.min(course[:,0]))/np.max(
										course[:,0]-np.min(course[:,0]))

	course[:, 1] = (course[:,1]-np.min(course[:,1]))/np.max(
										course[:,1]-np.min(course[:,1]))

	course[:, 0] = (course[:, 0]*int(width - width*pcent)) +
					int(width*(pcent/2))

	course[:, 1] = (course[:, 1]*int(width - width*pcent)) +
					int(width*(pcent/2))

	course = course.astype(int)
	if wid%2 == 0:
		lw1 = int(wid/2)
		lw2 = int(wid/2)
	else:
		lw1 = int(wid/2)
		lw2 = int(wid/2)+1

	for u in range(len(course)):
		carte[course[u, 0]-lw1:course[u, 0]+lw2,
			  course[u, 1]-lw1:course[u, 1]+lw2] = 1

	color_map = np.zeros((width, width, 3))
	color_map[carte == 0] = [0, 0.8, 0]
	color_map[carte == 1] = [.4, .4, .4]
	lok = np.argwhere(carte == 0)
	for i in range(len(lok)):
		area = carte[lok[i, 0]-lw1:lok[i, 0]+lw2,
					 lok[i, 1]-lw1:lok[i, 1]+lw2]

		if np.sum(area) > 0:
			color_map[lok[i, 0], lok[i, 1]] = [.6, 0, 0]

	return color_map
