# -*- coding: utf-8 -*-
"""
@author: Matthieu Nougaret

This script countain functions to create random racetrack.

Sources / Bibliography:
(1) :
 https://www.gamedeveloper.com/programming/generating-procedural-racetracks
(2) : https://fr.wikipedia.org/wiki/Marche_de_Jarvis

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot
import convexe_method
import circle_method
import exploration_method
import warnings
#=============================================================================
way = [2]

if 1 in way:
	# Spacing in degrees between each circle break.
	dgl = 20
	# Standard deviation of the centered normal noise that changes the
	# distance to the center of the initial points that make up the
	# polygon.
	err = 0.4
	# Number of points generated for each segments during interpolation.
	ni = 100
	# Size of the smoothing kernel.
	ks = 81
	# Percentage of centering of the circuit. The stronger the circuit,
	# the less the circuit will approach the edges.
	long = 250
	# Size of the final square map (number of cells).
	centr = 0.1
	# Width of the road of the racetrack. wid have to be > 1.
	larg = 5

	for i in range(10):
		Road = circle_method.make_broke_circle_circuit(dgl, err, ni, ks,
													   long, centr, larg)

		plt.figure(figsize=(7, 7))
		plt.imshow(Road)
		plt.axis('off')
		plt.show()

if 2 in way:
	warnings.filterwarnings("ignore")
	# Size of the final square map
	sc = 25
	# Number of random dots from witch the convex hull will be calculated.
	nr = 5
	# Number of dots generated for each segments during interpolation.
	ni = 100
	# Length above which the lines are broken.
	mxl = 1
	# Percentage of centering of the circuit.
	ct = 0.18
	# Denominator of adding additional distance at the break of new edges.
	dv = 12
	# Size of the smoothing core.
	ks = 37
	# Width of the road in pixel on the final map. Have to be > 1.
	lg = 3

	for cir in range(10):
		road = convexe_method.make_hull_circuit(sc, nr, ni, mxl, ct,
												dv, ks, lg)

		plt.figure(figsize=(7, 7))
		plt.imshow(road)
		plt.axis('off')
		plt.show()

		print(road, '\n')

if 3 in way:
	# Shape of the map that will be used for the exploration stage.
	forme = (6, 6)
	# Number of dots that are adds during the smothing of th path.
	n_iter = 100
	# Size of the smoothing core.
	kern_sz = 103
	# Size of the final square map (number of cells).
	width = 250
	# Width of the road in pixel on the final map. wid have to be > 1.
	larg = 5

	for cir in range(10):
		# Initial position during the eploration stage.
		depart = np.array([np.random.randint(0, Forme[0]),
						   np.random.randint(0, Forme[1])])

		smt = exploration_method.make_explorer_circuit(forme, depart, n_iter,
													   kern_sz, width, larg)

		plt.figure(figsize=(6, 6))
		plt.imshow(smt)
		plt.axis('off')
		plt.show()
