# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 18:40:30 2022

@author: Matthieu Nougaret

This script countain functions to create random racetrack.

Sources / Bibliography:
(1) :
 http://blog.meltinglogic.com/2013/12/how-to-generate-procedural-racetracks/
(2) : https://fr.wikipedia.org/wiki/Marche_de_Jarvis

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot
import ConvexeMethod
import CircleMethod
import ExplorationMethod
import warnings
#=============================================================================

way = [2]

if 1 in way:
	larg = 5 ; centr = 0.1 ; long = 250 ; dgl = 20 ; err = 0.4
	ni = 100 ; ks = 81
	for i in range(10):
		Road = CircleMethod.MakeBroke_circleCircuit(dgl, err, ni, ks, long,
											  centr, larg)
		plt.figure(figsize=(7, 7))
		plt.imshow(Road)
		plt.axis('off')
		plt.show()

#http://blog.meltinglogic.com/2013/12/how-to-generate-procedural-racetracks/
if 2 in way:
	warnings.filterwarnings("ignore")
	sc = 250 ; nr = 16 ; ni = 100 ; mxl = 50
	ct = 0.18 ; dv = 12 ; ks = 67 ; lg = 5
	for cir in range(10):
		Road = ConvexeMethod.MakeHullCircuit(sc, nr, ni, mxl, ct, dv, ks, lg)
		plt.figure(figsize=(7, 7))
		plt.imshow(Road)
		plt.axis('off')
		plt.show()

if 3 in way:
	for cir in range(10):
		Forme = [6, 6] ; Sz = 250 ; larg = 5 ; ni = 100 ; ks = 103
		Depart = np.array([np.random.randint(0, Forme[0]),
						   np.random.randint(0, Forme[1])])
		smt = ExplorationMethod.MakeExplorerCircuit(Forme, Depart, ni, ks, Sz,
											  larg)

		plt.figure(figsize=(6, 6))
		plt.imshow(smt)
		plt.axis('off')
		plt.show()
