#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 15:18:16 2018

@author: gabriel
"""
import numpy as np
import cv2
import math
import pathlib
import os


def obtineDescriptoriExempleNegative(parametri):
    """
    descriptoriExempleNegative = matrice MxD, unde:
      M = numarul de exemple negative de antrenare (NU sunt fete de oameni),
      D = numarul de dimensiuni al descriptorului
    """
    imgPathsUnsorted = pathlib.Path(
            parametri.numeDirectorExempleNegative).glob('*.jpg')
    imgPaths = sorted([x for x in imgPathsUnsorted])

    numarImagini = len(imgPaths)

    dimensiuneDescriptoriImagine = round(
            parametri.orientari
            * (parametri.dimensiuneBloc[0] /
               parametri.dimensiuneCelulaHOG[0]) ** 2
            * ((parametri.dimensiuneFereastra[0] - parametri.dimensiuneBloc[0])
                / parametri.pasBloc[0] + 1) ** 2)

    descriptoriExempleNegative = np.zeros((
            parametri.numarExempleNegative,
            dimensiuneDescriptoriImagine))

    descriptorNul = np.zeros((dimensiuneDescriptoriImagine,),
                             np.float64)

    hog = cv2.HOGDescriptor(parametri.dimensiuneFereastra,
                            parametri.dimensiuneBloc,
                            parametri.pasBloc,
                            parametri.dimensiuneCelulaHOG,
                            parametri.orientari)

    print('Exista un numar de imagini = ' + str(numarImagini)
          + ' ce contin numai exemple negative')

    contor = 0
    numarFerestreImagine = math.ceil(parametri.numarExempleNegative
                                     / numarImagini)
    for idx in range(numarImagini):
        print('Procesam imaginea numarul ' + str(idx))

        img = cv2.imread(str(imgPaths[idx]), 0)

        h, w = img.shape

        for idxFereastra in range(numarFerestreImagine):

            randY = np.random.randint(0, h-parametri.dimensiuneFereastra[0]+1)
            randX = np.random.randint(0, w-parametri.dimensiuneFereastra[0]+1)

            fereastra = img[randY:randY + parametri.dimensiuneFereastra[0],
                            randX:randX + parametri.dimensiuneFereastra[0]]

            desc = hog.compute(fereastra)

            descriptoriExempleNegative[contor] = np.ravel(desc)

            # afiseaza numele imaginilor care genereaza descriptori nuli
            if np.array_equal(descriptoriExempleNegative[contor],
                              descriptorNul):
                numeImg = os.path.basename(os.path.normpath(imgPaths[idx]))
                print('Imaginea ' + numeImg + ' a generat un descriptor nul')

            contor += 1

            if contor == parametri.numarExempleNegative:
                return descriptoriExempleNegative
