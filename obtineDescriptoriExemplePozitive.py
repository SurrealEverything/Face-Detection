#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 15:17:55 2018

@author: gabriel
"""
import numpy as np
import cv2
import pathlib


def obtineDescriptoriExemplePozitive(parametri):
    """
    descriptoriExemplePozitive = matrice NxD, unde:
      N = numarul de exemple pozitive de antrenare (fete de oameni)
      D = numarul de dimensiuni al descriptorului
    """
    imgPathsUnsorted = pathlib.Path(
            parametri.numeDirectorExemplePozitive).glob('*.jpg')
    imgPaths = sorted([x for x in imgPathsUnsorted])

    numarImagini = len(imgPaths)

    dimensiuneDescriptoriImagine = round(
            parametri.orientari
            * (parametri.dimensiuneBloc[0] /
               parametri.dimensiuneCelulaHOG[0]) ** 2
            * ((parametri.dimensiuneFereastra[0] - parametri.dimensiuneBloc[0])
                / parametri.pasBloc[0] + 1) ** 2)

    descriptoriExemplePozitive = np.zeros((
            parametri.numarExemplePozitive,
            dimensiuneDescriptoriImagine))

    hog = cv2.HOGDescriptor(parametri.dimensiuneFereastra,
                            parametri.dimensiuneBloc,
                            parametri.pasBloc,
                            parametri.dimensiuneCelulaHOG,
                            parametri.orientari)

    print('Exista un numar de exemple pozitive = ' + str(numarImagini))

    for idx in range(numarImagini):
        print('Procesam exemplul pozitiv numarul ' + str(idx))

        img = cv2.imread(str(imgPaths[idx]), 0)

        desc = hog.compute(img)

        descriptoriExemplePozitive[idx] = np.ravel(desc)

    return descriptoriExemplePozitive
